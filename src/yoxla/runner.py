"""
Benchmark runner.

The runner is generic: it resolves tasks from the registry, renders
their prompts, calls the model adapter, parses and evaluates the
answer, and writes the artifacts. It contains no task-specific
branches, so a new task is a registry entry, not a runner change.

Two rules are enforced here:

* one benchmark example is one model attempt - a badly formatted
  answer is a result, never a reason to re-prompt;
* retries exist only for transport failures, never for bad content.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yoxla import __version__
from yoxla.benchmark.evaluators import create_evaluator
from yoxla.benchmark.loaders import (
    load_examples,
    load_rows,
    row_to_example,
    rows_fingerprint,
)
from yoxla.benchmark.manifest import load_manifest
from yoxla.benchmark.parsing import ParsedPrediction, parse_prediction
from yoxla.benchmark.prompting import (
    render_prompt,
    render_system_prompt,
    text_hash,
)
from yoxla.benchmark.registry import BENCHMARK_VERSIONS, get_task
from yoxla.benchmark.schema import BenchmarkExample, BenchmarkTask
from yoxla.benchmark.validation import validate_task
from yoxla.inference import GenerationRequest
from yoxla.results.schema import (
    PredictionRecord,
    RunInfo,
    TaskRunInfo,
)
from yoxla.results.writer import (
    ResultWriter,
    build_run_id,
    utc_now,
)
from yoxla.scoring import (
    compute_benchmark_score,
    compute_block_score,
    compute_task_score,
    group_task_scores,
)

RETRYABLE_STATUS_CODES = {
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
}

RETRYABLE_EXCEPTION_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "ConnectionError",
    "Timeout",
    "ReadTimeout",
}


class BenchmarkValidationError(RuntimeError):
    pass


@dataclass
class LoadedTask:
    task: BenchmarkTask

    examples: list[BenchmarkExample]

    fingerprint: str

    warnings: list[str] = field(default_factory=list)


# ==================================================================
# Loading
# ==================================================================


def prepare_tasks(
    tasks: list[BenchmarkTask],
    revision: str | None = None,
    token: str | None = None,
    validate: bool = True,
    limit: int | None = None,
    verbose: bool = True,
) -> list[LoadedTask]:
    """
    Download and check every task before any inference starts.

    Failing here costs nothing; failing halfway through a paid run
    costs the whole run.
    """

    loaded: list[LoadedTask] = []

    failures: list[str] = []

    for task in tasks:
        rows = load_rows(
            task,
            revision=revision,
            token=token,
        )

        fingerprint = rows_fingerprint(rows)

        warnings: list[str] = []

        if validate:
            result = validate_task(
                task,
                rows=rows,
                manifest=load_manifest(task.block),
            )

            warnings = result.warnings

            if not result.ok:
                for message in result.errors:
                    failures.append(
                        f"{task.task_id}: {message}"
                    )

                if verbose:
                    print(
                        f"  {task.task_id:<20} "
                        f"{len(rows)}/{task.expected_examples} "
                        f"FAILED"
                    )

                continue

            examples = result.examples

        else:
            examples = [
                row_to_example(task, row) for row in rows
            ]

        if verbose:
            status = "OK" if not warnings else "OK (warnings)"

            print(
                f"  {task.task_id:<20} "
                f"{len(examples)}/{task.expected_examples} "
                f"{status}"
            )

            for warning in warnings:
                print(f"      warning: {warning}")

        if limit is not None:
            examples = examples[:limit]

        loaded.append(
            LoadedTask(
                task=task,
                examples=examples,
                fingerprint=fingerprint,
                warnings=warnings,
            )
        )

    if failures:
        raise BenchmarkValidationError(
            "Benchmark validation failed:\n"
            + "\n".join(f"  - {item}" for item in failures)
        )

    return loaded


# ==================================================================
# Inference with retries
# ==================================================================


def is_retryable(exception: Exception) -> bool:
    """
    Only transport-level failures may be retried.

    A malformed or wrong model answer is a benchmark result and must
    never be given a second attempt.
    """

    status_code = getattr(exception, "status_code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    if isinstance(
        exception,
        (TimeoutError, ConnectionError),
    ):
        return True

    return (
        type(exception).__name__
        in RETRYABLE_EXCEPTION_NAMES
    )


def generate_with_retry(
    model,
    request: GenerationRequest,
    max_retries: int = 4,
    backoff: float = 2.0,
):
    attempt = 0

    while True:
        try:
            return model.generate(request)

        except Exception as exception:
            if attempt >= max_retries or not is_retryable(
                exception
            ):
                raise

            delay = backoff * (2**attempt)

            print(
                f"      retry {attempt + 1}/{max_retries} "
                f"after {type(exception).__name__} "
                f"({delay:.0f}s)"
            )

            time.sleep(delay)

            attempt += 1


# ==================================================================
# Running
# ==================================================================


def restore_prediction(
    record: dict[str, Any],
) -> ParsedPrediction:
    """
    Rebuild a parsed prediction from a stored record so that a resumed
    run scores exactly like an uninterrupted one.
    """

    return ParsedPrediction(
        raw=record.get("raw_prediction") or "",
        value=record.get("parsed_prediction"),
        valid=bool(record.get("prediction_valid")),
        error=record.get("parse_error"),
        abstained=bool(
            record.get("prediction_abstained")
        ),
    )


def dump_provider_response(response) -> Any:
    """
    Serialize the provider payload behind an unusable answer.

    Kept small on purpose: it is stored only for answers that failed
    to parse, where an empty ``text`` field is not enough to tell a
    reasoning channel from a refusal.
    """

    raw = response.raw_response

    if raw is None:
        return None

    if isinstance(raw, (dict, list, str, int, float, bool)):
        return raw

    for method in ("model_dump", "to_dict", "dict"):
        dump = getattr(raw, method, None)

        if callable(dump):
            try:
                return dump()

            except Exception:  # noqa: BLE001, S112
                continue

    return str(raw)[:4000]


def run_benchmark(
    model,
    tasks: list[BenchmarkTask],
    output_dir: str = "runs",
    run_id: str | None = None,
    limit: int | None = None,
    revision: str | None = None,
    token: str | None = None,
    resume: bool = False,
    validate: bool = True,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
    workers: int = 1,
) -> dict[str, Any]:
    run_id = run_id or build_run_id(
        model=model.model,
        provider=model.provider,
    )

    print(f"Model:    {model.model}")
    print(f"Provider: {model.provider}")
    print(f"Run:      {run_id}\n")

    print(f"Loading {len(tasks)} tasks...")

    loaded_tasks = prepare_tasks(
        tasks,
        revision=revision,
        token=token,
        validate=validate,
        limit=limit,
    )

    total_examples = sum(
        len(item.examples) for item in loaded_tasks
    )

    print(f"\nRunning {total_examples} examples...\n")

    started_at = utc_now()

    run_info = RunInfo(
        run_id=run_id,
        benchmark="YOXLA",
        benchmark_versions={
            item.task.block: BENCHMARK_VERSIONS.get(
                item.task.block,
                item.task.block,
            )
            for item in loaded_tasks
        },
        yoxla_version=__version__,
        model=model.model,
        provider=model.provider,
        workers=workers,
        started_at=started_at,
    )

    task_scores = []

    generation_errors = 0

    with ResultWriter(output_dir, run_id) as writer:
        completed = (
            writer.load_completed() if resume else {}
        )

        if resume:
            # A transport failure is not a model attempt: the example
            # never received an answer, so a resumed run must ask
            # again instead of keeping the zero. Only real answers -
            # including badly formatted ones - are final.
            failed = [
                key
                for key, record in completed.items()
                if record.get("error")
            ]

            for key in failed:
                del completed[key]

            if failed:
                print(
                    f"Retrying {len(failed)} examples that "
                    f"failed with provider errors\n"
                )

                _write_predictions(
                    writer.directory,
                    completed,
                )

        writer.open_predictions(resume=resume)

        for index, item in enumerate(loaded_tasks, start=1):
            task = item.task

            print(
                f"[{index}/{len(loaded_tasks)}] "
                f"{task.task_id} "
                f"({len(item.examples)} examples)"
            )

            run_info.tasks.append(
                TaskRunInfo(
                    task_id=task.task_id,
                    block=task.block,
                    dataset_repo=task.dataset_repo,
                    dataset_config=task.dataset_config,
                    dataset_split=task.dataset_split,
                    dataset_revision=revision,
                    dataset_fingerprint=item.fingerprint,
                    expected_examples=task.expected_examples,
                    loaded_examples=len(item.examples),
                    prompt_version=task.prompt.version,
                    system_prompt_hash=text_hash(
                        render_system_prompt(task)
                    ),
                    generation={
                        "max_output_tokens": (
                            task.generation.max_output_tokens
                        ),
                        "temperature": (
                            task.generation.temperature
                        ),
                        "top_p": task.generation.top_p,
                        "thinking": (
                            task.generation.thinking
                        ),
                    },
                    warnings=item.warnings,
                )
            )

            evaluator = create_evaluator(task)

            errors = _run_task(
                model=model,
                task=task,
                examples=item.examples,
                evaluator=evaluator,
                writer=writer,
                run_id=run_id,
                completed=completed,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                workers=workers,
            )

            generation_errors += errors

            metrics = evaluator.compute()

            task_score = compute_task_score(task, metrics)

            task_scores.append(task_score)

            print(
                f"  -> {task_score.primary_metric}: "
                f"{_format_metric(task_score.primary_value)}"
                f"   score: {task_score.score:.2f}\n"
            )

        run_info.finished_at = utc_now()

        writer.write_run_info(run_info)

        writer.write_task_scores(
            {
                task_score.task_id: task_score.to_dict()
                for task_score in task_scores
            }
        )

        block_scores = [
            compute_block_score(block, scores)
            for block, scores in group_task_scores(
                task_scores
            ).items()
        ]

        benchmark_score = compute_benchmark_score(
            block_scores
        )

        summary = {
            "run_id": run_id,
            "model": model.model,
            "provider": model.provider,
            "yoxla_version": __version__,
            "benchmark_versions": (
                run_info.benchmark_versions
            ),
            "dataset_revision": revision,
            "started_at": started_at,
            "finished_at": run_info.finished_at,
            "examples": total_examples,
            "limit": limit,
            "partial": limit is not None,
            "generation_errors": generation_errors,
            "task_scores": {
                task_score.task_id: task_score.score
                for task_score in task_scores
            },
            "blocks": {
                block.block: block.score
                for block in block_scores
            },
            "score": benchmark_score.score,
        }

        writer.write_summary(summary)

        _print_summary(summary, task_scores)

        print(f"Results saved to: {writer.directory}")

    return summary


def rescore_run(
    run_directory: str | Path,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """
    Recompute the scores of a finished run from its stored answers.

    Raw model output is never discarded, so an evaluator or metric
    change can be applied to past runs without paying for inference
    again.
    """

    directory = Path(run_directory)

    run_info = json.loads(
        (directory / "run.json").read_text(
            encoding="utf-8"
        )
    )

    summary = json.loads(
        (directory / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    writer = ResultWriter(
        directory.parent,
        directory.name,
    )

    records = writer.load_completed()

    print(f"Rescoring {directory.name}")
    print(f"Model:    {summary.get('model')}\n")

    task_scores = []

    updated: dict[tuple[str, str], dict[str, Any]] = {}

    stale = []

    for entry in run_info["tasks"]:
        task = get_task(entry["task_id"])

        recorded = entry.get("system_prompt_hash")

        current = text_hash(render_system_prompt(task))

        if recorded and recorded != current:
            stale.append(
                f"{task.task_id}: answers were produced under "
                f"prompt {recorded[:12]}, the task now asks "
                f"{current[:12]}"
            )

    if stale:
        raise ValueError(
            "This run cannot be rescored - its answers were given to "
            "a different question:\n  "
            + "\n  ".join(stale)
            + "\n\nRe-run the model instead. Rescoring replays "
            "stored answers through new evaluators, not through a "
            "new prompt."
        )

    for entry in run_info["tasks"]:
        task = get_task(entry["task_id"])

        examples = load_examples(
            task,
            revision=(
                revision
                or entry.get("dataset_revision")
            ),
            token=token,
        )

        evaluator = create_evaluator(task)

        missing = 0

        for example in examples:
            record = records.get(
                (task.task_id, example.id)
            )

            if record is None:
                missing += 1
                continue

            metrics = evaluator.add(
                example,
                restore_prediction(record),
            )

            record["metrics"] = metrics

            updated[(task.task_id, example.id)] = record

        task_score = compute_task_score(
            task,
            evaluator.compute(),
        )

        task_scores.append(task_score)

        print(
            f"  {task.task_id:<22} "
            f"{task_score.primary_metric:<18} "
            f"{task_score.score:6.2f}"
            + (
                f"   ({missing} examples missing)"
                if missing
                else ""
            )
        )

    _write_predictions(directory, updated)

    writer.write_task_scores(
        {
            task_score.task_id: task_score.to_dict()
            for task_score in task_scores
        }
    )

    block_scores = [
        compute_block_score(block, scores)
        for block, scores in group_task_scores(
            task_scores
        ).items()
    ]

    benchmark_score = compute_benchmark_score(block_scores)

    summary.update(
        {
            "task_scores": {
                task_score.task_id: task_score.score
                for task_score in task_scores
            },
            "blocks": {
                block.block: block.score
                for block in block_scores
            },
            "score": benchmark_score.score,
            "rescored_at": utc_now(),
            "rescored_with_yoxla_version": __version__,
        }
    )

    writer.write_summary(summary)

    _print_summary(summary, task_scores)

    return summary


def _write_predictions(
    directory: Path,
    records: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """
    Rewrite predictions.jsonl with updated per-example metrics.

    Written to a temporary file first so that an interrupted rescore
    cannot destroy the stored model answers.
    """

    target = directory / "predictions.jsonl"

    temporary = directory / "predictions.jsonl.tmp"

    with temporary.open("w", encoding="utf-8") as file:
        for record in records.values():
            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    temporary.replace(target)


def _run_task(
    model,
    task: BenchmarkTask,
    examples: list[BenchmarkExample],
    evaluator,
    writer: ResultWriter,
    run_id: str,
    completed: dict[tuple[str, str], dict[str, Any]],
    max_retries: int,
    retry_backoff: float,
    workers: int = 1,
) -> int:
    total = len(examples)

    generation_errors = 0

    pending: list[tuple[int, BenchmarkExample]] = []

    for position, example in enumerate(examples, start=1):
        previous = completed.get(
            (task.task_id, example.id)
        )

        if previous is not None:
            evaluator.add(
                example,
                restore_prediction(previous),
            )

            print(
                f"  [{position}/{total}] {example.id} "
                f"... cached"
            )

            continue

        pending.append((position, example))

    def ask(example: BenchmarkExample):
        """
        One request, and nothing else.

        Raises nothing: a provider failure is this example's result,
        not the run's, and with several threads in flight an
        exception escaping here would take the others with it.
        """

        rendered = render_prompt(task, example)

        request = GenerationRequest(
            messages=rendered.messages(),
            max_output_tokens=(
                task.generation.max_output_tokens
            ),
            temperature=task.generation.temperature,
            top_p=task.generation.top_p,
            thinking=task.generation.thinking,
        )

        try:
            response = generate_with_retry(
                model,
                request,
                max_retries=max_retries,
                backoff=retry_backoff,
            )

            return rendered, response, None

        except Exception as exception:  # noqa: BLE001
            return (
                rendered,
                None,
                f"{type(exception).__name__}: {exception}",
            )

    def record(
        position: int,
        example: BenchmarkExample,
        answered: tuple[Any, Any, str | None],
    ) -> int:
        """
        Score one answer and write its line. Called in example order
        on one thread, so the evaluator and the file see the run as
        if it had been sequential.
        """

        rendered, response, error = answered

        print(
            f"  [{position}/{total}] {example.id}",
            end=" ... ",
            flush=True,
        )

        if response is not None:
            prediction = parse_prediction(
                task,
                response.text,
            )

        else:
            prediction = ParsedPrediction(
                raw="",
                value=None,
                valid=False,
                error="generation_error",
            )

        metrics = evaluator.add(example, prediction)

        writer.write_prediction(
            PredictionRecord(
                run_id=run_id,
                task_id=task.task_id,
                example_id=example.id,
                gold=example.gold,
                raw_prediction=(
                    response.raw_text
                    if response is not None
                    else None
                ),
                parsed_prediction=prediction.value,
                prediction_valid=prediction.valid,
                prediction_abstained=prediction.abstained,
                parse_error=prediction.error,
                metrics=metrics,
                model=model.model,
                provider=model.provider,
                finish_reason=(
                    response.finish_reason
                    if response is not None
                    else None
                ),
                input_tokens=(
                    response.usage.input_tokens
                    if response is not None
                    else None
                ),
                output_tokens=(
                    response.usage.output_tokens
                    if response is not None
                    else None
                ),
                latency_ms=(
                    response.latency_ms
                    if response is not None
                    else None
                ),
                user_prompt_hash=rendered.user_hash,
                metadata=example.metadata,
                provider_response=(
                    dump_provider_response(response)
                    if response is not None
                    and not prediction.valid
                    else None
                ),
                error=error,
            )
        )

        print(_status(prediction, error))

        return 1 if error is not None else 0

    # Answers are consumed in example order rather than as they
    # finish. The pool runs ahead; each example is scored and written
    # when its own turn comes, so the predictions file reads the same
    # whatever the schedule was and a resumed run still picks up from
    # the last line on disk.
    if workers <= 1:
        for position, example in pending:
            generation_errors += record(
                position, example, ask(example)
            )

    else:
        with ThreadPoolExecutor(
            max_workers=workers
        ) as pool:
            answers = [
                (
                    position,
                    example,
                    pool.submit(ask, example),
                )
                for position, example in pending
            ]

            for position, example, answer in answers:
                generation_errors += record(
                    position, example, answer.result()
                )

    return generation_errors


def _status(
    prediction: ParsedPrediction,
    error: str | None,
) -> str:
    if error is not None:
        return f"ERROR: {error}"

    if not prediction.valid:
        return f"invalid ({prediction.error})"

    if prediction.abstained:
        return "abstain"

    value = str(prediction.value)

    if len(value) > 40:
        value = value[:37] + "..."

    return value


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.4f}"


def _print_summary(
    summary: dict[str, Any],
    task_scores,
) -> None:
    print("=" * 52)
    print("Summary")
    print("=" * 52)

    unparsed = []

    for task_score in task_scores:
        print(
            f"  {task_score.task_id:<22} "
            f"{task_score.primary_metric:<22} "
            f"{task_score.score:6.2f}"
        )

        rate = task_score.metrics.get("invalid_output_rate") or 0.0

        if rate:
            unparsed.append((task_score.task_id, rate))

    print("-" * 52)

    for block, score in summary["blocks"].items():
        print(f"  {block:<45} {score:6.2f}")

    print(f"\n  {'YOXLA score':<45} {summary['score']:6.2f}")

    if unparsed:
        # The parser refuses anything but the exact answer, so a zero
        # can mean the model was wrong or that it never answered in
        # the required format. Those are different findings about a
        # model and the score alone does not separate them.
        print(
            "\n  output that was not in the required format:"
        )

        for task_id, rate in unparsed:
            print(f"  {task_id:<45} {rate * 100:6.2f}%")

    if summary["generation_errors"]:
        print(
            f"\n  generation errors: "
            f"{summary['generation_errors']}"
        )

    if summary.get("partial"):
        # A --limit run exercises the pipeline, it does not measure a
        # model: rank correlations in particular are meaningless on a
        # handful of examples.
        print(
            f"\n  PARTIAL RUN: only {summary['limit']} examples "
            f"per task were used.\n"
            f"  These scores are not a benchmark result and must "
            f"not be compared with full runs."
        )

    print()
