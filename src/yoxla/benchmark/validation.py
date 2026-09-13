"""
Benchmark validation.

Validation runs before any inference. A benchmark run that silently
starts against changed or broken data is worse than one that refuses
to start, especially when the model is billed per token.
"""

from dataclasses import dataclass, field
from typing import Any

from yoxla.benchmark.loaders import (
    load_rows,
    row_to_example,
    rows_fingerprint,
)
from yoxla.benchmark.manifest import load_manifest
from yoxla.benchmark.prompting import render_prompt
from yoxla.benchmark.schema import (
    AnswerMode,
    BenchmarkExample,
    BenchmarkTask,
)


@dataclass
class TaskValidation:
    task_id: str

    rows: int
    expected_rows: int

    fingerprint: str

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    examples: list[BenchmarkExample] = field(
        default_factory=list
    )

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_task(
    task: BenchmarkTask,
    rows: list[dict[str, Any]] | None = None,
    revision: str | None = None,
    token: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> TaskValidation:
    if rows is None:
        rows = load_rows(
            task,
            revision=revision,
            token=token,
        )

    result = TaskValidation(
        task_id=task.task_id,
        rows=len(rows),
        expected_rows=task.expected_examples,
        fingerprint=rows_fingerprint(rows),
    )

    if len(rows) != task.expected_examples:
        result.errors.append(
            f"expected {task.expected_examples} rows, "
            f"found {len(rows)}"
        )

    _validate_columns(task, rows, result)

    if result.errors:
        return result

    examples = [
        row_to_example(task, row) for row in rows
    ]

    _validate_ids(examples, result)
    _validate_gold(task, examples, result)
    _validate_prompts(task, examples, result)
    _validate_manifest(task, result, manifest)

    result.examples = examples

    return result


def _validate_columns(
    task: BenchmarkTask,
    rows: list[dict[str, Any]],
    result: TaskValidation,
) -> None:
    if not rows:
        result.errors.append("dataset is empty")
        return

    missing = [
        column
        for column in task.required_columns()
        if column not in rows[0]
    ]

    if missing:
        result.errors.append(
            f"missing columns: {', '.join(missing)}"
        )


def _validate_ids(
    examples: list[BenchmarkExample],
    result: TaskValidation,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for example in examples:
        if not example.id:
            result.errors.append("found an empty example id")

        if example.id in seen:
            duplicates.add(example.id)

        seen.add(example.id)

    if duplicates:
        listed = ", ".join(sorted(duplicates)[:5])

        result.errors.append(
            f"duplicate ids ({len(duplicates)}): {listed}"
        )


def _validate_gold(
    task: BenchmarkTask,
    examples: list[BenchmarkExample],
    result: TaskValidation,
) -> None:
    mode = task.answer.mode

    allowed = set(task.answer.choice_values())

    invalid_choice: list[str] = []
    out_of_range: list[str] = []
    empty_gold: list[str] = []

    for example in examples:
        gold = example.gold

        if gold is None:
            if not task.gold_nullable:
                empty_gold.append(example.id)

            continue

        if mode is AnswerMode.CHOICE:
            if str(gold) not in allowed:
                invalid_choice.append(example.id)

        elif mode is AnswerMode.NUMERIC:
            try:
                value = float(gold)

            except (TypeError, ValueError):
                out_of_range.append(example.id)
                continue

            minimum = task.answer.minimum
            maximum = task.answer.maximum

            if (
                minimum is not None and value < minimum
            ) or (
                maximum is not None and value > maximum
            ):
                out_of_range.append(example.id)

        elif isinstance(gold, (list, tuple)):
            # A text answer may arrive as the list of its accepted
            # surface forms. It is empty only when none of them is.
            if not any(
                str(form).strip() for form in gold
            ):
                empty_gold.append(example.id)

        elif not str(gold).strip():
            empty_gold.append(example.id)

    if invalid_choice:
        result.errors.append(
            f"{len(invalid_choice)} gold labels outside the "
            f"declared answer space "
            f"(first: {invalid_choice[0]})"
        )

    if out_of_range:
        result.errors.append(
            f"{len(out_of_range)} gold values outside "
            f"[{task.answer.minimum}, {task.answer.maximum}] "
            f"(first: {out_of_range[0]})"
        )

    if empty_gold:
        result.errors.append(
            f"{len(empty_gold)} empty gold values "
            f"(first: {empty_gold[0]})"
        )

    _validate_evaluator_gold(task, examples, result)


def _validate_evaluator_gold(
    task: BenchmarkTask,
    examples: list[BenchmarkExample],
    result: TaskValidation,
) -> None:
    """
    Whatever the evaluator needs from a gold beyond the generic rules.

    A checks-list gold is well-formed or it is not, and only the
    evaluator that consumes it knows which - so it is asked rather
    than second-guessed here.
    """

    from yoxla.benchmark.evaluators import create_evaluator

    evaluator = create_evaluator(task)

    problems: list[str] = []

    for example in examples:
        problems.extend(evaluator.validate_gold(example))

    if problems:
        result.errors.append(
            f"{len(problems)} gold values rejected by the "
            f"'{task.evaluator}' evaluator "
            f"(first: {problems[0]})"
        )


def _validate_prompts(
    task: BenchmarkTask,
    examples: list[BenchmarkExample],
    result: TaskValidation,
) -> None:
    for example in examples:
        try:
            rendered = render_prompt(task, example)

        except ValueError as exc:
            result.errors.append(str(exc))
            return

        if not rendered.user.strip():
            result.errors.append(
                f"empty user prompt for example {example.id}"
            )

            return


def _validate_manifest(
    task: BenchmarkTask,
    result: TaskValidation,
    manifest: dict[str, Any] | None,
) -> None:
    if manifest is None:
        manifest = load_manifest(task.block)

    if manifest is None:
        return

    entry = next(
        (
            item
            for item in manifest.get("tasks", [])
            if item.get("task_id") == task.task_id
        ),
        None,
    )

    if entry is None:
        result.warnings.append(
            "task is not present in the frozen benchmark manifest"
        )

        return

    if entry.get("fingerprint") != result.fingerprint:
        result.warnings.append(
            "data differs from the frozen benchmark manifest - "
            "results are not comparable with published runs"
        )


def validate_tasks(
    tasks: list[BenchmarkTask],
    revision: str | None = None,
    token: str | None = None,
) -> list[TaskValidation]:
    return [
        validate_task(
            task,
            revision=revision,
            token=token,
        )
        for task in tasks
    ]
