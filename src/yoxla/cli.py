import argparse
import os
import sys

from yoxla import __version__
from yoxla.benchmark.manifest import write_manifest
from yoxla.benchmark.prompting import render_system_prompt, text_hash
from yoxla.benchmark.registry import (
    BENCHMARK_VERSIONS,
    list_blocks,
    resolve_tasks,
)
from yoxla.benchmark.validation import validate_task
from yoxla.inference import (
    GenerationRequest,
    Message,
    create_model,
    load_providers_config,
)
from yoxla.runner import rescore_run, run_benchmark
from yoxla.smoke_runner import run_dataset


def build_model_options(
    args: argparse.Namespace,
) -> dict:
    """
    Build runtime options for local model backends.
    """

    options = {}

    if getattr(
        args,
        "load_in_4bit",
        False,
    ):
        options["load_in_4bit"] = True

    if getattr(
        args,
        "load_in_8bit",
        False,
    ):
        options["load_in_8bit"] = True

    if getattr(
        args,
        "trust_remote_code",
        False,
    ):
        options["trust_remote_code"] = True

    device_map = getattr(
        args,
        "device_map",
        None,
    )

    if device_map is not None:
        options["device_map"] = (
            device_map
        )

    dtype = getattr(
        args,
        "dtype",
        None,
    )

    if dtype is not None:
        options["dtype"] = dtype

    return options


def add_local_model_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """
    Add options used by local Transformers models.
    """

    quantization_group = (
        parser.add_mutually_exclusive_group()
    )

    quantization_group.add_argument(
        "--load-in-4bit",
        action="store_true",
        help=(
            "Load Transformers model "
            "using 4-bit bitsandbytes"
        ),
    )

    quantization_group.add_argument(
        "--load-in-8bit",
        action="store_true",
        help=(
            "Load Transformers model "
            "using 8-bit bitsandbytes"
        ),
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow execution of custom "
            "Hugging Face model code"
        ),
    )

    parser.add_argument(
        "--device-map",
        default=None,
        help=(
            "Transformers device map "
            "(for example: auto, cpu)"
        ),
    )

    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Torch dtype "
            "(auto, float16, bfloat16, float32)"
        ),
    )


def add_benchmark_selection_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """
    Add benchmark task selection and dataset options.
    """

    parser.add_argument(
        "--block",
        default=None,
        help=(
            "Benchmark block to run (for example: "
            "understanding), or 'all' for every block"
        ),
    )

    parser.add_argument(
        "--task",
        action="append",
        default=None,
        dest="tasks",
        help=(
            "Single task id. "
            "May be repeated."
        ),
    )

    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Hugging Face dataset revision "
            "(tag or commit). Pin it for "
            "reproducible results."
        ),
    )

    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token. "
            "Defaults to HF_TOKEN."
        ),
    )


def resolve_token(
    args: argparse.Namespace,
) -> str | None:
    return args.hf_token or os.environ.get("HF_TOKEN")


def select_tasks(args: argparse.Namespace):
    if not args.block and not args.tasks:
        raise ValueError(
            "Select a benchmark block with --block "
            "or a task with --task."
        )

    if args.block and args.tasks:
        raise ValueError(
            "--block and --task cannot be combined."
        )

    return resolve_tasks(
        block=args.block,
        task_ids=args.tasks,
    )


def command_providers(
    args: argparse.Namespace,
) -> None:
    """
    List configured providers.
    """

    providers = load_providers_config(
        config_path=args.config
    )

    print("Available providers:\n")

    for name, config in providers.items():
        backend = config.get(
            "backend",
            "unknown",
        )

        base_url = config.get(
            "base_url"
        )

        print(f"  {name}")
        print(
            f"    backend:  {backend}"
        )

        if base_url:
            print(
                f"    base_url: {base_url}"
            )

        print()


def command_tasks(
    args: argparse.Namespace,
) -> None:
    """
    List registered benchmark tasks.
    """

    for block in list_blocks():
        version = BENCHMARK_VERSIONS.get(block, block)

        tasks = resolve_tasks(block=block)

        total = sum(
            task.expected_examples for task in tasks
        )

        print(f"{block} ({version}) - {total} examples\n")

        for task in tasks:
            print(f"  {task.task_id}")
            print(
                f"    dataset:   "
                f"{task.dataset_repo}:"
                f"{task.dataset_config}"
                f"[{task.dataset_split}]"
            )
            print(
                f"    examples:  "
                f"{task.expected_examples}"
            )
            print(
                f"    answer:    "
                f"{task.answer.mode.value}"
            )
            print(
                f"    evaluator: {task.evaluator} "
                f"(primary: {task.primary_metric})"
            )
            print()


def command_validate_benchmark(
    args: argparse.Namespace,
) -> None:
    """
    Check benchmark data before spending inference budget.
    """

    tasks = select_tasks(args)

    token = resolve_token(args)

    print("Validating benchmark data...\n")

    results = []

    failed = False

    for task in tasks:
        result = validate_task(
            task,
            revision=args.revision,
            token=token,
        )

        results.append((task, result))

        status = "OK" if result.ok else "FAILED"

        print(
            f"  {task.task_id:<22} "
            f"{result.rows}/{result.expected_rows} "
            f"{status}"
        )

        for message in result.errors:
            failed = True
            print(f"      error:   {message}")

        for message in result.warnings:
            print(f"      warning: {message}")

    total_rows = sum(
        result.rows for _, result in results
    )

    total_expected = sum(
        result.expected_rows for _, result in results
    )

    print("  " + "-" * 44)
    print(
        f"  {'TOTAL':<22} "
        f"{total_rows}/{total_expected}"
    )

    if failed:
        print("\nValidation failed.", file=sys.stderr)
        sys.exit(1)

    print("\nPrompts, gold labels and ids validated.")

    if args.write_manifest:
        blocks = {task.block for task, _ in results}

        if len(blocks) != 1:
            print(
                "\nA manifest is one file per block, so it can only "
                "be written for one block at a time. Selected: "
                f"{', '.join(sorted(blocks))}.",
                file=sys.stderr,
            )

            sys.exit(1)

        block = blocks.pop()

        path = write_manifest(
            block=block,
            revision=args.revision,
            tasks=[
                {
                    "task_id": task.task_id,
                    "rows": result.rows,
                    "fingerprint": result.fingerprint,
                    "prompt_version": task.prompt.version,
                    "system_prompt_hash": text_hash(
                        render_system_prompt(task)
                    ),
                }
                for task, result in results
            ],
        )

        print(f"Manifest written to: {path}")


def command_test(
    args: argparse.Namespace,
) -> None:
    """
    Test a single model request.
    """

    try:
        model_options = (
            build_model_options(args)
        )

        model = create_model(
            provider_name=args.provider,
            model=args.model,
            config_path=args.config,
            model_options=model_options,
        )

        request = GenerationRequest(
            messages=[
                Message(
                    role="user",
                    content=args.prompt,
                )
            ],
            max_output_tokens=(
                args.max_tokens
            ),
            temperature=(
                args.temperature
            ),
            top_p=args.top_p,
            thinking=args.thinking,
        )

        print(
            f"Testing {args.model} "
            f"via {args.provider}...\n"
        )

        response = model.generate(
            request
        )

        print(
            f"Response:      "
            f"{response.text}"
        )

        print(
            f"Model:         "
            f"{response.model}"
        )

        print(
            f"Provider:      "
            f"{response.provider}"
        )

        print(
            f"Finish reason: "
            f"{response.finish_reason}"
        )

        if (
            response.usage.input_tokens
            is not None
        ):
            print(
                f"Input tokens:  "
                f"{response.usage.input_tokens}"
            )

        if (
            response.usage.output_tokens
            is not None
        ):
            print(
                f"Output tokens: "
                f"{response.usage.output_tokens}"
            )

        if (
            response.usage.total_tokens
            is not None
        ):
            print(
                f"Total tokens:  "
                f"{response.usage.total_tokens}"
            )

        if response.latency_ms is not None:
            print(
                f"Latency:       "
                f"{response.latency_ms:.0f} ms"
            )

    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def command_run(
    args: argparse.Namespace,
) -> None:
    """
    Run a model on benchmark tasks.
    """

    try:
        tasks = select_tasks(args)

        model_options = (
            build_model_options(args)
        )

        model = create_model(
            provider_name=args.provider,
            model=args.model,
            config_path=args.config,
            model_options=model_options,
        )

        run_benchmark(
            model=model,
            tasks=tasks,
            output_dir=args.output_dir,
            run_id=args.run_id,
            limit=args.limit,
            revision=args.revision,
            token=resolve_token(args),
            resume=args.resume,
            validate=not args.no_validate,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            workers=args.workers,
        )

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Re-run with --resume "
            "and the same --run-id to continue.",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def command_rescore(
    args: argparse.Namespace,
) -> None:
    """
    Recompute the scores of a finished run.
    """

    try:
        rescore_run(
            run_directory=args.run_directory,
            revision=args.revision,
            token=resolve_token(args),
        )

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def command_run_jsonl(
    args: argparse.Namespace,
) -> None:
    """
    Run a model against a plain JSONL file.

    Technical smoke path only; it is not part of the benchmark.
    """

    try:
        model_options = (
            build_model_options(args)
        )

        model = create_model(
            provider_name=args.provider,
            model=args.model,
            config_path=args.config,
            model_options=model_options,
        )

        run_dataset(
            model=model,
            input_path=args.input,
            output_path=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking=args.thinking,
            limit=args.limit,
        )

    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yoxla",
        description=(
            "Azerbaijani LLM evaluation "
            "framework and benchmark"
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"YOXLA {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # ==============================================================
    # yoxla providers
    # ==============================================================

    providers_parser = (
        subparsers.add_parser(
            "providers",
            help=(
                "List configured providers"
            ),
        )
    )

    providers_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional custom providers "
            "YAML configuration"
        ),
    )

    providers_parser.set_defaults(
        func=command_providers
    )

    # ==============================================================
    # yoxla tasks
    # ==============================================================

    tasks_parser = (
        subparsers.add_parser(
            "tasks",
            help=(
                "List benchmark tasks"
            ),
        )
    )

    tasks_parser.set_defaults(
        func=command_tasks
    )

    # ==============================================================
    # yoxla validate-benchmark
    # ==============================================================

    validate_parser = (
        subparsers.add_parser(
            "validate-benchmark",
            help=(
                "Validate benchmark data "
                "before running a model"
            ),
        )
    )

    add_benchmark_selection_arguments(
        validate_parser
    )

    validate_parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Freeze row counts and dataset "
            "fingerprints into the bundled "
            "manifest"
        ),
    )

    validate_parser.set_defaults(
        func=command_validate_benchmark
    )

    # ==============================================================
    # yoxla test
    # ==============================================================

    test_parser = (
        subparsers.add_parser(
            "test",
            help=(
                "Test a model connection"
            ),
        )
    )

    test_parser.add_argument(
        "--provider",
        required=True,
        help=(
            "Provider name "
            "(openrouter, google, openai, "
            "vllm, llamacpp, transformers)"
        ),
    )

    test_parser.add_argument(
        "--model",
        required=True,
        help=(
            "Provider model ID "
            "or local model path"
        ),
    )

    test_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional custom providers "
            "YAML configuration"
        ),
    )

    test_parser.add_argument(
        "--prompt",
        default=(
            "Azərbaycanın paytaxtı hansıdır? "
            "Yalnız şəhərin adını yaz."
        ),
        help="Test prompt",
    )

    test_parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help=(
            "Maximum generated tokens"
        ),
    )

    test_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )

    test_parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value",
    )

    test_parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Enable reasoning/thinking mode "
            "when supported"
        ),
    )

    add_local_model_arguments(
        test_parser
    )

    test_parser.set_defaults(
        func=command_test
    )

    # ==============================================================
    # yoxla run
    # ==============================================================

    run_parser = (
        subparsers.add_parser(
            "run",
            help=(
                "Run a model on the benchmark"
            ),
        )
    )

    run_parser.add_argument(
        "--provider",
        required=True,
        help="Provider name",
    )

    run_parser.add_argument(
        "--model",
        required=True,
        help=(
            "Provider model ID "
            "or local model path"
        ),
    )

    run_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional custom providers "
            "YAML configuration"
        ),
    )

    add_benchmark_selection_arguments(
        run_parser
    )

    run_parser.add_argument(
        "--output-dir",
        default="runs",
        help=(
            "Directory that receives "
            "run subdirectories"
        ),
    )

    run_parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run identifier. "
            "Required together with --resume."
        ),
    )

    run_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse predictions already stored "
            "in the run directory"
        ),
    )

    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N examples "
            "of every task"
        ),
    )

    run_parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Skip benchmark data validation"
        ),
    )

    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help=(
            "Retries for transport failures "
            "(never for bad model output)"
        ),
    )

    run_parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help=(
            "Base seconds for exponential "
            "retry backoff"
        ),
    )

    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Requests in flight at once. Defaults to 1. "
            "Raising it shortens a run but is part of its "
            "conditions: under load a provider answers 429 more "
            "often, and an example whose retries run out counts "
            "as a generation error. Check that the summary "
            "reports none before comparing scores."
        ),
    )

    add_local_model_arguments(
        run_parser
    )

    run_parser.set_defaults(
        func=command_run
    )

    # ==============================================================
    # yoxla rescore
    # ==============================================================


    rescore_parser = (
        subparsers.add_parser(
            "rescore",
            help=(
                "Recompute scores of a finished "
                "run from its stored answers"
            ),
        )
    )

    rescore_parser.add_argument(
        "run_directory",
        help=(
            "Run directory, for example "
            "runs/openrouter_model_20260818T135510Z"
        ),
    )

    rescore_parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Dataset revision. Defaults to the "
            "revision recorded in run.json."
        ),
    )

    rescore_parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token. "
            "Defaults to HF_TOKEN."
        ),
    )

    rescore_parser.set_defaults(
        func=command_rescore
    )

    # ==============================================================
    # yoxla run-jsonl
    # ==============================================================

    jsonl_parser = (
        subparsers.add_parser(
            "run-jsonl",
            help=(
                "Run a model on a plain JSONL "
                "file (smoke path)"
            ),
        )
    )

    jsonl_parser.add_argument(
        "--provider",
        required=True,
        help="Provider name",
    )

    jsonl_parser.add_argument(
        "--model",
        required=True,
        help=(
            "Provider model ID "
            "or local model path"
        ),
    )

    jsonl_parser.add_argument(
        "--input",
        default="data/smoke.jsonl",
        help=(
            "Input JSONL dataset"
        ),
    )

    jsonl_parser.add_argument(
        "--output",
        default="runs/results.jsonl",
        help=(
            "Output JSONL results file"
        ),
    )

    jsonl_parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional custom providers "
            "YAML configuration"
        ),
    )

    jsonl_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N examples"
        ),
    )

    jsonl_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help=(
            "Maximum generated tokens "
            "per example"
        ),
    )

    jsonl_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )

    jsonl_parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value",
    )

    jsonl_parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Enable reasoning/thinking mode "
            "when supported"
        ),
    )

    add_local_model_arguments(
        jsonl_parser
    )

    jsonl_parser.set_defaults(
        func=command_run_jsonl
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    if not hasattr(
        args,
        "func",
    ):
        parser.print_help()
        return

    try:
        args.func(args)

    except ValueError as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
