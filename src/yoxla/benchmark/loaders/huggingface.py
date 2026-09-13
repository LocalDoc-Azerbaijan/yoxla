"""
Hugging Face dataset loader.

The loader contains no task-specific logic: it maps raw dataset rows
onto ``BenchmarkExample`` using the field mapping declared by the
task. Benchmark data therefore stays on the Hub and is versioned
independently from the ``yoxla`` package.
"""

import hashlib
import json
from typing import Any

from yoxla.benchmark.schema import BenchmarkExample, BenchmarkTask


def load_rows(
    task: BenchmarkTask,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Download the raw rows of a task.

    ``revision`` pins a dataset tag or commit. Reproducible runs
    should always pin it.
    """

    try:
        from datasets import load_dataset

    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load benchmark "
            "data. Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(
        task.dataset_repo,
        task.dataset_config,
        split=task.dataset_split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )

    return [dict(row) for row in dataset]


def row_to_example(
    task: BenchmarkTask,
    row: dict[str, Any],
) -> BenchmarkExample:
    missing = [
        column
        for column in task.required_columns()
        if column not in row
    ]

    if missing:
        raise ValueError(
            f"Dataset '{task.dataset_repo}/{task.dataset_config}' "
            f"is missing columns required by task "
            f"'{task.task_id}': {', '.join(missing)}"
        )

    return BenchmarkExample(
        id=str(row["id"]),
        task_id=task.task_id,
        inputs={
            field: row[field]
            for field in task.input_fields
        },
        gold=row[task.gold_field],
        metadata={
            field: row[field]
            for field in task.metadata_fields
        },
    )


def load_examples(
    task: BenchmarkTask,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
) -> list[BenchmarkExample]:
    rows = load_rows(
        task,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )

    return [
        row_to_example(task, row) for row in rows
    ]


def rows_fingerprint(
    rows: list[dict[str, Any]],
) -> str:
    """
    Content hash of a dataset config.

    Used to detect that a frozen benchmark config changed on the Hub
    after results were published.
    """

    digest = hashlib.sha256()

    for row in rows:
        canonical = json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")

    return f"sha256:{digest.hexdigest()}"
