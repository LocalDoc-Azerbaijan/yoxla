"""
Task registry.

The runner resolves tasks and evaluators only through this registry,
so adding a benchmark task never requires touching the runner or the
inference adapters.
"""

from yoxla.benchmark.schema import BenchmarkTask
from yoxla.benchmark.tasks.knowledge import (
    KNOWLEDGE_TASKS,
    RETIRED_KNOWLEDGE_TASKS,
)
from yoxla.benchmark.tasks.language import (
    LANGUAGE_TASKS,
    RETIRED_LANGUAGE_TASKS,
)
from yoxla.benchmark.tasks.rag import (
    RAG_TASKS,
    RETIRED_RAG_TASKS,
)
from yoxla.benchmark.tasks.understanding import (
    RETIRED_UNDERSTANDING_TASKS,
    UNDERSTANDING_TASKS,
)

# Order matters: it is the order blocks and tasks appear in a report,
# and it is the order a multi-block run executes in.
_ALL_TASKS: tuple[BenchmarkTask, ...] = (
    *UNDERSTANDING_TASKS,
    *LANGUAGE_TASKS,
    *KNOWLEDGE_TASKS,
    *RAG_TASKS,
    *RETIRED_UNDERSTANDING_TASKS,
    *RETIRED_LANGUAGE_TASKS,
    *RETIRED_KNOWLEDGE_TASKS,
    *RETIRED_RAG_TASKS,
)

TASKS: dict[str, BenchmarkTask] = {
    task.task_id: task for task in _ALL_TASKS
}

BENCHMARK_VERSIONS: dict[str, str] = {
    "understanding": "understanding_v1",
    "language": "language_v1",
    "knowledge": "knowledge_v1",
    "rag": "rag_v1",
}


def list_blocks() -> list[str]:
    """
    Blocks that currently have at least one active task.
    """

    blocks: list[str] = []

    for task in _ALL_TASKS:
        if task.retired:
            continue

        if task.block not in blocks:
            blocks.append(task.block)

    return blocks


def get_task(task_id: str) -> BenchmarkTask:
    if task_id not in TASKS:
        available = ", ".join(sorted(TASKS))

        raise ValueError(
            f"Unknown task '{task_id}'. "
            f"Available tasks: {available}"
        )

    return TASKS[task_id]


# Selects every active task rather than one block. The whole
# benchmark is the most common thing to run and it had no shorter
# spelling than ten repeated --task flags.
ALL_BLOCKS = "all"


def get_block_tasks(block: str) -> list[BenchmarkTask]:
    """
    Active tasks of a block, or of the whole benchmark for ``all``.

    Retired tasks stay in the registry - an old run can be reproduced
    with ``--task extractive_qa_v1`` - but they are not part of the
    block any more.
    """

    tasks = [
        task
        for task in _ALL_TASKS
        if not task.retired
        and (block == ALL_BLOCKS or task.block == block)
    ]

    if not tasks:
        available = ", ".join([*list_blocks(), ALL_BLOCKS])

        raise ValueError(
            f"Unknown block '{block}'. "
            f"Available blocks: {available}"
        )

    return tasks


def resolve_tasks(
    block: str | None = None,
    task_ids: list[str] | None = None,
) -> list[BenchmarkTask]:
    """
    Resolve a task selection.

    Exactly one of ``block`` or ``task_ids`` is expected. Tasks keep
    their registry order so that runs stay comparable.
    """

    if task_ids:
        selected = [
            get_task(task_id) for task_id in task_ids
        ]

        return selected

    if block:
        return get_block_tasks(block)

    raise ValueError(
        "Either a block or at least one task must be selected."
    )
