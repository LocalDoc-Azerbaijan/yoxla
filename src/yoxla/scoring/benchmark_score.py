"""
Benchmark scoring.

The aggregate over blocks is a macro average as well. Only the
Understanding block exists today, so the aggregate currently equals
the Understanding score; block weights stay unfrozen until the
remaining blocks are built.
"""

from dataclasses import dataclass, field
from typing import Any

from yoxla.scoring.block_score import BlockScore
from yoxla.scoring.task_score import TaskScore


@dataclass
class BenchmarkScore:
    score: float

    blocks: list[BlockScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "blocks": [
                block.to_dict() for block in self.blocks
            ],
        }


def compute_benchmark_score(
    block_scores: list[BlockScore],
) -> BenchmarkScore:
    scores = [block.score for block in block_scores]

    average = (
        round(sum(scores) / len(scores), 4)
        if scores
        else 0.0
    )

    return BenchmarkScore(
        score=average,
        blocks=list(block_scores),
    )


def group_task_scores(
    task_scores: list[TaskScore],
) -> dict[str, list[TaskScore]]:
    grouped: dict[str, list[TaskScore]] = {}

    for task_score in task_scores:
        grouped.setdefault(
            task_score.block, []
        ).append(task_score)

    return grouped
