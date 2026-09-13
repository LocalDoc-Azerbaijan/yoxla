"""
Block scoring.

The block score is a macro average over task scores, never an average
over examples. Understanding v1 contains 200 QA examples but only 50
intent examples, so example-level averaging would give QA four times
the weight of intent by accident.

Explicit task weights may be introduced later, once enough models
have been measured to see which tasks actually discriminate.
"""

from dataclasses import dataclass, field
from typing import Any

from yoxla.scoring.task_score import TaskScore


@dataclass
class BlockScore:
    block: str
    score: float

    task_scores: list[TaskScore] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "score": self.score,
            "tasks": {
                task_score.task_id: task_score.score
                for task_score in self.task_scores
            },
        }


def compute_block_score(
    block: str,
    task_scores: list[TaskScore],
) -> BlockScore:
    scores = [
        task_score.score for task_score in task_scores
    ]

    average = (
        round(sum(scores) / len(scores), 4)
        if scores
        else 0.0
    )

    return BlockScore(
        block=block,
        score=average,
        task_scores=list(task_scores),
    )
