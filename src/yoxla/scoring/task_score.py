"""
Task scoring.

Raw metrics are not comparable across tasks: accuracy, exact match
and a rank correlation live on different scales. Every task is
therefore mapped onto a normalized ``[0, 100]`` score before any
aggregation happens.

    classification   accuracy            * 100
    extractive QA    normalized_em       * 100
    QA abstention    overall_accuracy    * 100
    STS              max(0, spearman)    * 100

Negative correlation means the model ranked similarity backwards; it
is clamped to zero rather than allowed to subtract from other tasks.
"""

from dataclasses import dataclass, field
from typing import Any

from yoxla.benchmark.schema import BenchmarkTask


@dataclass
class TaskScore:
    task_id: str
    block: str

    score: float

    primary_metric: str
    primary_value: float | None

    count: int

    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "block": self.block,
            "score": self.score,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "count": self.count,
            "metrics": self.metrics,
        }


def normalize_metric(value: float | None) -> float:
    """
    Map a metric onto [0, 100].
    """

    if value is None:
        return 0.0

    clamped = max(0.0, min(1.0, float(value)))

    return round(clamped * 100, 4)


def compute_task_score(
    task: BenchmarkTask,
    metrics: dict[str, Any],
) -> TaskScore:
    primary_value = metrics.get(task.primary_metric)

    if primary_value is not None:
        primary_value = float(primary_value)

    return TaskScore(
        task_id=task.task_id,
        block=task.block,
        score=normalize_metric(primary_value),
        primary_metric=task.primary_metric,
        primary_value=primary_value,
        count=int(metrics.get("count", 0)),
        metrics=metrics,
    )
