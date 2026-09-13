"""
Evaluator registry.

A task names its evaluator as a string, so the runner never needs
task-specific branches.
"""

from yoxla.benchmark.evaluators.base import Evaluator
from yoxla.benchmark.evaluators.classification import (
    ClassificationEvaluator,
)
from yoxla.benchmark.evaluators.span_qa import (
    SpanAbstentionEvaluator,
    SpanQAEvaluator,
)
from yoxla.benchmark.evaluators.sts import STSEvaluator
from yoxla.benchmark.schema import BenchmarkTask

EVALUATORS: dict[str, type[Evaluator]] = {
    ClassificationEvaluator.name: ClassificationEvaluator,
    SpanAbstentionEvaluator.name: SpanAbstentionEvaluator,
    SpanQAEvaluator.name: SpanQAEvaluator,
    STSEvaluator.name: STSEvaluator,
}


def create_evaluator(task: BenchmarkTask) -> Evaluator:
    if task.evaluator not in EVALUATORS:
        available = ", ".join(sorted(EVALUATORS))

        raise ValueError(
            f"Task '{task.task_id}' requests unknown evaluator "
            f"'{task.evaluator}'. Available: {available}"
        )

    return EVALUATORS[task.evaluator](task)


__all__ = [
    "EVALUATORS",
    "ClassificationEvaluator",
    "Evaluator",
    "SpanAbstentionEvaluator",
    "SpanQAEvaluator",
    "STSEvaluator",
    "create_evaluator",
]
