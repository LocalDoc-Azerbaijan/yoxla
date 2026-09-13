"""
Evaluator interface.

An evaluator compares parsed predictions with gold values. It never
performs inference, never re-prompts a model and never sees raw
provider responses.

``add()`` returns the per-example metrics that are written into
``predictions.jsonl``; ``compute()`` returns the aggregated task
metrics.
"""

from abc import ABC, abstractmethod
from typing import Any

from yoxla.benchmark.parsing import ParsedPrediction
from yoxla.benchmark.schema import BenchmarkExample, BenchmarkTask


class Evaluator(ABC):
    name: str = ""

    def __init__(self, task: BenchmarkTask):
        self.task = task

        self.total = 0
        self.invalid = 0

    def add(
        self,
        example: BenchmarkExample,
        prediction: ParsedPrediction,
    ) -> dict[str, Any]:
        self.total += 1

        if not prediction.valid:
            self.invalid += 1

        return self._add(example, prediction)

    def compute(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "count": self.total,
            "invalid_output_rate": (
                self.invalid / self.total
                if self.total
                else 0.0
            ),
        }

        metrics.update(self._compute())

        return metrics

    def validate_gold(
        self,
        example: BenchmarkExample,
    ) -> list[str]:
        """
        Problems with an example's gold value, before any scoring.

        The generic checks in ``validation`` know that a gold must not
        be empty and must sit inside the declared answer space. What a
        well-formed gold looks like beyond that is the evaluator's
        business, so an evaluator that needs more says so here.
        """

        return []

    @abstractmethod
    def _add(
        self,
        example: BenchmarkExample,
        prediction: ParsedPrediction,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _compute(self) -> dict[str, Any]:
        raise NotImplementedError


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)
