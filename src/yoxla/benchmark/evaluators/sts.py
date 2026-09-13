"""
STS evaluator.

Invalid outputs are not dropped. A model that produces unusable
answers for half of the examples must not be scored on the other half
as if nothing happened, so invalid predictions are imputed with the
midpoint of the scale: a constant carries no covariance and therefore
lowers the correlation in proportion to how often it occurs.

The valid-only correlations are reported alongside as diagnostics.
"""

import math
from typing import Any

from yoxla.benchmark.evaluators.base import Evaluator
from yoxla.benchmark.evaluators.correlation import pearson, spearman
from yoxla.benchmark.parsing import ParsedPrediction
from yoxla.benchmark.schema import BenchmarkExample, BenchmarkTask


class STSEvaluator(Evaluator):
    name = "sts_regression"

    def __init__(self, task: BenchmarkTask):
        super().__init__(task)

        minimum = task.answer.minimum or 0.0
        maximum = (
            task.answer.maximum
            if task.answer.maximum is not None
            else 5.0
        )

        self.imputed_value = (minimum + maximum) / 2

        self.golds: list[float] = []
        self.predictions: list[float] = []

        self.valid_golds: list[float] = []
        self.valid_predictions: list[float] = []

    def _add(
        self,
        example: BenchmarkExample,
        prediction: ParsedPrediction,
    ) -> dict[str, Any]:
        gold = float(example.gold)

        if prediction.valid:
            value = float(prediction.value)

            self.valid_golds.append(gold)
            self.valid_predictions.append(value)

        else:
            value = self.imputed_value

        self.golds.append(gold)
        self.predictions.append(value)

        return {
            "gold_score": gold,
            "scored_prediction": value,
            "absolute_error": abs(value - gold),
        }

    def _compute(self) -> dict[str, Any]:
        errors = [
            predicted - gold
            for gold, predicted in zip(
                self.golds, self.predictions, strict=True
            )
        ]

        count = len(errors) or 1

        return {
            "spearman": _or_zero(
                spearman(self.golds, self.predictions)
            ),
            "pearson": _or_zero(
                pearson(self.golds, self.predictions)
            ),
            "spearman_valid_only": _or_zero(
                spearman(
                    self.valid_golds,
                    self.valid_predictions,
                )
            ),
            "pearson_valid_only": _or_zero(
                pearson(
                    self.valid_golds,
                    self.valid_predictions,
                )
            ),
            "mae": sum(abs(error) for error in errors)
            / count,
            "rmse": math.sqrt(
                sum(error**2 for error in errors) / count
            ),
        }


def _or_zero(value: float | None) -> float:
    """
    An undefined correlation - too few points, or a model that
    answered with a single constant - scores zero rather than null.
    """

    return 0.0 if value is None else value
