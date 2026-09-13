"""
Extractive QA and abstention, scored on character offsets.

Both tasks put the answer inside a passage the model is given, so both
are scored the same way: did the quote cover the gold span, and how
much else did it drag in. See ``span.py`` for why the two are reported
apart instead of folded into one number.

The older string metrics stay as diagnostics. They are what previous
runs were scored on, and keeping them visible is what makes the change
in the headline number explainable rather than mysterious.
"""

from typing import Any

from yoxla.benchmark.evaluators.base import Evaluator, mean
from yoxla.benchmark.evaluators.extractive_qa import score_answer
from yoxla.benchmark.evaluators.span import (
    check_gold_is_locatable,
    score_span,
)
from yoxla.benchmark.parsing import ParsedPrediction
from yoxla.benchmark.schema import BenchmarkExample, BenchmarkTask


class SpanScores:
    """
    The running totals both evaluators keep.
    """

    def __init__(self) -> None:
        self.recall: list[float] = []
        self.precision: list[float] = []

        self.exact: list[float] = []
        self.unsupported: list[float] = []
        self.outside: list[float] = []

        self.token_f1: list[float] = []
        self.normalized_em: list[float] = []
        self.strict_em: list[float] = []

    def add(
        self,
        example: BenchmarkExample,
        prediction: ParsedPrediction,
        gold: str,
    ) -> dict[str, Any]:
        if not prediction.valid:
            match = None
            legacy = {
                "normalized_em": 0.0,
                "token_f1": 0.0,
            }

        else:
            match = score_span(
                example.inputs["context"],
                gold,
                str(prediction.value),
            )

            legacy = score_answer(str(prediction.value), gold)

        recall = match.recall if match else 0.0
        precision = match.precision if match else 0.0

        # An answer that cannot be found in the passage was not
        # copied from it. That is a different failure from quoting
        # the wrong part, and it is the one worth counting separately.
        unsupported = float(match is None or not match.found)

        self.recall.append(recall)
        self.precision.append(precision)
        self.exact.append(float(bool(match and match.exact)))
        self.unsupported.append(unsupported)
        self.outside.append(
            float(bool(match and match.outside_sentence))
        )

        self.token_f1.append(legacy["token_f1"])
        self.normalized_em.append(legacy["normalized_em"])
        self.strict_em.append(
            float(prediction.raw.strip() == gold.strip())
        )

        return {
            "span_recall": recall,
            "span_precision": precision,
            "exact_span": float(bool(match and match.exact)),
            "unsupported": unsupported,
            "outside_sentence": float(
                bool(match and match.outside_sentence)
            ),
            "token_f1": legacy["token_f1"],
            "normalized_em": legacy["normalized_em"],
        }

    def compute(self) -> dict[str, Any]:
        return {
            "span_recall": mean(self.recall),
            "span_precision": mean(self.precision),
            "exact_span_rate": mean(self.exact),
            "unsupported_rate": mean(self.unsupported),
            "over_sentence_rate": mean(self.outside),
            "token_f1": mean(self.token_f1),
            "normalized_em": mean(self.normalized_em),
            "strict_em": mean(self.strict_em),
        }


class SpanQAEvaluator(Evaluator):
    name = "span_qa"

    def __init__(self, task: BenchmarkTask):
        super().__init__(task)

        self.spans = SpanScores()

    def validate_gold(
        self,
        example: BenchmarkExample,
    ) -> list[str]:
        return check_gold_is_locatable(example, str(example.gold))

    def _add(
        self,
        example: BenchmarkExample,
        prediction: ParsedPrediction,
    ) -> dict[str, Any]:
        return self.spans.add(
            example, prediction, str(example.gold)
        )

    def _compute(self) -> dict[str, Any]:
        return self.spans.compute()


class SpanAbstentionEvaluator(Evaluator):
    name = "span_abstention"

    def __init__(self, task: BenchmarkTask):
        super().__init__(task)

        self.spans = SpanScores()

        self.answerability: list[float] = []
        self.overall: list[float] = []

        self.unanswerable: list[float] = []

    def validate_gold(
        self,
        example: BenchmarkExample,
    ) -> list[str]:
        if example.gold is None:
            return []

        return check_gold_is_locatable(example, str(example.gold))

    def _add(
        self,
        example: BenchmarkExample,
        prediction: ParsedPrediction,
    ) -> dict[str, Any]:
        gold_is_unanswerable = example.gold is None

        # An invalid output is not an abstention: the model simply
        # failed to answer in the required format.
        predicted_abstain = (
            prediction.valid and prediction.abstained
        )

        answerability_correct = float(
            predicted_abstain == gold_is_unanswerable
        )

        metrics: dict[str, Any] = {
            "answerability_correct": answerability_correct,
            "predicted_abstain": predicted_abstain,
        }

        if gold_is_unanswerable:
            score = answerability_correct

            self.unanswerable.append(answerability_correct)

        elif predicted_abstain:
            # Refusing an answerable question costs the whole example;
            # there is no span to score.
            score = 0.0

            metrics.update(
                {
                    "span_recall": 0.0,
                    "span_precision": 0.0,
                    "unsupported": 0.0,
                }
            )

        else:
            span = self.spans.add(
                example, prediction, str(example.gold)
            )

            metrics.update(span)

            score = span["span_recall"]

        self.answerability.append(answerability_correct)
        self.overall.append(score)

        metrics["score"] = score

        return metrics

    def _compute(self) -> dict[str, Any]:
        metrics = {
            "overall_score": mean(self.overall),
            "answerability_accuracy": mean(self.answerability),
            "unanswerable_accuracy": mean(self.unanswerable),
        }

        # Span metrics describe the answerable half only, so they are
        # averaged over the examples that had a span to quote.
        metrics.update(
            {
                f"answerable_{key}": value
                for key, value in self.spans.compute().items()
            }
        )

        return metrics
