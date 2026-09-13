"""
Exact-match classification evaluator.

Used by NLI, sentiment and intent. An invalid output counts as a
wrong answer - it is never silently dropped and never repaired.
"""

from collections import Counter
from typing import Any

from yoxla.benchmark.evaluators.base import Evaluator
from yoxla.benchmark.parsing import ParsedPrediction
from yoxla.benchmark.schema import BenchmarkExample, BenchmarkTask


class ClassificationEvaluator(Evaluator):
    name = "classification_exact"

    def __init__(self, task: BenchmarkTask):
        super().__init__(task)

        self.golds: list[str] = []
        self.predictions: list[str | None] = []

        self.breakdown_field = task.breakdown_field

        self.by_group: dict[str, list[float]] = {}

        self.groups = task.answer.choice_groups()

        self.use_groups = len(self.groups) == len(
            task.answer.choices
        )

    def _add(
        self,
        example: BenchmarkExample,
        prediction: ParsedPrediction,
    ) -> dict[str, Any]:
        gold = str(example.gold)

        predicted = (
            prediction.value if prediction.valid else None
        )

        self.golds.append(gold)
        self.predictions.append(predicted)

        correct = int(predicted == gold)

        metrics: dict[str, Any] = {"correct": correct}

        if self.breakdown_field:
            key = str(
                example.metadata.get(self.breakdown_field, "")
            )

            self.by_group.setdefault(key, []).append(
                float(correct)
            )

        if self.use_groups:
            metrics["group_correct"] = int(
                predicted is not None
                and self.groups.get(predicted)
                == self.groups.get(gold)
            )

        return metrics

    def _compute(self) -> dict[str, Any]:
        if not self.golds:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "per_class_accuracy": {},
            }

        correct = sum(
            gold == predicted
            for gold, predicted in zip(
                self.golds, self.predictions, strict=True
            )
        )

        # Macro F1 is averaged over the classes present in the gold
        # labels; classes that never occur would otherwise contribute
        # an undefined score.
        labels = sorted(set(self.golds))

        f1_scores: list[float] = []
        per_class: dict[str, float] = {}

        for label in labels:
            true_positive = sum(
                gold == label and predicted == label
                for gold, predicted in zip(
                    self.golds, self.predictions, strict=True
                )
            )

            false_positive = sum(
                gold != label and predicted == label
                for gold, predicted in zip(
                    self.golds, self.predictions, strict=True
                )
            )

            gold_count = sum(
                gold == label for gold in self.golds
            )

            precision = (
                true_positive
                / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )

            recall = (
                true_positive / gold_count
                if gold_count
                else 0.0
            )

            f1_scores.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )

            per_class[label] = recall

        # A model with no opinion still has to answer, and it tends to
        # answer the same way every time - the first option, usually.
        # Accuracy alone cannot tell that apart from knowing: a model
        # that picks option 1 everywhere scores at chance and looks
        # merely weak. This says how much of the run was one repeated
        # answer, so guessing is visible rather than averaged in.
        answered = [p for p in self.predictions if p is not None]

        modal_share = (
            max(Counter(answered).values()) / len(self.predictions)
            if answered
            else 0.0
        )

        metrics: dict[str, Any] = {
            "accuracy": correct / len(self.golds),
            "macro_f1": sum(f1_scores) / len(f1_scores),
            "per_class_accuracy": per_class,
            "modal_answer_share": modal_share,
        }

        if self.breakdown_field:
            metrics[
                f"per_{self.breakdown_field}_accuracy"
            ] = {
                key: sum(values) / len(values)
                for key, values in sorted(self.by_group.items())
            }

        if self.use_groups:
            group_correct = sum(
                predicted is not None
                and self.groups.get(predicted)
                == self.groups.get(gold)
                for gold, predicted in zip(
                    self.golds, self.predictions, strict=True
                )
            )

            metrics["group_accuracy"] = (
                group_correct / len(self.golds)
            )

        return metrics
