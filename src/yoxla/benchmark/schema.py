"""
YOXLA Task Contract v1.

A benchmark task is fully described by this schema: dataset location,
field mapping, prompt, answer space, generation settings, parser mode
and evaluator. A task that defines all of it is automatically
compatible with the runner - no runner or adapter changes required.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1


class AnswerMode(str, Enum):
    """
    How a model answer must be interpreted.
    """

    TEXT = "text"
    TEXT_OR_ABSTAIN = "text_or_abstain"
    CHOICE = "choice"
    NUMERIC = "numeric"


@dataclass(frozen=True)
class ChoiceSpec:
    """
    One allowed label of a CHOICE task.

    ``description`` is rendered into the prompt so that the benchmark
    measures understanding of the task rather than memorization of an
    external label taxonomy.

    ``group`` is an optional coarse category used for diagnostics.
    """

    value: str
    description: str
    group: str | None = None


@dataclass(frozen=True)
class RubricLevel:
    """
    One level of a NUMERIC rating scale.
    """

    value: str
    description: str


@dataclass(frozen=True)
class AnswerSpec:
    """
    The answer space of a task.
    """

    mode: AnswerMode

    choices: tuple[ChoiceSpec, ...] = ()

    abstain_value: str | None = None

    rubric: tuple[RubricLevel, ...] = ()

    minimum: float | None = None
    maximum: float | None = None

    def choice_values(self) -> tuple[str, ...]:
        return tuple(
            choice.value for choice in self.choices
        )

    def choice_groups(self) -> dict[str, str]:
        return {
            choice.value: choice.group
            for choice in self.choices
            if choice.group is not None
        }


@dataclass(frozen=True)
class PromptSpec:
    """
    A frozen, deterministic prompt.

    ``version`` is bumped only together with the task id, because a
    prompt change makes results incomparable.
    """

    version: int
    system: str
    user: str


@dataclass(frozen=True)
class GenerationSpec:
    """
    Task-level generation settings.

    Benchmark v1 is deterministic: temperature 0, no reasoning.
    """

    max_output_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    thinking: bool = False


@dataclass(frozen=True)
class BenchmarkTask:
    """
    A frozen benchmark task definition.
    """

    task_id: str
    block: str
    language: str

    dataset_repo: str
    dataset_config: str
    dataset_split: str

    expected_examples: int

    input_fields: tuple[str, ...]
    gold_field: str
    metadata_fields: tuple[str, ...]

    prompt: PromptSpec
    answer: AnswerSpec
    generation: GenerationSpec

    evaluator: str
    primary_metric: str
    diagnostic_metrics: tuple[str, ...] = ()

    gold_nullable: bool = False

    # Named normalizer applied to both sides of an answer comparison.
    # It carries what is a fact about the language - "Quba rayonu" and
    # "Quba" are one district - while accepted_forms carries what is a
    # fact about the world. Frozen with the task id like a prompt.

    # A metadata column to break the score down by. minimal_pairs
    # carries eight grammatical phenomena in one task, and one number
    # over all of them says far less than eight numbers do.
    breakdown_field: str | None = None

    # A retired task stays registered so that older runs remain
    # reproducible, but it is no longer part of its block.
    retired: bool = False

    schema_version: int = SCHEMA_VERSION

    def required_columns(self) -> tuple[str, ...]:
        columns = ["id", *self.input_fields, self.gold_field]

        columns.extend(self.metadata_fields)

        if self.breakdown_field:
            columns.append(self.breakdown_field)

        seen: list[str] = []

        for column in columns:
            if column not in seen:
                seen.append(column)

        return tuple(seen)


@dataclass(frozen=True)
class BenchmarkExample:
    """
    One runtime example.

    Produced by the loader from a raw dataset row. Everything
    downstream of the loader works only with this structure, so the
    runner never sees provider- or dataset-specific shapes.
    """

    id: str
    task_id: str
    inputs: dict[str, Any]
    gold: Any
    metadata: dict[str, Any]

    schema_version: int = SCHEMA_VERSION
