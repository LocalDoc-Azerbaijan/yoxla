"""
Result schema.

A run must be reproducible from what it writes: YOXLA version,
benchmark version, dataset revision and fingerprint, prompt version
and prompt hashes, model, provider and generation settings.

Raw model output is never discarded, so metrics can be recomputed
later without paying for inference again.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

RESULT_SCHEMA_VERSION = 1


@dataclass
class TaskRunInfo:
    task_id: str
    block: str

    dataset_repo: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str | None
    dataset_fingerprint: str

    expected_examples: int
    loaded_examples: int

    prompt_version: int
    system_prompt_hash: str

    generation: dict[str, Any]

    warnings: list[str] = field(default_factory=list)


@dataclass
class RunInfo:
    run_id: str

    benchmark: str
    benchmark_versions: dict[str, str]

    yoxla_version: str

    model: str
    provider: str

    started_at: str
    finished_at: str | None = None

    result_schema_version: int = RESULT_SCHEMA_VERSION

    tasks: list[TaskRunInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRecord:
    """
    One benchmark example, one model attempt.
    """

    run_id: str
    task_id: str
    example_id: str

    gold: Any

    raw_prediction: str | None
    parsed_prediction: Any
    prediction_valid: bool
    prediction_abstained: bool
    parse_error: str | None

    metrics: dict[str, Any]

    model: str
    provider: str

    finish_reason: str | None

    input_tokens: int | None
    output_tokens: int | None

    latency_ms: float | None

    user_prompt_hash: str

    metadata: dict[str, Any] = field(default_factory=dict)

    # Filled only when the answer could not be parsed: an empty
    # completion tells you nothing on its own, while the provider
    # payload shows whether the tokens went into a reasoning channel,
    # a refusal, or a field this adapter does not read yet.
    provider_response: Any = None

    error: str | None = None

    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
