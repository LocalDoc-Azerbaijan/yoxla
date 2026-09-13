from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class GenerationRequest:
    messages: list[Message]

    max_output_tokens: int = 512
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    seed: int | None = None

    thinking: bool = False

    stop: list[str] | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None

        return self.input_tokens + self.output_tokens


@dataclass
class GenerationResponse:
    text: str

    raw_text: str | None = None

    model: str = ""
    provider: str = ""

    finish_reason: str | None = None

    usage: TokenUsage = field(default_factory=TokenUsage)

    latency_ms: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    raw_response: Any = None