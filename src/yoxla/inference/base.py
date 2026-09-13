from abc import ABC, abstractmethod
from dataclasses import dataclass

from yoxla.inference.schemas import (
    GenerationRequest,
    GenerationResponse,
)


@dataclass
class ModelCapabilities:
    system_prompt: bool = True

    temperature: bool = True
    top_p: bool = True
    seed: bool = False
    stop: bool = True

    structured_output: bool = False

    max_context_tokens: int | None = None


class ModelAdapter(ABC):
    """
    Base interface for all models used by YOXLA.

    Implementations may call:
    - OpenAI
    - OpenRouter
    - vLLM
    - llama.cpp
    - Hugging Face Transformers
    - Unsloth
    - other providers
    """

    def __init__(
        self,
        model: str,
        provider: str,
        capabilities: ModelCapabilities | None = None,
    ):
        self.model = model
        self.provider = provider

        self.capabilities = capabilities or ModelCapabilities()

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """
        Generate a response for a single request.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"provider={self.provider!r})"
        )