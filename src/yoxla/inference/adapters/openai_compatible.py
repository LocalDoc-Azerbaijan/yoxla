import time
from typing import Any

from yoxla.inference.base import ModelAdapter, ModelCapabilities
from yoxla.inference.errors import ProviderError
from yoxla.inference.schemas import (
    GenerationRequest,
    GenerationResponse,
    TokenUsage,
)


def _openai_client(**kwargs: Any):
    """
    Build the client, importing the SDK only when one is asked for.

    `openai` ships under the `api` extra, and importing it at module
    level made `pip install yoxla` unimportable: the adapters package
    is loaded eagerly by the factory, so a plain install failed on
    `import yoxla` before it could run anything.
    """

    try:
        from openai import OpenAI

    except ImportError as error:  # pragma: no cover
        raise ProviderError(
            "This provider needs the OpenAI SDK. Install it with:\n"
            '    pip install "yoxla[api]"'
        ) from error

    return OpenAI(**kwargs)


class OpenAICompatibleAdapter(ModelAdapter):
    """
    Adapter for OpenAI-compatible APIs.

    Examples:
    - OpenAI
    - OpenRouter
    - vLLM
    - llama.cpp server
    - other compatible endpoints
    """

    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        capabilities: ModelCapabilities | None = None,
        default_headers: dict[str, str] | None = None,
        max_tokens_param: str = "max_tokens",
        thinking_control: str | None = None,
    ):
        super().__init__(
            model=model,
            provider=provider,
            capabilities=capabilities,
        )

        self.max_tokens_param = max_tokens_param
        self.thinking_control = thinking_control

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
        }

        if base_url is not None:
            client_kwargs["base_url"] = base_url

        if default_headers is not None:
            client_kwargs["default_headers"] = default_headers

        self.client = _openai_client(**client_kwargs)

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.max_tokens_param: request.max_output_tokens,
        }

        if (
            request.temperature is not None
            and self.capabilities.temperature
        ):
            params["temperature"] = request.temperature

        if (
            request.top_p is not None
            and self.capabilities.top_p
        ):
            params["top_p"] = request.top_p

        if (
            request.seed is not None
            and self.capabilities.seed
        ):
            params["seed"] = request.seed

        if (
            request.stop is not None
            and self.capabilities.stop
        ):
            params["stop"] = request.stop

        extra_body = dict(request.extra)

        if self.thinking_control == "chat_template_kwargs":
            chat_template_kwargs = dict(
                extra_body.get("chat_template_kwargs", {})
            )

            chat_template_kwargs["enable_thinking"] = (
                request.thinking
            )

            extra_body["chat_template_kwargs"] = (
                chat_template_kwargs
            )

        elif self.thinking_control == "reasoning":
            # Gateways that expose a unified reasoning switch enable
            # it by default for reasoning models, and the effort they
            # pick can consume the whole output budget before a single
            # answer token is produced. The benchmark asks for
            # thinking=False, so it has to be said explicitly.
            reasoning = dict(
                extra_body.get("reasoning", {})
            )

            reasoning.setdefault(
                "enabled",
                request.thinking,
            )

            extra_body["reasoning"] = reasoning

        if extra_body:
            params["extra_body"] = extra_body

        start = time.perf_counter()

        response = self.client.chat.completions.create(
            **params
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        # An OpenAI-compatible gateway may report an upstream failure
        # inside a 200 response, leaving 'choices' empty. Without this
        # the next line fails with an unhelpful TypeError.
        if not getattr(response, "choices", None):
            raise self._response_error(response)

        choice = response.choices[0]

        content = choice.message.content or ""

        reasoning_content = getattr(
            choice.message,
            "reasoning_content",
            None,
        ) or getattr(
            choice.message,
            "reasoning",
            None,
        )

        raw_text = content

        if reasoning_content:
            raw_text = (
                f"<think>\n"
                f"{reasoning_content}\n"
                f"</think>\n\n"
                f"{content}"
            ).strip()

        usage = TokenUsage()

        if response.usage is not None:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return GenerationResponse(
            text=content.strip(),
            raw_text=raw_text,
            model=self.model,
            provider=self.provider,
            finish_reason=choice.finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            metadata={
                "thinking": request.thinking,
            },
            raw_response=response,
        )

    def _response_error(self, response) -> ProviderError:
        """
        Turn a response without choices into a readable error.

        The upstream status code is preserved so that the runner can
        retry a rate limit and give up on a bad request.
        """

        error = getattr(response, "error", None)

        if error is None and isinstance(response, dict):
            error = response.get("error")

        message = None
        status_code = None

        if isinstance(error, dict):
            message = error.get("message")
            status_code = error.get("code")

        elif error is not None:
            message = str(error)

        if not isinstance(status_code, int):
            status_code = None

        return ProviderError(
            f"Provider '{self.provider}' returned no completion for "
            f"model '{self.model}': "
            f"{message or error or 'empty response'}",
            status_code=status_code,
        )