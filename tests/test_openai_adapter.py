from types import SimpleNamespace

import pytest

from yoxla.inference.adapters import OpenAICompatibleAdapter
from yoxla.inference.errors import ProviderError
from yoxla.inference.schemas import GenerationRequest, Message
from yoxla.runner import is_retryable

REQUEST = GenerationRequest(
    messages=[Message(role="user", content="Salam")],
    max_output_tokens=16,
)


class FakeCompletions:
    def __init__(self, response):
        self.response = response

        self.params = None

    def create(self, **params):
        self.params = params

        return self.response


def build_adapter(response, **kwargs):
    adapter = OpenAICompatibleAdapter(
        model="test/model",
        provider="openrouter",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        **kwargs,
    )

    completions = FakeCompletions(response)

    adapter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    return adapter, completions


def completion(
    content,
    reasoning=None,
    reasoning_content=None,
    finish_reason="stop",
):
    message = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
    )

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
        ),
    )


def test_reasoning_is_switched_off_explicitly():
    adapter, completions = build_adapter(
        completion("Bakı"),
        thinking_control="reasoning",
    )

    adapter.generate(REQUEST)

    assert completions.params["extra_body"] == {
        "reasoning": {"enabled": False}
    }


def test_reasoning_is_switched_on_when_requested():
    adapter, completions = build_adapter(
        completion("Bakı"),
        thinking_control="reasoning",
    )

    adapter.generate(
        GenerationRequest(
            messages=REQUEST.messages,
            max_output_tokens=16,
            thinking=True,
        )
    )

    assert completions.params["extra_body"] == {
        "reasoning": {"enabled": True}
    }


def test_no_reasoning_parameter_without_the_control():
    adapter, completions = build_adapter(completion("Bakı"))

    adapter.generate(REQUEST)

    assert "extra_body" not in completions.params


def test_reasoning_text_is_kept_in_raw_output():
    adapter, _ = build_adapter(
        completion("Bakı", reasoning="düşünürəm..."),
        thinking_control="reasoning",
    )

    response = adapter.generate(REQUEST)

    assert response.text == "Bakı"
    assert "düşünürəm..." in response.raw_text
    assert response.raw_text.startswith("<think>")


def test_response_without_choices_raises_a_readable_error():
    adapter, _ = build_adapter(
        SimpleNamespace(
            choices=None,
            error={
                "message": "Provider returned error",
                "code": 429,
            },
            usage=None,
        )
    )

    with pytest.raises(ProviderError) as raised:
        adapter.generate(REQUEST)

    assert "Provider returned error" in str(raised.value)
    assert "test/model" in str(raised.value)

    # Wrapped upstream rate limits stay retryable.
    assert raised.value.status_code == 429
    assert is_retryable(raised.value)


def test_bad_request_reported_in_the_body_is_not_retried():
    adapter, _ = build_adapter(
        SimpleNamespace(
            choices=[],
            error={"message": "No such model", "code": 400},
            usage=None,
        )
    )

    with pytest.raises(ProviderError) as raised:
        adapter.generate(REQUEST)

    assert not is_retryable(raised.value)
