import time
from typing import Any

from yoxla.inference.base import ModelAdapter, ModelCapabilities
from yoxla.inference.schemas import (
    GenerationRequest,
    GenerationResponse,
    TokenUsage,
)


class TransformersAdapter(ModelAdapter):
    """
    Adapter for local Hugging Face causal language models.
    """

    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        capabilities: ModelCapabilities | None = None,
    ):
        super().__init__(
            model=model,
            provider="transformers",
            capabilities=capabilities,
        )

        if load_in_4bit and load_in_8bit:
            raise ValueError(
                "load_in_4bit and load_in_8bit cannot both be enabled."
            )

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend requires local dependencies. "
                'Install them with: pip install -e ".[local]"'
            ) from exc

        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        if dtype == "auto":
            model_kwargs["dtype"] = "auto"
        else:
            torch_dtype = getattr(torch, dtype, None)

            if torch_dtype is None:
                raise ValueError(
                    f"Unknown torch dtype: {dtype}"
                )

            model_kwargs["dtype"] = torch_dtype

        if load_in_4bit:
            model_kwargs["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            )

        elif load_in_8bit:
            model_kwargs["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            )

        self.model_instance = (
            AutoModelForCausalLM.from_pretrained(
                model,
                **model_kwargs,
            )
        )

        self.model_instance.eval()

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

        # Qwen3-like models support enable_thinking.
        # For models that ignore it, chat template behavior
        # remains model-dependent.
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=request.thinking,
                **template_kwargs,
            )
        except TypeError:
            # Models whose chat template does not support
            # enable_thinking.
            inputs = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

        device = self._input_device()

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_tokens = inputs["input_ids"].shape[-1]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_output_tokens,
        }

        if (
            request.temperature is not None
            and request.temperature > 0
        ):
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = (
                request.temperature
            )

            if request.top_p is not None:
                generation_kwargs["top_p"] = (
                    request.top_p
                )

        else:
            # Deterministic benchmark mode.
            generation_kwargs["do_sample"] = False

            # Remove sampling parameters inherited from
            # model generation_config.
            generation_kwargs["temperature"] = None
            generation_kwargs["top_p"] = None
            generation_kwargs["top_k"] = None

        if request.extra:
            generation_kwargs.update(
                request.extra
            )

        start = time.perf_counter()

        with self.torch.inference_mode():
            output = self.model_instance.generate(
                **inputs,
                **generation_kwargs,
            )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        generated_tokens = output[0][input_tokens:]

        output_tokens = len(generated_tokens)

        raw_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        final_text = self._extract_final_text(
            raw_text
        )

        finish_reason = self._get_finish_reason(
            generated_tokens=generated_tokens,
            max_output_tokens=request.max_output_tokens,
        )

        return GenerationResponse(
            text=final_text,
            raw_text=raw_text,
            model=self.model,
            provider=self.provider,
            finish_reason=finish_reason,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            latency_ms=latency_ms,
            metadata={
                "backend": "transformers",
                "thinking": request.thinking,
            },
        )

    def _extract_final_text(
        self,
        raw_text: str,
    ) -> str:
        """
        Remove reasoning blocks from models that return:

        <think>
        ...
        </think>

        final answer
        """

        closing_tag = "</think>"

        if closing_tag in raw_text:
            return raw_text.rsplit(
                closing_tag,
                1,
            )[1].strip()

        # Model was truncated while still reasoning.
        # Do not treat reasoning text as the final answer.
        if raw_text.lstrip().startswith("<think>"):
            return ""

        return raw_text.strip()

    def _get_finish_reason(
        self,
        generated_tokens,
        max_output_tokens: int,
    ) -> str:
        """
        Normalize Transformers stopping reasons to:
        - stop
        - length
        """

        if len(generated_tokens) == 0:
            return "stop"

        eos_token_id = (
            self.model_instance
            .generation_config
            .eos_token_id
        )

        eos_ids: set[int] = set()

        if isinstance(eos_token_id, int):
            eos_ids.add(eos_token_id)

        elif isinstance(
            eos_token_id,
            (list, tuple),
        ):
            eos_ids.update(eos_token_id)

        last_token_id = int(
            generated_tokens[-1].item()
        )

        if last_token_id in eos_ids:
            return "stop"

        if len(generated_tokens) >= max_output_tokens:
            return "length"

        return "stop"

    def _input_device(self):
        try:
            return self.model_instance.device

        except AttributeError:
            return next(
                self.model_instance.parameters()
            ).device