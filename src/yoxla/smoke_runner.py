import json
from datetime import datetime, timezone
from pathlib import Path

from yoxla.inference import (
    GenerationRequest,
    Message,
)


def run_dataset(
    model,
    input_path: str | Path,
    output_path: str | Path,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    thinking: bool = False,
    limit: int | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    if limit is not None:
        examples = examples[:limit]

    total = len(examples)

    print(f"Running {total} examples...\n")

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, example in enumerate(
            examples,
            start=1,
        ):
            example_id = example["id"]

            print(
                f"[{index}/{total}] "
                f"{example_id}",
                end=" ... ",
                flush=True,
            )

            messages = [
                Message(
                    role=message["role"],
                    content=message["content"],
                )
                for message in example["messages"]
            ]

            request = GenerationRequest(
                messages=messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                thinking=thinking,
            )

            try:
                response = model.generate(request)

                result = {
                    "example_id": example_id,

                    "model": response.model,
                    "provider": response.provider,

                    "response": response.text,
                    "raw_response": response.raw_text,

                    "finish_reason": response.finish_reason,

                    "usage": {
                        "input_tokens": (
                            response.usage.input_tokens
                        ),
                        "output_tokens": (
                            response.usage.output_tokens
                        ),
                        "total_tokens": (
                            response.usage.total_tokens
                        ),
                    },

                    "latency_ms": response.latency_ms,

                    "generation": {
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "thinking": thinking,
                    },

                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),

                    "error": None,
                }

                print("OK")

            except Exception as exc:
                result = {
                    "example_id": example_id,
                    "model": model.model,
                    "provider": model.provider,
                    "response": None,
                    "raw_response": None,
                    "finish_reason": None,
                    "usage": None,
                    "latency_ms": None,
                    "generation": {
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "thinking": thinking,
                    },
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "error": str(exc),
                }

                print(f"ERROR: {exc}")

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

    print(
        f"\nResults saved to: {output_path}"
    )