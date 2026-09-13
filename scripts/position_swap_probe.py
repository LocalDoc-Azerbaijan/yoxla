"""
Ask whether a two-sentence task is measuring the sentences or the
position they sit in.

Two models of different sizes answered `az_tr_interference_v1` the
same lopsided way: 98% right when the Azerbaijani sentence was A, 48
to 57% right when it was B. Position is balanced in the data - exactly
100 of each, checked - so the asymmetry belongs to the models or to
something in the items that travels with position.

The two readings are separable by one experiment. Run the same items
with the two sentences swapped:

    the bias is positional   accuracy stays, A stays near 98%, and the
                             items that were right are now the ones
                             that are wrong
    the bias is the items    accuracy stays and the same items stay
                             right, whichever slot they sit in

Nothing is published or written to the benchmark. The swapped copy
lives in memory for one run, and the report prints both arrangements
side by side.

Usage:

    python scripts/position_swap_probe.py \\
        --provider openrouter --model qwen/qwen3.8-27b
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

from yoxla.benchmark.evaluators import create_evaluator
from yoxla.benchmark.loaders.huggingface import load_examples
from yoxla.benchmark.parsing import parse_prediction
from yoxla.benchmark.prompting import render_prompt
from yoxla.benchmark.registry import get_task
from yoxla.benchmark.schema import BenchmarkExample
from yoxla.inference import GenerationRequest, create_model

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", line_buffering=True
)


def swapped(example: BenchmarkExample) -> BenchmarkExample:
    """
    The same item with its two sentences exchanged.

    The label moves with the sentence, so the item still asks the
    same question of the same pair of sentences - only the slot each
    occupies changes.
    """

    inputs = dict(example.inputs)

    inputs["sentence_a"], inputs["sentence_b"] = (
        example.inputs["sentence_b"],
        example.inputs["sentence_a"],
    )

    return BenchmarkExample(
        id=example.id,
        task_id=example.task_id,
        inputs=inputs,
        gold="B" if example.gold == "A" else "A",
        metadata=example.metadata,
    )


def answer(model, task, example) -> str | None:
    rendered = render_prompt(task, example)

    response = model.generate(
        GenerationRequest(
            messages=rendered.messages(),
            max_output_tokens=task.generation.max_output_tokens,
            temperature=task.generation.temperature,
            top_p=task.generation.top_p,
            thinking=task.generation.thinking,
        )
    )

    return parse_prediction(task, response.text).value


def run(model, task, examples, label: str) -> dict[str, object]:
    print(f"\n{label}")

    correct: dict[str, bool] = {}

    answers: list[str | None] = []

    for position, example in enumerate(examples, 1):
        given = answer(model, task, example)

        answers.append(given)

        correct[example.id] = given == example.gold

        if position % 25 == 0:
            print(
                f"   {position}/{len(examples)} "
                f"({sum(correct.values())} right)"
            )

    by_gold = Counter()
    right_by_gold = Counter()

    for example in examples:
        by_gold[example.gold] += 1

        if correct[example.id]:
            right_by_gold[example.gold] += 1

    accuracy = sum(correct.values()) / len(examples)

    said = Counter(a for a in answers if a is not None)

    print(f"   accuracy {accuracy * 100:.1f}%")

    for gold in sorted(by_gold):
        print(
            f"   gold {gold}: "
            f"{right_by_gold[gold]}/{by_gold[gold]} right"
        )

    if said:
        top, count = said.most_common(1)[0]

        print(
            f"   answered {top} "
            f"{count / len(examples) * 100:.1f}% of the time"
        )

    return {"accuracy": accuracy, "correct": correct}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task", default="az_tr_interference_v1"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output",
        default="runs/position_swap_probe.json",
    )

    args = parser.parse_args()

    task = get_task(args.task)

    examples = load_examples(task)

    if args.limit:
        examples = examples[: args.limit]

    model = create_model(
        provider_name=args.provider,
        model=args.model,
        config_path=args.config,
    )

    print(f"{args.model} on {task.task_id}, {len(examples)} items")

    original = run(model, task, examples, "as published")

    flipped = run(
        model,
        task,
        [swapped(example) for example in examples],
        "with the two sentences swapped",
    )

    # The question the whole probe exists for: are the items that
    # survive the swap the same ones, or does the slot decide?
    agree = sum(
        1
        for example in examples
        if original["correct"][example.id]
        == flipped["correct"][example.id]
    )

    both_right = sum(
        1
        for example in examples
        if original["correct"][example.id]
        and flipped["correct"][example.id]
    )

    print()
    print("=" * 56)
    print(
        f"accuracy      {original['accuracy'] * 100:.1f}%  ->  "
        f"{flipped['accuracy'] * 100:.1f}%"
    )
    print(
        f"same verdict  {agree}/{len(examples)} items "
        f"({agree / len(examples) * 100:.1f}%)"
    )
    print(
        f"right in both {both_right}/{len(examples)}"
    )
    print()
    print(
        "An item the model reads rather than guesses is right in\n"
        "both arrangements. If 'same verdict' sits near chance, the\n"
        "model is answering by slot and the task's score is a\n"
        "position measurement wearing a lexical label."
    )

    path = Path(args.output)

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            {
                "task": task.task_id,
                "model": args.model,
                "provider": args.provider,
                "items": len(examples),
                "accuracy_as_published": original["accuracy"],
                "accuracy_swapped": flipped["accuracy"],
                "same_verdict": agree,
                "right_in_both": both_right,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
