"""
Deterministic prompt rendering.

Prompts belong to the benchmark task, never to an inference adapter.
Rendering is pure: the same task and example always produce the same
messages, and the resulting hashes are recorded with every run so a
leaderboard entry can be reconstructed exactly.
"""

import hashlib
from dataclasses import dataclass

from yoxla.benchmark.schema import BenchmarkExample, BenchmarkTask
from yoxla.inference import Message


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str

    system_hash: str
    user_hash: str

    def messages(self) -> list[Message]:
        return [
            Message(role="system", content=self.system),
            Message(role="user", content=self.user),
        ]


def text_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def render_choice_catalog(task: BenchmarkTask) -> str:
    """
    Render the allowed labels of a CHOICE task.

    Choices live in the task, not in dataset rows: the intent task
    would otherwise repeat the same 25 labels in every example.
    """

    return "\n".join(
        f"- {choice.value}: {choice.description}"
        for choice in task.answer.choices
    )


def render_rubric_catalog(task: BenchmarkTask) -> str:
    return "\n".join(
        f"{level.value} — {level.description}"
        for level in task.answer.rubric
    )


def render_option_list(values) -> str:
    """
    Number a per-example list of options.

    A choice whose options differ from example to example cannot live
    in the task contract the way a label catalogue does, so it arrives
    as a dataset column. Numbering it here keeps the answer space a
    small closed set of labels while the options themselves vary.
    """

    return "\n".join(
        f"{index}. {value}"
        for index, value in enumerate(values, 1)
    )


def build_variables(
    task: BenchmarkTask,
    example: BenchmarkExample,
) -> dict[str, str]:
    variables: dict[str, str] = {
        field: (
            render_option_list(value)
            if isinstance(value, (list, tuple))
            else str(value)
        )
        for field, value in example.inputs.items()
    }

    if task.answer.choices:
        variables["choice_catalog"] = (
            render_choice_catalog(task)
        )

        variables["choice_values"] = ", ".join(
            task.answer.choice_values()
        )

    if task.answer.rubric:
        variables["rubric_catalog"] = (
            render_rubric_catalog(task)
        )

    if task.answer.abstain_value is not None:
        variables["abstain_value"] = (
            task.answer.abstain_value
        )

    return variables


def _format(
    template: str,
    variables: dict[str, str],
    task_id: str,
) -> str:
    try:
        return template.format(**variables)

    except KeyError as exc:
        raise ValueError(
            f"Prompt of task '{task_id}' references unknown "
            f"variable {exc}."
        ) from exc


def render_system_prompt(task: BenchmarkTask) -> str:
    """
    Render the system prompt on its own.

    System prompts never reference example fields - only the choice
    catalog, the rubric and the abstain sentinel - so a task has
    exactly one system prompt, and its hash identifies the frozen
    instruction text.
    """

    empty = BenchmarkExample(
        id="",
        task_id=task.task_id,
        inputs={},
        gold=None,
        metadata={},
    )

    return _format(
        task.prompt.system,
        build_variables(task, empty),
        task.task_id,
    )


def render_prompt(
    task: BenchmarkTask,
    example: BenchmarkExample,
) -> RenderedPrompt:
    variables = build_variables(task, example)

    system = _format(
        task.prompt.system,
        variables,
        task.task_id,
    )

    user = _format(
        task.prompt.user,
        variables,
        task.task_id,
    )

    return RenderedPrompt(
        system=system,
        user=user,
        system_hash=text_hash(system),
        user_hash=text_hash(user),
    )


def render_messages(
    task: BenchmarkTask,
    example: BenchmarkExample,
) -> list[Message]:
    return render_prompt(task, example).messages()
