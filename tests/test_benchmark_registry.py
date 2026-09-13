import pytest

from yoxla.benchmark.evaluators import EVALUATORS
from yoxla.benchmark.registry import (
    ALL_BLOCKS,
    BENCHMARK_VERSIONS,
    TASKS,
    get_block_tasks,
    resolve_tasks,
)
from yoxla.benchmark.schema import AnswerMode

UNDERSTANDING_SIZES = {
    "extractive_qa_v1": 200,
    "qa_abstention_v1": 100,
    "nli_v1": 100,
    "sts_v1": 100,
    "sentiment_v1": 50,
    "intent_v1": 50,
}


def test_understanding_block_is_600_examples():
    tasks = get_block_tasks("understanding")

    sizes = {
        task.task_id: task.expected_examples
        for task in tasks
    }

    assert sizes == UNDERSTANDING_SIZES
    assert sum(sizes.values()) == 600

    assert BENCHMARK_VERSIONS["understanding"] == (
        "understanding_v1"
    )


BLOCK_SIZES = {
    "understanding": 600,
    "language": 400,
    "knowledge": 150,
    "rag": 293,
}


def test_every_block_has_its_declared_size():
    for block, expected in BLOCK_SIZES.items():
        tasks = get_block_tasks(block)

        assert sum(
            task.expected_examples for task in tasks
        ) == expected

        assert block in BENCHMARK_VERSIONS


def test_every_answer_space_is_closed():
    """
    No task returns free text any more.

    The open-answer knowledge task was the last one, and it went
    because scoring it meant deciding which spellings of a name count
    - a judgement the benchmark has no deterministic way to make.
    Anything reintroducing an open answer space has to answer that
    question first.
    """

    for task in TASKS.values():
        assert task.answer.mode in {
            AnswerMode.CHOICE,
            AnswerMode.NUMERIC,
            AnswerMode.TEXT,
            AnswerMode.TEXT_OR_ABSTAIN,
        }

        if task.answer.mode in {
            AnswerMode.TEXT,
            AnswerMode.TEXT_OR_ABSTAIN,
        }:
            # A quoted span is checked against the passage it was
            # quoted from, so its gold is not a spelling judgement.
            assert task.evaluator in {
                "span_qa",
                "span_abstention",
            }, task.task_id


def test_breakdown_fields_are_requested_from_the_dataset():
    for task_id in (
        "minimal_pairs_v1",
        "knowledge_choice_v1",
        "rag_verification_v1",
    ):
        task = TASKS[task_id]

        assert task.breakdown_field
        assert task.breakdown_field in task.required_columns()


def test_all_selects_every_active_task():
    """
    Running the whole benchmark is the common case and used to need
    ten repeated --task flags.
    """

    every = get_block_tasks(ALL_BLOCKS)

    assert [task.task_id for task in every] == [
        task.task_id
        for task in TASKS.values()
        if not task.retired
    ]

    assert sum(
        task.expected_examples for task in every
    ) == sum(BLOCK_SIZES.values())


def test_an_unknown_block_names_all_among_the_options():
    with pytest.raises(ValueError) as error:
        get_block_tasks("nonexistent")

    assert "all" in str(error.value)


def test_every_task_satisfies_the_task_contract():
    for task in TASKS.values():
        assert task.task_id
        assert task.block
        assert task.dataset_repo
        assert task.dataset_config
        assert task.dataset_split
        assert task.expected_examples > 0

        assert task.input_fields
        assert task.gold_field

        assert task.prompt.version >= 1
        assert task.prompt.system.strip()
        assert task.prompt.user.strip()

        assert task.generation.max_output_tokens > 0
        assert task.generation.temperature == 0.0
        assert task.generation.thinking is False

        assert task.evaluator in EVALUATORS
        assert task.primary_metric

        if task.answer.mode is AnswerMode.CHOICE:
            assert len(task.answer.choices) >= 2

            for choice in task.answer.choices:
                assert choice.description.strip()

        if task.answer.mode is AnswerMode.TEXT_OR_ABSTAIN:
            assert task.answer.abstain_value

        if task.answer.mode is AnswerMode.NUMERIC:
            assert task.answer.rubric
            assert task.answer.minimum is not None
            assert task.answer.maximum is not None


def test_intent_choices_are_unique_and_grouped():
    task = TASKS["intent_v1"]

    values = task.answer.choice_values()

    assert len(values) == 25
    assert len(set(values)) == 25

    assert len(task.answer.choice_groups()) == 25


def test_no_task_is_retired_before_publication():
    # A task id is frozen from its first publication, not from its
    # first commit, so nothing here is carrying an old run.
    assert not any(task.retired for task in TASKS.values())

    assert len(TASKS) == sum(
        len(get_block_tasks(block)) for block in BLOCK_SIZES
    )


def test_resolve_tasks_requires_a_selection():
    assert resolve_tasks(task_ids=["nli_v1"]) == [
        TASKS["nli_v1"]
    ]

    try:
        resolve_tasks()

    except ValueError:
        pass

    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
