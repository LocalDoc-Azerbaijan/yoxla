from yoxla.benchmark.prompting import (
    render_prompt,
    render_system_prompt,
)
from yoxla.benchmark.registry import TASKS
from yoxla.benchmark.schema import BenchmarkExample

QA_INPUTS = {
    "context": "Bakı Azərbaycanın paytaxtıdır.",
    "question": "Azərbaycanın paytaxtı haradır?",
}

ABSTENTION_INPUTS = {
    "context": "Bakı Azərbaycanın paytaxtıdır.",
    "question": "Şəhərin əhalisi nə qədərdir?",
}

SAMPLE_INPUTS = {
    "extractive_qa_v1": QA_INPUTS,
    "qa_abstention_v1": ABSTENTION_INPUTS,
    "nli_v1": {
        "premise": "Yorğun bir kişi yarışı bitirir.",
        "hypothesis": "Kişi sonuncu yerdədir.",
    },
    "sts_v1": {
        "sentence1": "Çöldə oturan iki qadın gülür.",
        "sentence2": "İki qadın avtobusun qarşısındadır.",
    },
    "sentiment_v1": {
        "text": "Tortları çox dadlıdır.",
    },
    "intent_v1": {
        "text": "Köçürməm rədd edildi.",
    },
    "minimal_pairs_v1": {
        "sentence_a": "Tələbələr kitabxanada dərs oxuyurlar.",
        "sentence_b": "Tələbələr kitabxanada dərs oxuyurler.",
    },
    "az_tr_interference_v1": {
        "sentence_a": "Uşaqlar məktəbdə çox gözəl danışmaq öyrənirlər.",
        "sentence_b": "Uşaqlar məktəbdə çox gözəl konuşmak öyrənirlər.",
    },
    "rag_selection_v1": {
        "question": "Nizami Gəncəvi harada anadan olmuşdur?",
        "passages": [
            "Nizami Gəncəvi 1141-ci ildə anadan olmuşdur.",
            "Nizami Gəncəvi Gəncə şəhərində dünyaya gəlmişdir.",
        ],
    },
    "rag_verification_v1": {
        "context": "Bakı 1918-ci ildə paytaxt elan edilmişdir.",
        "claim": "Bakı 1920-ci ildə paytaxt olmuşdur.",
    },
    "knowledge_choice_v1": {
        "question": "Nizami Gəncəvi hansı şəhərdə anadan olmuşdur?",
        "candidates": ["Gəncə", "Şamaxı", "Bakı", "Şəki"],
    },
}


def build_example(task_id: str) -> BenchmarkExample:
    return BenchmarkExample(
        id=f"{task_id}_sample",
        task_id=task_id,
        inputs=SAMPLE_INPUTS[task_id],
        gold=None,
        metadata={},
    )


def test_every_task_renders_two_messages():
    for task_id, task in TASKS.items():
        rendered = render_prompt(
            task,
            build_example(task_id),
        )

        messages = rendered.messages()

        assert [message.role for message in messages] == [
            "system",
            "user",
        ]

        assert "{" not in rendered.system
        assert "{" not in rendered.user

        for value in SAMPLE_INPUTS[task_id].values():
            if isinstance(value, list):
                # A per-example option list is numbered, not pasted.
                for option in value:
                    assert option in rendered.user

                continue

            assert value in rendered.user


def test_a_per_example_option_list_is_numbered():
    """
    Options that change per example cannot live in the task contract
    the way a label catalogue does, so they arrive as a column and are
    numbered at render time.
    """

    rendered = render_prompt(
        TASKS["knowledge_choice_v1"],
        build_example("knowledge_choice_v1"),
    )

    assert "1. Gəncə" in rendered.user
    assert "4. Şəki" in rendered.user

    # The raw list must never reach the model.
    assert "[" not in rendered.user


def test_choice_catalog_is_rendered_from_the_task():
    task = TASKS["nli_v1"]

    system = render_system_prompt(task)

    for choice in task.answer.choices:
        assert f"- {choice.value}: " in system
        assert choice.description in system


def test_intent_prompt_lists_all_25_intents():
    task = TASKS["intent_v1"]

    system = render_system_prompt(task)

    for choice in task.answer.choices:
        assert choice.value in system
        assert choice.description in system


def test_rubric_is_rendered_for_sts():
    task = TASKS["sts_v1"]

    system = render_system_prompt(task)

    for level in task.answer.rubric:
        assert level.description in system


def test_the_span_prompt_states_the_granularity():
    """
    Left open, the instruction lets a model answer with the whole
    sentence containing the answer and still obey it - which the
    score then reads as a comprehension failure.
    """

    system = render_system_prompt(TASKS["extractive_qa_v1"])

    assert "ən qısa mətn parçasını" in system
    assert "Cümləni bütövlükdə yazma" in system
    assert "eynilə köçür" in system


def test_abstention_keeps_its_original_prompt():
    """
    The minimality rule belongs to extractive QA only.

    Two wordings were tried here and neither beat the original: the
    differences turned out to sit inside one model's run-to-run
    variance on this task.
    """

    system = render_system_prompt(
        TASKS["qa_abstention_v1"]
    )

    assert "ən qısa" not in system
    assert TASKS["qa_abstention_v1"].prompt.version == 1


def test_abstain_sentinel_appears_in_the_prompt():
    task = TASKS["qa_abstention_v1"]

    system = render_system_prompt(task)

    assert task.answer.abstain_value in system


def test_rendering_is_deterministic():
    task = TASKS["nli_v1"]

    example = build_example("nli_v1")

    first = render_prompt(task, example)
    second = render_prompt(task, example)

    assert first.system_hash == second.system_hash
    assert first.user_hash == second.user_hash

    other = render_prompt(
        task,
        BenchmarkExample(
            id="other",
            task_id="nli_v1",
            inputs={
                "premise": "Başqa cümlə.",
                "hypothesis": "Başqa hipotez.",
            },
            gold=None,
            metadata={},
        ),
    )

    assert other.user_hash != first.user_hash
    assert other.system_hash == first.system_hash
