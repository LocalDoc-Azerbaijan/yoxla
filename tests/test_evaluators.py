import pytest

from yoxla.benchmark.evaluators import create_evaluator
from yoxla.benchmark.evaluators.extractive_qa import score_answer
from yoxla.benchmark.evaluators.correlation import pearson, spearman
from yoxla.benchmark.parsing import parse_prediction
from yoxla.benchmark.registry import TASKS
from yoxla.benchmark.schema import BenchmarkExample


def make_example(task_id, gold, metadata=None, context=None):
    return BenchmarkExample(
        id=f"{task_id}_x",
        task_id=task_id,
        inputs={"context": context} if context else {},
        gold=gold,
        metadata=metadata or {},
    )


def evaluate_with_context(task, triples):
    """
    For tasks whose answer is located inside a passage.
    """

    evaluator = create_evaluator(task)

    for context, gold, answer in triples:
        evaluator.add(
            make_example(task.task_id, gold, context=context),
            parse_prediction(task, answer),
        )

    return evaluator.compute()


def evaluate(task, pairs):
    evaluator = create_evaluator(task)

    for gold, answer in pairs:
        evaluator.add(
            make_example(task.task_id, gold),
            parse_prediction(task, answer),
        )

    return evaluator.compute()


# ==================================================================
# Classification
# ==================================================================


def test_classification_counts_invalid_output_as_wrong():
    task = TASKS["nli_v1"]

    metrics = evaluate(
        task,
        [
            ("neutral", "neutral"),
            ("entailment", "entailment"),
            ("contradiction", "The answer is contradiction"),
            ("neutral", "entailment"),
        ],
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["invalid_output_rate"] == pytest.approx(
        0.25
    )
    assert metrics["count"] == 4


def test_per_class_accuracy_is_recall_per_gold_label():
    task = TASKS["sentiment_v1"]

    metrics = evaluate(
        task,
        [
            ("positive", "positive"),
            ("positive", "neutral"),
            ("negative", "negative"),
        ],
    )

    assert metrics["per_class_accuracy"] == {
        "positive": pytest.approx(0.5),
        "negative": pytest.approx(1.0),
    }

    assert 0.0 < metrics["macro_f1"] <= 1.0


def test_intent_group_accuracy_is_reported():
    task = TASKS["intent_v1"]

    metrics = evaluate(
        task,
        [
            # right group, wrong intent
            ("declined_transfer", "failed_transfer"),
            # wrong group
            ("declined_transfer", "card_arrival"),
        ],
    )

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["group_accuracy"] == pytest.approx(0.5)


# ==================================================================
# Answer comparison
#
# The open-answer knowledge task is gone, but these functions are
# not: span_qa scores a quoted span with score_answer, so the
# normalization rules below still decide live scores.
# ==================================================================


def test_case_and_whitespace_are_not_differences():
    assert score_answer("5  dəfə", "5 dəfə")[
        "normalized_em"
    ] == pytest.approx(1.0)

    assert score_answer("BAKI", "Bakı")[
        "normalized_em"
    ] == pytest.approx(1.0)


def test_azerbaijani_letters_are_not_folded_to_ascii():
    """
    "Gence" for "Gəncə" is a misspelling, not a spelling variant.
    Folding the diacritics would reward exactly the mistake an
    Azerbaijani benchmark exists to catch.
    """

    assert score_answer("Gence", "Gəncə")[
        "normalized_em"
    ] == pytest.approx(0.0)


def test_final_punctuation_is_not_a_difference():
    scores = score_answer(
        "qəzəb, qorxu, sevinc.", "qəzəb, qorxu, sevinc"
    )

    assert scores["normalized_em"] == pytest.approx(1.0)
    assert scores["token_f1"] == pytest.approx(1.0)


def test_an_answer_inside_a_sentence_is_not_an_exact_match():
    scores = score_answer("cavab 5 dəfə olmuşdur", "5 dəfə")

    assert scores["normalized_em"] == pytest.approx(0.0)
    assert scores["containment"] == pytest.approx(1.0)
    assert scores["token_f1"] > 0.0


def test_classification_reports_a_per_phenomenon_breakdown():
    task = TASKS["minimal_pairs_v1"]

    evaluator = create_evaluator(task)

    cases = [
        ("vowel_harmony_suffix", "A", "A"),
        ("vowel_harmony_suffix", "B", "B"),
        ("question_particle", "A", "B"),
        ("question_particle", "B", "B"),
    ]

    for phenomenon, gold, answer in cases:
        evaluator.add(
            make_example(
                task.task_id,
                gold,
                {"phenomenon": phenomenon},
            ),
            parse_prediction(task, answer),
        )

    metrics = evaluator.compute()

    # One number over eight grammatical rules would hide that the
    # model handles harmony and not the question particle.
    assert metrics["accuracy"] == pytest.approx(0.75)

    assert metrics["per_phenomenon_accuracy"] == {
        "vowel_harmony_suffix": pytest.approx(1.0),
        "question_particle": pytest.approx(0.5),
    }


def test_one_repeated_answer_is_visible_not_averaged_in():
    """
    A model with no opinion still has to answer, and it answers the
    same way every time. Accuracy reads that as merely weak; this
    reads it as not answering.
    """

    task = TASKS["knowledge_choice_v1"]

    evaluator = create_evaluator(task)

    # Nine defaults to option 1, one real choice.
    for index in range(9):
        evaluator.add(
            make_example(task.task_id, "7"),
            parse_prediction(task, "1"),
        )

    evaluator.add(
        make_example(task.task_id, "7"),
        parse_prediction(task, "7"),
    )

    metrics = evaluator.compute()

    assert metrics["accuracy"] == pytest.approx(0.1)
    assert metrics["modal_answer_share"] == pytest.approx(0.9)


def test_a_model_that_spreads_its_answers_has_a_low_modal_share():
    task = TASKS["knowledge_choice_v1"]

    evaluator = create_evaluator(task)

    for value in ("1", "5", "9", "14"):
        evaluator.add(
            make_example(task.task_id, "5"),
            parse_prediction(task, value),
        )

    assert evaluator.compute()[
        "modal_answer_share"
    ] == pytest.approx(0.25)


def test_a_task_without_a_breakdown_field_reports_none():
    metrics = evaluate(
        TASKS["nli_v1"],
        [("entailment", "entailment")],
    )

    assert not any(
        key.startswith("per_") and key.endswith("_accuracy")
        and key != "per_class_accuracy"
        for key in metrics
    )


# ==================================================================
# QA abstention
# ==================================================================


def test_abstention_scores_both_decisions():
    task = TASKS["qa_abstention_v1"]

    passage = (
        "Metro 2004-cü ildə açıldı. Stansiya Bakı şəhərindədir."
    )

    metrics = evaluate_with_context(
        task,
        [
            # unanswerable, correctly refused
            (passage, None, "Cavab yoxdur"),
            # unanswerable, answered anyway
            (passage, None, "2004-cü ildə"),
            # answerable, answered correctly
            (passage, "2004-cü ildə", "2004-cü ildə"),
            # answerable, wrongly refused
            (passage, "Bakı", "Cavab yoxdur"),
        ],
    )

    assert metrics[
        "answerability_accuracy"
    ] == pytest.approx(0.5)
    assert metrics[
        "unanswerable_accuracy"
    ] == pytest.approx(0.5)

    # Two of the four score: the correctly refused unanswerable
    # one, and the answerable one whose quote covers its span. The
    # other two are answerability errors.
    assert metrics["overall_score"] == pytest.approx(0.5)


def test_invalid_output_is_not_an_abstention():
    task = TASKS["qa_abstention_v1"]

    metrics = evaluate_with_context(
        task,
        [("Bakı Azərbaycanın paytaxtıdır.", None, "")],
    )

    assert metrics["overall_score"] == pytest.approx(0.0)
    assert metrics["invalid_output_rate"] == pytest.approx(
        1.0
    )


# ==================================================================
# STS
# ==================================================================


def test_sts_perfect_ranking():
    task = TASKS["sts_v1"]

    metrics = evaluate(
        task,
        [
            (0.0, "0"),
            (1.0, "1"),
            (3.0, "3"),
            (5.0, "5"),
        ],
    )

    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_sts_invalid_outputs_lower_the_correlation():
    task = TASKS["sts_v1"]

    clean = evaluate(
        task,
        [
            (0.0, "0"),
            (1.0, "1"),
            (3.0, "3"),
            (5.0, "5"),
        ],
    )

    broken = evaluate(
        task,
        [
            (0.0, "0"),
            (1.0, "1"),
            (3.0, "Score: 3"),
            (5.0, "təxminən 5"),
        ],
    )

    assert broken["invalid_output_rate"] == pytest.approx(
        0.5
    )

    # The two valid answers alone would still look perfect.
    assert broken[
        "spearman_valid_only"
    ] == pytest.approx(1.0)

    assert broken["spearman"] < clean["spearman"]


def test_constant_predictions_score_zero():
    task = TASKS["sts_v1"]

    metrics = evaluate(
        task,
        [
            (0.0, "3"),
            (1.0, "3"),
            (5.0, "3"),
        ],
    )

    assert metrics["spearman"] == pytest.approx(0.0)


# ==================================================================
# Correlation helpers
# ==================================================================


def test_pearson_and_spearman_basics():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(
        1.0
    )

    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(
        -1.0
    )

    # Monotone but not linear: Spearman still sees a perfect rank
    # agreement while Pearson does not.
    assert spearman(
        [1, 2, 3, 4], [1, 4, 9, 16]
    ) == pytest.approx(1.0)

    assert pearson([1, 1, 1], [1, 2, 3]) is None
