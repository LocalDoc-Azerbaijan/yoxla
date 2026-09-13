import pytest

from yoxla.benchmark.evaluators import create_evaluator
from yoxla.benchmark.evaluators.span import (
    flatten,
    locate_gold,
    score_span,
    sentence_window,
)
from yoxla.benchmark.parsing import parse_prediction
from yoxla.benchmark.registry import TASKS
from yoxla.benchmark.schema import BenchmarkExample

TASK = TASKS["extractive_qa_v1"]

PASSAGE = (
    "Bakı Azərbaycanın paytaxtıdır. "
    "Şəhər Xəzər dənizinin sahilində yerləşir. "
    "Metro 1967-ci ildə açılmışdır."
)


def score(gold, answer, context=PASSAGE, metadata=None):
    evaluator = create_evaluator(TASK)

    return evaluator.add(
        BenchmarkExample(
            id="sample",
            task_id=TASK.task_id,
            inputs={"context": context},
            gold=gold,
            metadata=metadata or {},
        ),
        parse_prediction(TASK, answer),
    )


# ==================================================================
# Recall and precision are separate abilities
# ==================================================================


def test_an_exact_quote_scores_both_perfectly():
    metrics = score("Bakı", "Bakı")

    assert metrics["span_recall"] == pytest.approx(1.0)
    assert metrics["span_precision"] == pytest.approx(1.0)
    assert metrics["exact_span"] == pytest.approx(1.0)


def test_quoting_more_than_the_answer_keeps_full_recall():
    # The model found Baku. That it dragged the category noun along
    # is a formatting difference, and it belongs in precision - this
    # is the case the old single score punished as a miss.
    metrics = score("Bakı", "Bakı Azərbaycanın")

    assert metrics["span_recall"] == pytest.approx(1.0)
    assert metrics["span_precision"] < 0.5
    assert metrics["exact_span"] == pytest.approx(0.0)


def test_quoting_less_than_the_answer_loses_recall():
    # Half the answer is half an answer, and precision is perfect
    # because every character quoted was inside the span.
    metrics = score("Xəzər dənizinin", "Xəzər")

    assert metrics["span_recall"] < 0.5
    assert metrics["span_precision"] == pytest.approx(1.0)


def test_the_wrong_part_of_the_passage_scores_nothing():
    metrics = score("Bakı", "Xəzər dənizinin")

    assert metrics["span_recall"] == pytest.approx(0.0)
    assert metrics["span_precision"] == pytest.approx(0.0)
    assert metrics["unsupported"] == pytest.approx(0.0)


# ==================================================================
# The guard against quoting everything
# ==================================================================


def test_quoting_the_whole_passage_earns_no_recall():
    # Otherwise recall would be free: copy the input, score 1.0.
    metrics = score("Bakı", PASSAGE)

    assert metrics["span_recall"] == pytest.approx(0.0)
    assert metrics["outside_sentence"] == pytest.approx(1.0)


def test_quoting_the_answer_s_whole_sentence_still_counts():
    # Within one sentence is ordinary behaviour, not gaming.
    metrics = score("Bakı", "Bakı Azərbaycanın paytaxtıdır.")

    assert metrics["span_recall"] == pytest.approx(1.0)
    assert metrics["outside_sentence"] == pytest.approx(0.0)


def test_a_gold_span_crossing_a_sentence_widens_the_window():
    context = "Bakı böyükdür. Şəhər sahildə yerləşir."

    metrics = score(
        "böyükdür. Şəhər",
        "böyükdür. Şəhər",
        context=context,
    )

    assert metrics["span_recall"] == pytest.approx(1.0)
    assert metrics["outside_sentence"] == pytest.approx(0.0)


# ==================================================================
# Answers that were not copied from the passage
# ==================================================================


def test_an_answer_absent_from_the_passage_is_unsupported():
    metrics = score("Bakı", "Gəncə")

    assert metrics["unsupported"] == pytest.approx(1.0)
    assert metrics["span_recall"] == pytest.approx(0.0)


def test_an_empty_answer_is_unsupported():
    assert score("Bakı", "")["unsupported"] == pytest.approx(1.0)


def test_case_and_spacing_do_not_make_a_quote_an_invention():
    # A lowercased copy is a formatting slip; strict_em records it.
    metrics = score("Bakı", "bakı")

    assert metrics["unsupported"] == pytest.approx(0.0)
    assert metrics["span_recall"] == pytest.approx(1.0)

    metrics = score("Xəzər dənizinin", "Xəzər  dənizinin")

    assert metrics["span_recall"] == pytest.approx(1.0)


def test_surrounding_quotation_marks_are_not_part_of_the_span():
    metrics = score("Bakı", '"Bakı"')

    assert metrics["span_recall"] == pytest.approx(1.0)
    assert metrics["span_precision"] == pytest.approx(1.0)


# ==================================================================
# Gold that cannot be scored on offsets
# ==================================================================


def build(gold, context, metadata=None):
    return BenchmarkExample(
        id="sample",
        task_id=TASK.task_id,
        inputs={"context": context},
        gold=gold,
        metadata=metadata or {},
    )


def test_a_gold_missing_from_its_passage_is_refused():
    evaluator = create_evaluator(TASK)

    assert evaluator.validate_gold(build("Gəncə", PASSAGE))


def test_a_gold_occurring_twice_is_refused():
    # Its position is ambiguous, so an offset score would be a guess.
    evaluator = create_evaluator(TASK)

    assert evaluator.validate_gold(
        build("Bakı", "Bakı və Bakı.")
    )


def test_a_disagreeing_answer_start_is_refused():
    evaluator = create_evaluator(TASK)

    assert not evaluator.validate_gold(
        build("Bakı", PASSAGE, {"answer_start": 0})
    )

    assert evaluator.validate_gold(
        build("Bakı", PASSAGE, {"answer_start": 12})
    )


def test_a_well_formed_example_passes():
    evaluator = create_evaluator(TASK)

    assert evaluator.validate_gold(build("Bakı", PASSAGE)) == []


# ==================================================================
# Primitives
# ==================================================================


def test_flatten_collapses_whitespace_and_keeps_case():
    assert flatten("  Bakı\n  şəhəri ") == "Bakı şəhəri"


def test_locate_gold_works_across_a_line_break():
    assert locate_gold("Heydər\nƏliyev", "Heydər Əliyev") == (
        0,
        13,
    )


def test_sentence_window_covers_the_span_only():
    text = "Bir. İki. Üç."

    assert sentence_window(text, 5, 8) == (4, 9)


def test_score_span_reports_absence_rather_than_guessing():
    match = score_span("Bakı şəhəri", "Gəncə", "Bakı")

    assert match.found is False
