import pytest

from yoxla.benchmark.parsing import parse_prediction
from yoxla.benchmark.registry import TASKS

NLI = TASKS["nli_v1"]
STS = TASKS["sts_v1"]
QA = TASKS["extractive_qa_v1"]
ABSTENTION = TASKS["qa_abstention_v1"]


@pytest.mark.parametrize(
    "text",
    [
        "neutral",
        " neutral ",
        "`neutral`",
        '"neutral"',
        "**neutral**",
        "Neutral",
        "NEUTRAL",
    ],
)
def test_choice_accepts_only_the_bare_label(text):
    parsed = parse_prediction(NLI, text)

    assert parsed.valid
    assert parsed.value == "neutral"


@pytest.mark.parametrize(
    "text",
    [
        "Neutral.",
        "The answer is neutral",
        "neutral - the second sentence may be true",
        "cavab: neutral",
        "",
        "   ",
    ],
)
def test_choice_rejects_anything_else(text):
    parsed = parse_prediction(NLI, text)

    assert not parsed.valid
    assert parsed.value is None
    assert parsed.error in {
        "not_one_of_allowed_choices",
        "empty_output",
    }


CHOICE = TASKS["knowledge_choice_v1"]


@pytest.mark.parametrize(
    "text",
    [
        # A number with a full stop is not the number.
        "2.",
        # The option line the prompt printed, echoed back.
        "3. 1873",
        # The same, cut off by the token budget.
        "1. Rauf Əliy",
        # The option named instead of numbered.
        "Gəncə",
        "cavab: 2",
    ],
)
def test_a_numbered_choice_takes_the_bare_number_only(text):
    """
    Emitting the required format is an ability the benchmark measures.
    A model told to write only the number, which writes anything else,
    has not answered - and invalid_output_rate is where that shows.
    """

    parsed = parse_prediction(CHOICE, text)

    assert not parsed.valid
    assert parsed.error == "not_one_of_allowed_choices"


@pytest.mark.parametrize("text", ["2", " 2 ", "`2`", "**2**"])
def test_a_numbered_choice_accepts_the_bare_number(text):
    parsed = parse_prediction(CHOICE, text)

    assert parsed.valid
    assert parsed.value == "2"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0.0),
        ("2", 2.0),
        ("2.4", 2.4),
        ("2,4", 2.4),
        ("5.0", 5.0),
        (" 3 ", 3.0),
    ],
)
def test_numeric_accepts_plain_numbers(text, expected):
    parsed = parse_prediction(STS, text)

    assert parsed.valid
    assert parsed.value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("Score: 4", "not_a_number"),
        ("4 because they match", "not_a_number"),
        ("təxminən 4", "not_a_number"),
        ("", "empty_output"),
        ("7", "out_of_range"),
        ("-1", "out_of_range"),
    ],
)
def test_numeric_rejects_everything_else(text, error):
    parsed = parse_prediction(STS, text)

    assert not parsed.valid
    assert parsed.error == error


def test_text_keeps_the_answer_as_written():
    parsed = parse_prediction(QA, "  5 dəfə  ")

    assert parsed.valid
    assert parsed.value == "5 dəfə"


def test_empty_text_is_invalid():
    parsed = parse_prediction(QA, "")

    assert not parsed.valid
    assert parsed.error == "empty_output"


def test_abstain_sentinel_is_detected():
    parsed = parse_prediction(
        ABSTENTION,
        "Cavab yoxdur",
    )

    assert parsed.valid
    assert parsed.abstained
    assert parsed.value is None


def test_abstain_detection_is_case_insensitive():
    parsed = parse_prediction(
        ABSTENTION,
        "cavab yoxdur",
    )

    assert parsed.abstained


def test_answer_is_not_confused_with_abstention():
    parsed = parse_prediction(
        ABSTENTION,
        "2004-cü ildə",
    )

    assert parsed.valid
    assert not parsed.abstained
    assert parsed.value == "2004-cü ildə"
