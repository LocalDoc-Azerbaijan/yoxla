"""
Prediction parsing.

The parser is deliberately strict. Extracting a label out of a
sentence ("The answer is neutral") would silently reward models that
ignore the instructions, so anything that is not exactly the required
answer format is recorded as an invalid output instead.

Only symmetric wrappers are removed:

    `neutral`      -> neutral
    "neutral"      -> neutral
    **neutral**    -> neutral

Nothing else is repaired, because emitting the required format is
an ability worth measuring rather than an obstacle to measuring:

    Neutral.               -> invalid
    17. 1873               -> invalid, even when option 17 is 1873
    The answer is neutral  -> invalid

Case is the one surface difference that is forgiven, because it does
not change which label was chosen; anything the answer space does not
contain does.
"""

import re
from dataclasses import dataclass
from typing import Any

from yoxla.benchmark.schema import AnswerMode, BenchmarkTask
from yoxla.text import az_lower, normalize_text

_NUMBER = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

_WRAPPERS = (
    ("```", "```"),
    ("**", "**"),
    ("`", "`"),
    ('"', '"'),
    ("'", "'"),
    ("«", "»"),
    ("“", "”"),
)


@dataclass(frozen=True)
class ParsedPrediction:
    raw: str

    value: Any
    valid: bool

    error: str | None = None

    abstained: bool = False


def strip_wrappers(text: str) -> str:
    """
    Remove symmetric quoting/formatting wrappers.
    """

    cleaned = text.strip()

    changed = True

    while changed:
        changed = False

        for opening, closing in _WRAPPERS:
            if (
                len(cleaned) > len(opening) + len(closing)
                and cleaned.startswith(opening)
                and cleaned.endswith(closing)
            ):
                cleaned = cleaned[
                    len(opening) : -len(closing)
                ].strip()

                changed = True

    return cleaned


def _clean(text: str) -> str:
    return normalize_text(strip_wrappers(text))


def parse_prediction(
    task: BenchmarkTask,
    text: str | None,
) -> ParsedPrediction:
    raw = text or ""

    mode = task.answer.mode

    if mode is AnswerMode.CHOICE:
        return _parse_choice(task, raw)

    if mode is AnswerMode.NUMERIC:
        return _parse_numeric(task, raw)

    if mode is AnswerMode.TEXT_OR_ABSTAIN:
        return _parse_text_or_abstain(task, raw)

    return _parse_text(raw)


def _parse_text(raw: str) -> ParsedPrediction:
    cleaned = _clean(raw)

    if not cleaned:
        return ParsedPrediction(
            raw=raw,
            value=None,
            valid=False,
            error="empty_output",
        )

    return ParsedPrediction(
        raw=raw,
        value=cleaned,
        valid=True,
    )


def _parse_text_or_abstain(
    task: BenchmarkTask,
    raw: str,
) -> ParsedPrediction:
    parsed = _parse_text(raw)

    if not parsed.valid:
        return parsed

    abstain_value = task.answer.abstain_value or ""

    if az_lower(parsed.value) == az_lower(
        normalize_text(abstain_value)
    ):
        return ParsedPrediction(
            raw=raw,
            value=None,
            valid=True,
            abstained=True,
        )

    return parsed


def _parse_choice(
    task: BenchmarkTask,
    raw: str,
) -> ParsedPrediction:
    cleaned = _clean(raw)

    if not cleaned:
        return ParsedPrediction(
            raw=raw,
            value=None,
            valid=False,
            error="empty_output",
        )

    lowered = az_lower(cleaned)

    for choice in task.answer.choices:
        if lowered == az_lower(choice.value):
            return ParsedPrediction(
                raw=raw,
                value=choice.value,
                valid=True,
            )

    return ParsedPrediction(
        raw=raw,
        value=None,
        valid=False,
        error="not_one_of_allowed_choices",
    )


def _parse_numeric(
    task: BenchmarkTask,
    raw: str,
) -> ParsedPrediction:
    cleaned = _clean(raw)

    if not cleaned:
        return ParsedPrediction(
            raw=raw,
            value=None,
            valid=False,
            error="empty_output",
        )

    if not _NUMBER.match(cleaned):
        return ParsedPrediction(
            raw=raw,
            value=None,
            valid=False,
            error="not_a_number",
        )

    value = float(cleaned.replace(",", "."))

    minimum = task.answer.minimum
    maximum = task.answer.maximum

    if (
        minimum is not None
        and value < minimum
    ) or (
        maximum is not None
        and value > maximum
    ):
        return ParsedPrediction(
            raw=raw,
            value=None,
            valid=False,
            error="out_of_range",
        )

    return ParsedPrediction(
        raw=raw,
        value=value,
        valid=True,
    )
