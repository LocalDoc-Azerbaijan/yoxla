"""
Span scoring against character offsets.

An extractive answer lives inside a passage the model was given, so
its correctness is a question about positions rather than about
strings. Locating the quoted text in the passage and comparing
character ranges removes every judgement call the string comparison
needed: no accepted forms, no normalizer, nothing to maintain when a
new way of writing the same span turns up.

It also separates two abilities the single score was conflating.
Measured over five models, span recall sits between 0.83 and 0.85 for
the top three - they find the answer equally well - while precision
spreads from 0.48 to 0.92. The model that ranked last on ``token_f1``
sits fourth on recall: most of its deficit was the length of its
quotes rather than its reading. That is a formatting difference being
scored as a comprehension failure, which is the problem this
evaluator exists to stop.

So ``span_recall`` is the primary metric - did the model find the
answer - and ``span_precision`` is reported beside it as a first-class
number rather than folded in.

Recall is credited only when the quote stays inside the sentences the
gold span occupies. Without that, quoting the whole passage would
score a perfect recall.

The guard is not free: a model that habitually returns two sentences
forfeits recall for those examples, which puts some of its verbosity
back into the recall number. Measured over five models that costs
between 2 and 10 per cent of examples, and it is the price of having
a headline metric that cannot be farmed by copying the input.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_WHITESPACE = re.compile(r"\s+")

_SENTENCE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")

# Quotation marks and closing punctuation a model puts around a span
# it copied. Stripping them is not repair: the span underneath is
# still judged on its own boundaries.
_TRIM = "\"'`«»“”.,;:!?()[]"


def flatten(text: str) -> str:
    """
    NFC plus whitespace collapse, case preserved.

    Both the passage and the quote go through this, so a line break
    inside the passage cannot make a correct quote unlocatable.
    """

    return _WHITESPACE.sub(
        " ", unicodedata.normalize("NFC", text)
    ).strip()


@dataclass(frozen=True)
class SpanMatch:
    found: bool

    recall: float = 0.0
    precision: float = 0.0

    exact: bool = False
    outside_sentence: bool = False


def sentence_window(context: str, start: int, end: int) -> tuple[int, int]:
    """
    The sentences the gold span occupies, start to end.

    A gold span that crosses a sentence boundary widens the window
    rather than making the example unwinnable.
    """

    low, high = None, None

    for match in _SENTENCE.finditer(context):
        if match.end() <= start or match.start() >= end:
            continue

        low = match.start() if low is None else min(low, match.start())
        high = match.end() if high is None else max(high, match.end())

    if low is None:
        return 0, len(context)

    return low, high


def locate_gold(context: str, gold: str) -> tuple[int, int] | None:
    """
    Where the gold answer sits in the passage.

    Every gold span in these tasks occurs exactly once in its own
    context, so the search is unambiguous - ``validate_gold`` is what
    keeps that true.
    """

    flat_context = flatten(context)
    flat_gold = flatten(gold)

    if not flat_gold:
        return None

    at = flat_context.find(flat_gold)

    if at < 0:
        return None

    return at, at + len(flat_gold)


def score_span(
    context: str,
    gold: str,
    prediction: str,
) -> SpanMatch:
    flat_context = flatten(context)

    span = locate_gold(context, gold)

    if span is None:
        return SpanMatch(found=False)

    gold_start, gold_end = span

    quote = flatten(prediction).strip(_TRIM).strip()

    if not quote:
        return SpanMatch(found=False)

    # Case-insensitive, because a model that lowercases a copied word
    # has not invented anything - that is a formatting slip, and
    # strict_em is where it counts.
    at = flat_context.lower().find(quote.lower())

    if at < 0:
        return SpanMatch(found=False)

    start, end = at, at + len(quote)

    window_start, window_end = sentence_window(
        flat_context, gold_start, gold_end
    )

    outside = start < window_start or end > window_end

    overlap = max(
        0, min(end, gold_end) - max(start, gold_start)
    )

    return SpanMatch(
        found=True,
        # Quoting past the answer's own sentence earns no recall; it
        # is the one way an unlimited quote could buy a perfect score.
        recall=(
            0.0
            if outside
            else overlap / (gold_end - gold_start)
        ),
        precision=overlap / (end - start),
        exact=(start, end) == (gold_start, gold_end),
        outside_sentence=outside,
    )


def check_gold_is_locatable(
    example: Any,
    gold: str,
) -> list[str]:
    """
    The gold must appear in its own passage exactly once.

    Offsets mean nothing if the answer is absent, and they are
    ambiguous if it repeats - so both are refused before a run rather
    than silently mis-scored during one.
    """

    context = example.inputs.get("context")

    if not context:
        return [f"{example.id}: no context to locate the answer in"]

    flat_context = flatten(context)
    flat_gold = flatten(gold)

    if not flat_gold:
        return [f"{example.id}: empty gold answer"]

    occurrences = flat_context.count(flat_gold)

    if occurrences == 0:
        return [
            f"{example.id}: the gold answer is not in its context"
        ]

    if occurrences > 1:
        return [
            f"{example.id}: the gold answer occurs {occurrences} "
            f"times in its context, so its position is ambiguous"
        ]

    # answer_start is not needed for scoring - the answer is unique -
    # but where the dataset ships it, disagreeing with it means one of
    # the two is wrong and the run should not start.
    declared = example.metadata.get("answer_start")

    if isinstance(declared, int) and declared >= 0:
        raw = unicodedata.normalize("NFC", context)
        quoted = raw[declared : declared + len(flatten(gold))]

        if flatten(quoted) != flat_gold:
            return [
                f"{example.id}: answer_start does not point at the "
                f"gold answer"
            ]

    return []
