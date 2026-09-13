"""
Text normalization utilities shared by parsers and evaluators.

Normalization is intentionally conservative. Azerbaijani-specific
characters are never folded into their ASCII counterparts, because
"ə -> e" or "ş -> s" would make the benchmark reward models that do
not actually write correct Azerbaijani.
"""

import re
import unicodedata
from collections import Counter

_WHITESPACE = re.compile(r"\s+")

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Azerbaijani has a dotted/dotless "i" pair.
# Python's default lowercase maps "I" -> "i" and "İ" -> "i̇",
# both of which are wrong for Azerbaijani.
_AZ_LOWER_MAP = str.maketrans(
    {
        "I": "ı",
        "İ": "i",
    }
)


def az_lower(text: str) -> str:
    """
    Lowercase text using Azerbaijani casing rules.
    """

    return text.translate(_AZ_LOWER_MAP).lower()


def normalize_text(text: str) -> str:
    """
    Unicode NFC + whitespace collapse + trim.

    Case is preserved.
    """

    normalized = unicodedata.normalize("NFC", text)

    normalized = _WHITESPACE.sub(" ", normalized)

    return normalized.strip()


def normalize_answer(text: str) -> str:
    """
    Normalization used for answer comparison.

    Unicode NFC
    trim
    collapse whitespace
    AZ-aware lowercase
    """

    return az_lower(normalize_text(text))


def tokenize(text: str) -> list[str]:
    """
    Split normalized text into word tokens.
    """

    return _TOKEN.findall(normalize_answer(text))


def token_f1(
    prediction: str,
    gold: str,
) -> float:
    """
    Token-level F1 between a prediction and a gold answer.
    """

    prediction_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)

    common = (
        Counter(prediction_tokens)
        & Counter(gold_tokens)
    )

    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)

    return (
        2 * precision * recall / (precision + recall)
    )


def containment(
    prediction: str,
    gold: str,
) -> float:
    """
    Whether the normalized gold answer appears inside the
    normalized prediction.
    """

    normalized_gold = normalize_answer(gold)

    if not normalized_gold:
        return 0.0

    return float(
        normalized_gold in normalize_answer(prediction)
    )
