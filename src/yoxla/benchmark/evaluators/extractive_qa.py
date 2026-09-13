"""
Comparing an answer with a gold span.

Used by the span evaluators. Gold spans are copied verbatim out of
the context, so their boundaries carry the case endings of the
sentence they came from: the gold answer is "Çin və Hindistandan"
where the answer itself is "Çin və Hindistan", and "Nyu-York şəhəri"
where the context said "Nyu-York şəhərində". Exact match would score
those as failures, which measures the annotation convention of an
agglutinative language rather than reading comprehension, so
``token_f1`` carries the score and ``normalized_em`` stays a
diagnostic. SQuAD and XQuAD report F1 as the headline metric for the
same reason.

Both sides are treated identically, and the one thing never forgiven
is a diacritic: "Gence" for "Gəncə" is a misspelling, not a spelling
variant.
"""

from yoxla.benchmark.parsing import strip_wrappers
from yoxla.text import (
    containment,
    normalize_answer,
    token_f1,
)

TRAILING_PUNCTUATION = ".!?;:,"


def comparable(text: str) -> str:
    """
    Normalization used when matching an answer against a gold span.

    Both sides are treated identically. Quotation marks are removed
    because some gold spans were copied out of the source together
    with them ("Röyasını görmüşdüm"), and closing punctuation is
    removed because a final period is a formatting difference, not a
    different answer - ``strict_em`` is where that still counts.
    """

    stripped = strip_wrappers(text).rstrip(
        TRAILING_PUNCTUATION
    )

    return normalize_answer(strip_wrappers(stripped))


def score_answer(
    prediction: str,
    gold: str,
) -> dict[str, float]:
    """
    Compare a cleaned prediction with a gold answer.
    """

    clean_prediction = comparable(prediction)
    clean_gold = comparable(gold)

    return {
        "normalized_em": float(
            clean_prediction == clean_gold
        ),
        "containment": containment(
            clean_prediction,
            clean_gold,
        ),
        "token_f1": token_f1(prediction, gold),
    }
