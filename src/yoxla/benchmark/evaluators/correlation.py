"""
Correlation helpers.

Implemented locally so that the benchmark does not pull in SciPy for
two functions.
"""

import math


def pearson(
    xs: list[float],
    ys: list[float],
) -> float | None:
    """
    Pearson correlation. Returns None when it is undefined.
    """

    if len(xs) != len(ys) or len(xs) < 2:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    covariance = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(xs, ys, strict=True)
    )

    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)

    if variance_x <= 0.0 or variance_y <= 0.0:
        # A constant series has no correlation to report.
        return None

    return covariance / math.sqrt(
        variance_x * variance_y
    )


def rank(values: list[float]) -> list[float]:
    """
    Fractional ranks with average ranks for ties.
    """

    order = sorted(
        range(len(values)),
        key=lambda index: values[index],
    )

    ranks = [0.0] * len(values)

    position = 0

    while position < len(order):
        end = position

        while (
            end + 1 < len(order)
            and values[order[end + 1]]
            == values[order[position]]
        ):
            end += 1

        average_rank = (position + end) / 2 + 1

        for index in range(position, end + 1):
            ranks[order[index]] = average_rank

        position = end + 1

    return ranks


def spearman(
    xs: list[float],
    ys: list[float],
) -> float | None:
    """
    Spearman rank correlation.
    """

    if len(xs) != len(ys) or len(xs) < 2:
        return None

    return pearson(rank(xs), rank(ys))
