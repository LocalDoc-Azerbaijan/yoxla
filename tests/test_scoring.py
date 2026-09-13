import pytest

from yoxla.benchmark.registry import TASKS
from yoxla.scoring import (
    compute_benchmark_score,
    compute_block_score,
    compute_task_score,
    normalize_metric,
)


def test_normalize_metric_clamps_to_0_100():
    assert normalize_metric(0.5) == pytest.approx(50.0)
    assert normalize_metric(-0.3) == pytest.approx(0.0)
    assert normalize_metric(1.4) == pytest.approx(100.0)
    assert normalize_metric(None) == pytest.approx(0.0)


def test_task_score_uses_the_declared_primary_metric():
    task = TASKS["nli_v1"]

    score = compute_task_score(
        task,
        {
            "count": 100,
            "accuracy": 0.42,
            "macro_f1": 0.9,
        },
    )

    assert score.primary_metric == "accuracy"
    assert score.score == pytest.approx(42.0)
    assert score.count == 100


def test_negative_correlation_scores_zero():
    task = TASKS["sts_v1"]

    score = compute_task_score(
        task,
        {"count": 100, "spearman": -0.4},
    )

    assert score.score == pytest.approx(0.0)


def test_block_score_is_a_macro_average_over_tasks():
    scores = [
        compute_task_score(
            TASKS["extractive_qa_v1"],
            {"count": 200, "span_recall": 1.0},
        ),
        compute_task_score(
            TASKS["intent_v1"],
            {"count": 50, "accuracy": 0.0},
        ),
    ]

    block = compute_block_score("understanding", scores)

    # 200 QA examples must not outweigh 50 intent examples.
    assert block.score == pytest.approx(50.0)

    benchmark = compute_benchmark_score([block])

    assert benchmark.score == pytest.approx(50.0)
