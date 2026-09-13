import json
from dataclasses import replace

import pytest

from yoxla.benchmark.loaders import row_to_example
from yoxla.benchmark.registry import TASKS
from yoxla.inference.base import ModelAdapter
from yoxla.inference.schemas import GenerationResponse, TokenUsage
from yoxla.runner import (
    BenchmarkValidationError,
    is_retryable,
    rescore_run,
    run_benchmark,
)

# The registry task expects 100 rows; the fixture below serves 3.
NLI = replace(TASKS["nli_v1"], expected_examples=3)

ROWS = [
    {
        "id": "understanding_nli_0001",
        "premise": "Yorğun bir kişi yarışı bitirir.",
        "hypothesis": "Kişi sonuncu yerdədir.",
        "label": "neutral",
        "length_bucket": "short",
        "source_id": "a",
    },
    {
        "id": "understanding_nli_0002",
        "premise": "Bir kişi binanın yanından keçir.",
        "hypothesis": "Bir kişi hərəkət edir.",
        "label": "entailment",
        "length_bucket": "short",
        "source_id": "b",
    },
    {
        "id": "understanding_nli_0003",
        "premise": "Bir motokros sürücüsü təpədədir.",
        "hypothesis": "Kişi eyvanda yatır.",
        "label": "contradiction",
        "length_bucket": "short",
        "source_id": "c",
    },
]


class ScriptedModel(ModelAdapter):
    """
    Returns a prepared answer per example, in order.
    """

    def __init__(self, answers):
        super().__init__(
            model="scripted-model",
            provider="test",
        )

        self.answers = list(answers)

        self.calls = 0

    def generate(self, request):
        answer = self.answers[self.calls]

        self.calls += 1

        return GenerationResponse(
            text=answer,
            raw_text=answer,
            model=self.model,
            provider=self.provider,
            finish_reason="stop",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=2,
            ),
            latency_ms=1.0,
        )


class ExplodingModel(ModelAdapter):
    def __init__(self):
        super().__init__(
            model="scripted-model",
            provider="test",
        )

    def generate(self, request):  # pragma: no cover
        raise AssertionError(
            "the model must not be called for cached examples"
        )


@pytest.fixture
def local_rows(monkeypatch):
    def fake_load_rows(task, **kwargs):
        assert task.task_id == "nli_v1"

        return [dict(row) for row in ROWS]

    monkeypatch.setattr(
        "yoxla.runner.load_rows",
        fake_load_rows,
    )


def read_predictions(directory):
    lines = (
        (directory / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )

    return [json.loads(line) for line in lines]


def test_run_writes_all_artifacts(local_rows, tmp_path):
    model = ScriptedModel(
        [
            "neutral",
            "entailment",
            "The answer is contradiction",
        ]
    )

    summary = run_benchmark(
        model=model,
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="test-run",
    )

    directory = tmp_path / "test-run"

    for name in (
        "run.json",
        "predictions.jsonl",
        "task_scores.json",
        "summary.json",
    ):
        assert (directory / name).exists()

    assert model.calls == 3

    # Two correct answers, one that ignored the output format.
    assert summary["task_scores"]["nli_v1"] == pytest.approx(
        66.6667,
        abs=0.001,
    )

    assert summary["blocks"]["understanding"] == (
        summary["score"]
    )

    predictions = read_predictions(directory)

    assert len(predictions) == 3

    invalid = predictions[2]

    assert invalid["prediction_valid"] is False
    assert invalid["parse_error"] == (
        "not_one_of_allowed_choices"
    )

    # The raw answer is always kept for later analysis.
    assert invalid["raw_prediction"] == (
        "The answer is contradiction"
    )

    run_info = json.loads(
        (directory / "run.json").read_text(
            encoding="utf-8"
        )
    )

    task_info = run_info["tasks"][0]

    assert task_info["prompt_version"] == 1
    assert task_info["system_prompt_hash"]
    assert task_info["dataset_fingerprint"].startswith(
        "sha256:"
    )
    assert run_info["benchmark_versions"] == {
        "understanding": "understanding_v1"
    }


def test_resume_reuses_stored_predictions(
    local_rows,
    tmp_path,
):
    first = run_benchmark(
        model=ScriptedModel(
            ["neutral", "entailment", "contradiction"]
        ),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="resumed",
    )

    second = run_benchmark(
        model=ExplodingModel(),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="resumed",
        resume=True,
    )

    assert second["task_scores"] == first["task_scores"]
    assert second["score"] == first["score"]

    assert (
        len(
            read_predictions(tmp_path / "resumed")
        )
        == 3
    )


def test_resume_reattempts_examples_lost_to_provider_errors(
    local_rows,
    tmp_path,
):
    class FlakyModel(ModelAdapter):
        """
        Fails the second example with a transport error.
        """

        def __init__(self, answers, fail_at):
            super().__init__(
                model="flaky",
                provider="test",
            )

            self.answers = list(answers)
            self.fail_at = fail_at
            self.calls = 0

        def generate(self, request):
            answer = self.answers[self.calls]

            self.calls += 1

            if self.calls == self.fail_at:
                raise ConnectionError("upstream gone")

            return GenerationResponse(
                text=answer,
                raw_text=answer,
                model=self.model,
                provider=self.provider,
                finish_reason="stop",
                usage=TokenUsage(),
            )

    answers = ["neutral", "entailment", "contradiction"]

    first = run_benchmark(
        model=FlakyModel(answers, fail_at=2),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="flaky",
        max_retries=0,
    )

    assert first["generation_errors"] == 1
    assert first["task_scores"]["nli_v1"] == pytest.approx(
        66.6667,
        abs=0.001,
    )

    # The lost example never received an answer, so resuming has to
    # ask for it again rather than keep the zero. Only the second
    # example is expected to be requested.
    model = FlakyModel(["entailment"], fail_at=None)

    second = run_benchmark(
        model=model,
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="flaky",
        resume=True,
    )

    assert model.calls == 1
    assert second["generation_errors"] == 0
    assert second["task_scores"]["nli_v1"] == 100.0

    records = read_predictions(tmp_path / "flaky")

    # No stale duplicate left behind for the retried example.
    assert len(records) == 3
    assert all(
        record["error"] is None for record in records
    )


def test_generation_errors_are_recorded_not_retried(
    local_rows,
    tmp_path,
):
    class FailingModel(ModelAdapter):
        def __init__(self):
            super().__init__(
                model="failing",
                provider="test",
            )

            self.calls = 0

        def generate(self, request):
            self.calls += 1

            raise ValueError("bad request")

    model = FailingModel()

    summary = run_benchmark(
        model=model,
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="failing",
        max_retries=2,
    )

    # A non-transport error is not retried.
    assert model.calls == 3

    assert summary["generation_errors"] == 3
    assert summary["task_scores"]["nli_v1"] == 0.0

    predictions = read_predictions(tmp_path / "failing")

    assert all(
        record["error"] for record in predictions
    )


def test_validation_failure_stops_before_inference(
    monkeypatch,
    tmp_path,
):
    def broken_rows(task, **kwargs):
        return [dict(ROWS[0])]

    monkeypatch.setattr(
        "yoxla.runner.load_rows",
        broken_rows,
    )

    with pytest.raises(BenchmarkValidationError):
        run_benchmark(
            model=ExplodingModel(),
            tasks=[NLI],
            output_dir=str(tmp_path),
            run_id="invalid",
        )


def test_provider_payload_is_stored_for_unusable_answers(
    local_rows,
    tmp_path,
):
    class PayloadModel(ScriptedModel):
        def generate(self, request):
            response = super().generate(request)

            response.raw_response = {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning": "uzun düşüncə...",
                        }
                    }
                ]
            }

            return response

    run_benchmark(
        model=PayloadModel(["neutral", "", "entailment"]),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="payloads",
    )

    records = read_predictions(tmp_path / "payloads")

    # Only the answer that failed to parse carries the payload.
    assert records[0]["provider_response"] is None
    assert records[2]["provider_response"] is None

    payload = records[1]["provider_response"]

    assert payload is not None
    assert (
        "uzun düşüncə..."
        in payload["choices"][0]["message"]["reasoning"]
    )


def test_limited_run_is_marked_as_partial(
    local_rows,
    tmp_path,
):
    summary = run_benchmark(
        model=ScriptedModel(["neutral"]),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="partial",
        limit=1,
    )

    assert summary["partial"] is True
    assert summary["limit"] == 1
    assert summary["examples"] == 1

    full = run_benchmark(
        model=ScriptedModel(
            ["neutral", "entailment", "contradiction"]
        ),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="full",
    )

    assert full["partial"] is False
    assert full["limit"] is None


def test_rescore_refuses_answers_to_a_different_question(
    local_rows,
    monkeypatch,
    tmp_path,
):
    """
    Rescoring replays stored answers through new evaluators. It must
    not replay them through a new prompt: the model would be scored
    on a question it was never asked.
    """

    monkeypatch.setattr(
        "yoxla.runner.load_examples",
        lambda task, **kwargs: [
            row_to_example(task, dict(row)) for row in ROWS
        ],
    )

    run_benchmark(
        model=ScriptedModel(
            ["neutral", "entailment", "contradiction"]
        ),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="stale-prompt",
    )

    directory = tmp_path / "stale-prompt"

    # Stand in for the prompt having been reworded since the run.
    run_file = directory / "run.json"

    info = json.loads(run_file.read_text(encoding="utf-8"))

    info["tasks"][0]["system_prompt_hash"] = "0" * 64

    run_file.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as raised:
        rescore_run(directory)

    assert "different question" in str(raised.value)


def test_rescore_recomputes_from_stored_answers(
    local_rows,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "yoxla.runner.load_examples",
        lambda task, **kwargs: [
            row_to_example(task, dict(row)) for row in ROWS
        ],
    )

    original = run_benchmark(
        model=ScriptedModel(
            [
                "neutral",
                "entailment",
                "The answer is contradiction",
            ]
        ),
        tasks=[NLI],
        output_dir=str(tmp_path),
        run_id="rescored",
    )

    directory = tmp_path / "rescored"

    before = read_predictions(directory)

    # No model call: the stored raw answers are enough.
    summary = rescore_run(directory)

    assert summary["task_scores"] == original["task_scores"]
    assert summary["rescored_at"]

    after = read_predictions(directory)

    assert len(after) == len(before)

    assert [r["raw_prediction"] for r in after] == [
        r["raw_prediction"] for r in before
    ]

    assert not (
        directory / "predictions.jsonl.tmp"
    ).exists()


def test_only_transport_failures_are_retryable():
    class Rate(Exception):
        status_code = 429

    class Bad(Exception):
        status_code = 400

    assert is_retryable(Rate())
    assert is_retryable(TimeoutError())
    assert is_retryable(ConnectionError())

    assert not is_retryable(Bad())
    assert not is_retryable(ValueError("bad output"))
