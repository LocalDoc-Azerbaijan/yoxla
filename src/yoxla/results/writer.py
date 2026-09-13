"""
Run artifacts.

    runs/<run_id>/
        run.json            configuration and provenance
        predictions.jsonl   one line per example, flushed immediately
        task_scores.json    metrics and normalized score per task
        summary.json        block and benchmark aggregates

``predictions.jsonl`` is appended and flushed after every example so
that a long API run can be resumed instead of repeated.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from yoxla.results.schema import PredictionRecord, RunInfo


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_run_id(model: str, provider: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", model).strip("-")

    stamp = datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return f"{provider}_{slug}_{stamp}"


class ResultWriter:
    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
    ):
        self.run_id = run_id

        self.directory = Path(output_dir) / run_id

        self.directory.mkdir(parents=True, exist_ok=True)

        self.predictions_path = (
            self.directory / "predictions.jsonl"
        )

        self._file = None

    # --------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------

    def load_completed(
        self,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Read already finished examples of this run directory.

        Malformed trailing lines - a run killed mid-write - are
        skipped rather than failing the resume.
        """

        completed: dict[tuple[str, str], dict[str, Any]] = {}

        if not self.predictions_path.exists():
            return completed

        with self.predictions_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                key = (
                    record.get("task_id", ""),
                    record.get("example_id", ""),
                )

                completed[key] = record

        return completed

    def open_predictions(self, resume: bool) -> None:
        mode = "a" if resume else "w"

        self._file = self.predictions_path.open(
            mode,
            encoding="utf-8",
        )

    def write_prediction(
        self,
        record: PredictionRecord,
    ) -> None:
        if self._file is None:
            self.open_predictions(resume=True)

        if not record.timestamp:
            record.timestamp = utc_now()

        self._file.write(
            json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                # Provider payloads stored for unparsable answers may
                # contain types json does not know.
                default=str,
            )
            + "\n"
        )

        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # --------------------------------------------------------------
    # Run metadata and scores
    # --------------------------------------------------------------

    def write_run_info(self, run_info: RunInfo) -> None:
        self._write_json("run.json", run_info.to_dict())

    def write_task_scores(
        self,
        task_scores: dict[str, Any],
    ) -> None:
        self._write_json("task_scores.json", task_scores)

    def write_summary(
        self,
        summary: dict[str, Any],
    ) -> None:
        self._write_json("summary.json", summary)

    def _write_json(
        self,
        name: str,
        payload: Any,
    ) -> None:
        path = self.directory / name

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
