"""
Frozen benchmark manifests.

A manifest records the row count and content hash of every task of a
block at freeze time. It ships inside the package, so YOXLA can warn
when the data behind a published leaderboard run has changed on the
Hub.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

MANIFEST_PACKAGE = "yoxla.benchmark.manifests"


def manifest_name(block: str) -> str:
    from yoxla.benchmark.registry import BENCHMARK_VERSIONS

    version = BENCHMARK_VERSIONS.get(block, block)

    return f"{version}.json"


def load_manifest(block: str) -> dict[str, Any] | None:
    """
    Return the bundled manifest of a block, or None when the block has
    not been frozen yet.
    """

    resource = files(MANIFEST_PACKAGE).joinpath(
        manifest_name(block)
    )

    if not resource.is_file():
        return None

    with resource.open("r", encoding="utf-8") as file:
        return json.load(file)


def manifest_path(block: str) -> Path:
    """
    Filesystem path used when writing a manifest during development.
    """

    return (
        Path(__file__).parent
        / "manifests"
        / manifest_name(block)
    )


def write_manifest(
    block: str,
    tasks: list[dict[str, Any]],
    revision: str | None = None,
) -> Path:
    from yoxla.benchmark.registry import BENCHMARK_VERSIONS

    payload = {
        "block": block,
        "benchmark_version": BENCHMARK_VERSIONS.get(
            block, block
        ),
        "dataset_revision": revision,
        "tasks": tasks,
    }

    path = manifest_path(block)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")

    return path
