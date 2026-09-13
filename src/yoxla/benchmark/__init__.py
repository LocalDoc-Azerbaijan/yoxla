"""
YOXLA benchmark layer.

Everything that knows what a benchmark task means lives here:
datasets, prompts, answer spaces, parsing and evaluation. The
inference layer stays benchmark-agnostic and never imports from this
package.
"""

from yoxla.benchmark.evaluators import create_evaluator
from yoxla.benchmark.loaders import load_examples, load_rows
from yoxla.benchmark.parsing import ParsedPrediction, parse_prediction
from yoxla.benchmark.prompting import render_messages, render_prompt
from yoxla.benchmark.registry import (
    BENCHMARK_VERSIONS,
    TASKS,
    get_block_tasks,
    get_task,
    list_blocks,
    resolve_tasks,
)
from yoxla.benchmark.schema import (
    AnswerMode,
    AnswerSpec,
    BenchmarkExample,
    BenchmarkTask,
    ChoiceSpec,
    GenerationSpec,
    PromptSpec,
    RubricLevel,
)
from yoxla.benchmark.validation import validate_task, validate_tasks

__all__ = [
    "BENCHMARK_VERSIONS",
    "TASKS",
    "AnswerMode",
    "AnswerSpec",
    "BenchmarkExample",
    "BenchmarkTask",
    "ChoiceSpec",
    "GenerationSpec",
    "ParsedPrediction",
    "PromptSpec",
    "RubricLevel",
    "create_evaluator",
    "get_block_tasks",
    "get_task",
    "list_blocks",
    "load_examples",
    "load_rows",
    "parse_prediction",
    "render_messages",
    "render_prompt",
    "resolve_tasks",
    "validate_task",
    "validate_tasks",
]
