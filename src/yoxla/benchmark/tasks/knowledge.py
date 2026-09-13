"""
Knowledge v1 - 150 frozen examples.

    knowledge_choice_v1     150

Facts about Azerbaijan, harvested from Wikidata and spread across
seven categories so that no single corner of the country's
geography, history or culture carries the block.

The answer space is closed: twenty candidates per question, and the
model returns a number. An open-answer version of the same facts was
built first and dropped, because sixty-two of the hundred and fifty
answers are multi-word entities - honorific titles no two people
would word alike, names with up to seven equally correct forms - and
every scoring dispute found in review came from them. Scoring those
needs either a judge or a list of accepted spellings that someone has
to keep guessing at, and neither belongs in a frozen benchmark.

Closing the space costs the distinction between recall and
recognition. What it buys is that a wrong answer is wrong because the
model chose wrong, never because a check disagreed about spelling.

The distractors have to be hard for that to be worth anything, so
they are drawn from the same probe, bracketed by notability, and the
finished set is run past cue-only adversaries before it ships.
"""

from yoxla.benchmark.schema import (
    AnswerMode,
    AnswerSpec,
    BenchmarkTask,
    ChoiceSpec,
    GenerationSpec,
    PromptSpec,
)

BLOCK = "knowledge"

DATASET_REPO = "LocalDoc/YOXLA-Benchmark"

LANGUAGE = "az"

# Twenty candidates per question. The number is part of the contract:
# it fixes chance at 5% and it is what the build-time adversaries are
# measured against.
OPTION_COUNT = 20


KNOWLEDGE_CHOICE_V1 = BenchmarkTask(
    task_id="knowledge_choice_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    # Named for the task, not for the block. The config was called
    # knowledge_v1 while an open-answer task shared this file; that
    # task is gone, and a directory named after it is a trap for the
    # next reader. Renamed while nothing was published under the old
    # name, so the change cost nothing.
    dataset_config="knowledge_choice_v1",
    dataset_split="test",
    expected_examples=150,
    input_fields=(
        "question",
        "candidates",
    ),
    gold_field="answer_index",
    metadata_fields=(
        "answer",
        "category",
        "probe",
        "value_kind",
        "subject",
        "subject_qid",
        "sitelinks",
        "source_id",
    ),
    breakdown_field="category",
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə Azərbaycanla bağlı bir sual və nömrələnmiş "
            "cavab variantları veriləcək.\n"
            "\n"
            "Yalnız düzgün variantın nömrəsini yaz.\n"
            "\n"
            "Heç bir izah, şərh və ya əlavə mətn yazma. "
            "Cavabı bilmirsənsə, ən ehtimallı nömrəni yaz."
        ),
        user=(
            "{question}\n"
            "\n"
            "{candidates}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=tuple(
            ChoiceSpec(
                value=str(number),
                description=(
                    f"siyahıdakı {number}-ci variant"
                ),
            )
            for number in range(1, OPTION_COUNT + 1)
        ),
    ),
    generation=GenerationSpec(
        max_output_tokens=8,
    ),
    evaluator="classification_exact",
    primary_metric="accuracy",
    diagnostic_metrics=(
        "modal_answer_share",
        "per_category_accuracy",
        "per_class_accuracy",
        "invalid_output_rate",
    ),
)


KNOWLEDGE_TASKS: tuple[BenchmarkTask, ...] = (
    KNOWLEDGE_CHOICE_V1,
)

RETIRED_KNOWLEDGE_TASKS: tuple[BenchmarkTask, ...] = ()
