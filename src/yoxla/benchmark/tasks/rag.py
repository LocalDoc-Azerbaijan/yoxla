"""
RAG v1 - 293 frozen examples.

    rag_verification_v1     160
    rag_selection_v1        133

The two halves of a retrieval system, measured apart.
`rag_selection_v1` asks whether the right passage was picked up;
`rag_verification_v1` asks what the model does with a passage once it
has one. A system can fail at either, and one number would hide which.

A passage and one claim about it. The model says whether the passage
supports the claim, contradicts it, or does not mention it at all.

Two numbers come out of it, and they are the two questions a RAG
system has to answer for:

    per_claim_type_accuracy["supported"]   did it find what is there
    per_claim_type_accuracy["absent"]      did it accept what is not

The third label is what makes the second one measurable. Asked only
"true or false", a model has nowhere to put a claim the passage is
silent about, and inventing support for it looks the same as reading
it correctly.

Absence is decidable here in a way it is not for open knowledge,
because the passage is built from a known list of facts. Anything
outside that list is provably not in it - a claim about a death year
in a passage that states only a birthplace is unsupported by
construction, not by judgement.

The fourth claim type is the one worth the most. The passage states an
altered value - a year moved, a district changed - and the claim
states the real one. A model that answers "supported" has read its own
memory instead of the passage, which is the failure that makes a RAG
system quietly wrong.
"""

from yoxla.benchmark.schema import (
    AnswerMode,
    AnswerSpec,
    BenchmarkTask,
    ChoiceSpec,
    GenerationSpec,
    PromptSpec,
)

BLOCK = "rag"

DATASET_REPO = "LocalDoc/YOXLA-Benchmark"

LANGUAGE = "az"


SUPPORTED = "TƏSDİQ"
CONTRADICTED = "ZİDD"
ABSENT = "YOXDUR"


RAG_VERIFICATION_V1 = BenchmarkTask(
    task_id="rag_verification_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="rag_verification_v1",
    dataset_split="test",
    expected_examples=160,
    input_fields=(
        "context",
        "claim",
    ),
    gold_field="label",
    metadata_fields=(
        "claim_type",
        "subject",
        "subject_qid",
        "source_id",
    ),
    # The score is an average over four claim types that fail in
    # different ways; one number hides which.
    breakdown_field="claim_type",
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə mətn və bir iddia veriləcək. "
            "Yalnız verilmiş mətnə əsaslanaraq qərar ver.\n"
            "\n"
            "Kateqoriyalar:\n"
            "{choice_catalog}\n"
            "\n"
            "Öz biliyinə deyil, yalnız mətndə yazılanlara "
            "əsaslan. Mətn səhv olsa belə, mətnə görə cavab ver.\n"
            "\n"
            "Cavab kimi yalnız kateqoriyanın adını yaz. "
            "Heç bir izah və ya əlavə mətn yazma."
        ),
        user=(
            "Mətn:\n"
            "{context}\n"
            "\n"
            "İddia:\n"
            "{claim}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=(
            ChoiceSpec(
                SUPPORTED,
                "mətn bu iddianı təsdiqləyir",
            ),
            ChoiceSpec(
                CONTRADICTED,
                "mətn bu iddia ilə ziddiyyət təşkil edir",
            ),
            ChoiceSpec(
                ABSENT,
                (
                    "mətndə bu barədə heç nə deyilmir - nə "
                    "təsdiq, nə inkar"
                ),
            ),
        ),
    ),
    generation=GenerationSpec(
        max_output_tokens=16,
    ),
    evaluator="classification_exact",
    primary_metric="accuracy",
    diagnostic_metrics=(
        "per_claim_type_accuracy",
        "per_class_accuracy",
        "macro_f1",
        "modal_answer_share",
        "invalid_output_rate",
    ),
)


NONE_OF_THEM = "HEÇ BİRİ"

# Six passages and the option of rejecting all of them, so chance is
# one in seven and the share of items with no answer is set to match.
PASSAGE_COUNT = 6


RAG_SELECTION_V1 = BenchmarkTask(
    task_id="rag_selection_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="rag_selection_v1",
    dataset_split="test",
    expected_examples=133,
    input_fields=(
        "question",
        "passages",
    ),
    gold_field="label",
    metadata_fields=(
        "item_type",
        "answer",
        "subject",
        "subject_qid",
        "category",
        "probe",
    ),
    # One item in seven has its answer in none of the passages, and
    # that is the column to read. A retrieval system that always
    # returns its best guess fails there and nowhere else.
    breakdown_field="item_type",
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə bir sual və nömrələnmiş mətn parçaları "
            "veriləcək.\n"
            "\n"
            "Sualın cavabı hansı parçada varsa, yalnız onun "
            "nömrəsini yaz.\n"
            "\n"
            "Əgər cavab heç bir parçada yoxdursa, "
            f"{NONE_OF_THEM} yaz.\n"
            "\n"
            "Heç bir izah, şərh və ya əlavə mətn yazma."
        ),
        user=(
            "Sual:\n"
            "{question}\n"
            "\n"
            "Parçalar:\n"
            "{passages}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=tuple(
            ChoiceSpec(
                value=str(number),
                description=f"cavab {number}-ci parçadadır",
            )
            for number in range(1, PASSAGE_COUNT + 1)
        )
        + (
            ChoiceSpec(
                value=NONE_OF_THEM,
                description=(
                    "cavab heç bir parçada yoxdur"
                ),
            ),
        ),
    ),
    generation=GenerationSpec(
        max_output_tokens=12,
    ),
    evaluator="classification_exact",
    primary_metric="accuracy",
    diagnostic_metrics=(
        "per_item_type_accuracy",
        "per_class_accuracy",
        "macro_f1",
        "modal_answer_share",
        "invalid_output_rate",
    ),
)


RAG_TASKS: tuple[BenchmarkTask, ...] = (
    RAG_VERIFICATION_V1,
    RAG_SELECTION_V1,
)

RETIRED_RAG_TASKS: tuple[BenchmarkTask, ...] = ()
