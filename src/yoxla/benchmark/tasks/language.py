"""
Language v1 - 400 frozen examples.

    minimal_pairs_v1        200
    az_tr_interference_v1   200

Both tasks show two sentences differing in one word and ask which one
is Azerbaijani. What differs is what the one word tests.

`minimal_pairs_v1` corrupts a grammatical rule - vowel harmony, the
question particle, an intervocalic q/k alternation, numeral-noun
agreement, predicate agreement, the definite accusative, the
possessive izafet, case government - so the gold is derivable from a
named rule rather than a matter of opinion.

`az_tr_interference_v1` replaces an Azerbaijani word with a Turkish
one. That is not a spelling slip but the failure mode of a model
trained mostly on the larger neighbouring language, and it is the one
measurement that separates a model which learned Azerbaijani from one
which learned Turkish and is guessing.

Position is balanced inside every breakdown of both tasks, so a model
that always answers A scores 50 on the task and 50 on each breakdown.
"""

from yoxla.benchmark.schema import (
    AnswerMode,
    AnswerSpec,
    BenchmarkTask,
    ChoiceSpec,
    GenerationSpec,
    PromptSpec,
)

BLOCK = "language"

DATASET_REPO = "LocalDoc/YOXLA-Benchmark"

LANGUAGE = "az"


MINIMAL_PAIRS_V1 = BenchmarkTask(
    task_id="minimal_pairs_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="minimal_pairs_v1",
    dataset_split="test",
    expected_examples=200,
    input_fields=(
        "sentence_a",
        "sentence_b",
    ),
    gold_field="label",
    metadata_fields=(
        "phenomenon",
        "target_correct",
        "target_corrupted",
        "domain",
        "difficulty",
        "source_id",
    ),
    breakdown_field="phenomenon",
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə Azərbaycan dilində iki cümlə veriləcək. "
            "Onlardan yalnız biri qrammatik cəhətdən "
            "düzgündür.\n"
            "\n"
            "Cavab kimi yalnız düzgün variantın hərfini yaz: "
            "A və ya B.\n"
            "\n"
            "Heç bir izah, şərh və ya əlavə mətn yazma."
        ),
        user=(
            "A) {sentence_a}\n"
            "B) {sentence_b}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=(
            ChoiceSpec(
                value="A",
                description="birinci cümlə düzgündür",
            ),
            ChoiceSpec(
                value="B",
                description="ikinci cümlə düzgündür",
            ),
        ),
    ),
    generation=GenerationSpec(
        max_output_tokens=8,
    ),
    evaluator="classification_exact",
    primary_metric="accuracy",
    diagnostic_metrics=(
        "macro_f1",
        "per_class_accuracy",
        "per_phenomenon_accuracy",
        "invalid_output_rate",
    ),
)


AZ_TR_INTERFERENCE_V1 = BenchmarkTask(
    task_id="az_tr_interference_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="az_tr_interference_v1",
    dataset_split="test",
    expected_examples=200,
    input_fields=(
        "sentence_a",
        "sentence_b",
    ),
    gold_field="label",
    metadata_fields=(
        "interference_type",
        "azerbaijani_word",
        "turkish_word",
        "az_count",
        "tr_count",
    ),
    # The two types fail for different reasons and one number hides
    # which. A cognate pair - kitab/kitap - is an orthographic
    # question; a distinct pair - danışmaq/konuşmak - is a lexical
    # one, and only the second is beyond a model that merely spells
    # Azerbaijani correctly.
    breakdown_field="interference_type",
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə iki cümlə veriləcək. Onlardan yalnız biri "
            "Azərbaycan dilindədir; digərində Azərbaycan "
            "dilinə aid olmayan bir söz var.\n"
            "\n"
            "Cavab kimi yalnız düzgün variantın hərfini yaz: "
            "A və ya B.\n"
            "\n"
            "Heç bir izah, şərh və ya əlavə mətn yazma."
        ),
        user=(
            "A) {sentence_a}\n"
            "B) {sentence_b}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=(
            ChoiceSpec(
                value="A",
                description="birinci cümlə Azərbaycan dilindədir",
            ),
            ChoiceSpec(
                value="B",
                description="ikinci cümlə Azərbaycan dilindədir",
            ),
        ),
    ),
    generation=GenerationSpec(
        max_output_tokens=8,
    ),
    evaluator="classification_exact",
    primary_metric="accuracy",
    diagnostic_metrics=(
        "macro_f1",
        "per_class_accuracy",
        "per_interference_type_accuracy",
        "invalid_output_rate",
    ),
)


LANGUAGE_TASKS: tuple[BenchmarkTask, ...] = (
    MINIMAL_PAIRS_V1,
    AZ_TR_INTERFERENCE_V1,
)

RETIRED_LANGUAGE_TASKS: tuple[BenchmarkTask, ...] = ()
