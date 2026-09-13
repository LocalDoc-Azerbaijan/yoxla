"""
Understanding v1 - 600 frozen examples.

    extractive_qa_v1     200
    qa_abstention_v1     100
    nli_v1               100
    sts_v1               100
    sentiment_v1          50
    intent_v1             50
                         ---
    TOTAL                600

A task id such as ``nli_v1`` freezes the whole contract: dataset,
prompt, answer space and evaluator. Any substantial change to any of
them means a new task id (``nli_v2``), never an edit in place.
"""

from yoxla.benchmark.schema import (
    AnswerMode,
    AnswerSpec,
    BenchmarkTask,
    ChoiceSpec,
    GenerationSpec,
    PromptSpec,
    RubricLevel,
)

BLOCK = "understanding"

DATASET_REPO = "LocalDoc/YOXLA-Benchmark"

LANGUAGE = "az"


# ==================================================================
# Extractive QA
# ==================================================================

EXTRACTIVE_QA_V1 = BenchmarkTask(
    task_id="extractive_qa_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="extractive_qa_v1",
    dataset_split="test",
    expected_examples=200,
    input_fields=(
        "context",
        "question",
    ),
    gold_field="answer",
    metadata_fields=(
        "title",
        "answer_start",
        "answer_type",
        "context_bucket",
        "source_id",
    ),
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə kontekst və sual veriləcək. "
            "Sualın cavabını yalnız verilmiş kontekstdən tap.\n"
            "Cavabı kontekstdən eynilə köçür və sualı "
            "cavablandıran ən qısa mətn parçasını seç.\n"
            "Cümləni bütövlükdə yazma və cavaba aid olmayan "
            "sözləri əlavə etmə.\n"
            "Heç bir izah, şərh və ya əlavə mətn yazma."
        ),
        user=(
            "Kontekst:\n"
            "{context}\n"
            "\n"
            "Sual:\n"
            "{question}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.TEXT,
    ),
    generation=GenerationSpec(
        max_output_tokens=128,
    ),
    evaluator="span_qa",
    # Finding the answer is the ability this block is named after.
    # Quoting it tightly is formatting, and it is reported beside the
    # score instead of inside it - see evaluators/span.py.
    primary_metric="span_recall",
    diagnostic_metrics=(
        "span_precision",
        "exact_span_rate",
        "unsupported_rate",
        "over_sentence_rate",
        "token_f1",
        "normalized_em",
        "strict_em",
        "invalid_output_rate",
    ),
)


# ==================================================================
# QA abstention
# ==================================================================

ABSTAIN_VALUE = "Cavab yoxdur"

QA_ABSTENTION_V1 = BenchmarkTask(
    task_id="qa_abstention_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="qa_abstention_v1",
    dataset_split="test",
    expected_examples=100,
    input_fields=(
        "context",
        "question",
    ),
    gold_field="answer",
    metadata_fields=(
        "title",
        "is_answerable",
        "expected_answer_type",
        "context_bucket",
        "source_id",
    ),
    gold_nullable=True,
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə kontekst və sual veriləcək. "
            "Sualın cavabını yalnız verilmiş kontekstdən tap.\n"
            "Əgər sualı cavablandırmaq üçün kontekstdə kifayət qədər "
            "məlumat yoxdursa, yalnız \"{abstain_value}\" yaz.\n"
            "Əgər cavab kontekstdə varsa, cavab kimi yalnız "
            "kontekstdəki uyğun mətn parçasını yaz.\n"
            "Heç bir izah, şərh və ya əlavə mətn yazma."
        ),
        user=(
            "Kontekst:\n"
            "{context}\n"
            "\n"
            "Sual:\n"
            "{question}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.TEXT_OR_ABSTAIN,
        abstain_value=ABSTAIN_VALUE,
    ),
    generation=GenerationSpec(
        max_output_tokens=128,
    ),
    evaluator="span_abstention",
    # An example scores only when the model both decides answerability
    # correctly and, for an answerable question, quotes the span that
    # carries the answer. The span half is credited with recall for
    # the same reason as in extractive QA.
    primary_metric="overall_score",
    diagnostic_metrics=(
        "answerability_accuracy",
        "unanswerable_accuracy",
        "answerable_span_recall",
        "answerable_span_precision",
        "answerable_unsupported_rate",
        "answerable_over_sentence_rate",
        "answerable_token_f1",
        "answerable_normalized_em",
        "invalid_output_rate",
    ),
)


# ==================================================================
# NLI
# ==================================================================

NLI_V1 = BenchmarkTask(
    task_id="nli_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="nli_v1",
    dataset_split="test",
    expected_examples=100,
    input_fields=(
        "premise",
        "hypothesis",
    ),
    gold_field="label",
    metadata_fields=(
        "length_bucket",
        "source_id",
    ),
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə iki cümlə veriləcək. Birinci cümlənin doğru "
            "olduğunu qəbul edərək ikinci cümlənin onunla məntiqi "
            "əlaqəsini müəyyən et.\n"
            "\n"
            "Kateqoriyalar:\n"
            "{choice_catalog}\n"
            "\n"
            "Cavab kimi yalnız kateqoriyanın adını yaz.\n"
            "Heç bir izah və ya əlavə mətn yazma."
        ),
        user=(
            "Birinci cümlə:\n"
            "{premise}\n"
            "\n"
            "İkinci cümlə:\n"
            "{hypothesis}\n"
            "\n"
            "Cavab:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=(
            ChoiceSpec(
                "entailment",
                (
                    "birinci cümlə doğru olduqda ikinci cümlə də "
                    "mütləq doğrudur"
                ),
            ),
            ChoiceSpec(
                "neutral",
                (
                    "ikinci cümlənin doğru və ya yanlış olduğunu "
                    "yalnız birinci cümləyə əsasən müəyyən etmək "
                    "mümkün deyil"
                ),
            ),
            ChoiceSpec(
                "contradiction",
                (
                    "birinci cümlə doğru olduqda ikinci cümlə "
                    "yanlışdır və ya onunla ziddiyyət təşkil edir"
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
        "macro_f1",
        "per_class_accuracy",
        "invalid_output_rate",
    ),
)


# ==================================================================
# STS
# ==================================================================

STS_V1 = BenchmarkTask(
    task_id="sts_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="sts_v1",
    dataset_split="test",
    expected_examples=100,
    input_fields=(
        "sentence1",
        "sentence2",
    ),
    gold_field="score",
    metadata_fields=(
        "scaled_score",
        "score_band",
        "length_bucket",
        "source_id",
    ),
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə iki cümlə veriləcək. Onların semantik məna "
            "oxşarlığını 0-dan 5-ə qədər qiymətləndir.\n"
            "\n"
            "Şkala:\n"
            "{rubric_catalog}\n"
            "\n"
            "Cavab kimi yalnız 0 ilə 5 arasında bir ədəd yaz. "
            "Onluq kəsr ədəd istifadə edə bilərsən.\n"
            "Heç bir izah, şərh və ya əlavə mətn yazma."
        ),
        user=(
            "Birinci cümlə:\n"
            "{sentence1}\n"
            "\n"
            "İkinci cümlə:\n"
            "{sentence2}\n"
            "\n"
            "Oxşarlıq balı:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.NUMERIC,
        minimum=0.0,
        maximum=5.0,
        rubric=(
            RubricLevel(
                "0",
                "cümlələrin mənaları əlaqəsizdir",
            ),
            RubricLevel(
                "1",
                (
                    "cümlələr yalnız zəif şəkildə əlaqəlidir və "
                    "əsas mənaları fərqlidir"
                ),
            ),
            RubricLevel(
                "2",
                (
                    "bəzi məna oxşarlığı var, lakin əhəmiyyətli "
                    "fərqlər mövcuddur"
                ),
            ),
            RubricLevel(
                "3",
                (
                    "əsas məna qismən oxşardır, lakin mühüm məlumat "
                    "fərqləri var"
                ),
            ),
            RubricLevel(
                "4",
                (
                    "cümlələr əsasən eyni mənanı ifadə edir, yalnız "
                    "kiçik fərqlər var"
                ),
            ),
            RubricLevel(
                "5",
                (
                    "cümlələr semantik baxımdan eyni və ya demək "
                    "olar ki, tam ekvivalentdir"
                ),
            ),
        ),
    ),
    generation=GenerationSpec(
        max_output_tokens=16,
    ),
    evaluator="sts_regression",
    primary_metric="spearman",
    diagnostic_metrics=(
        "pearson",
        "spearman_valid_only",
        "pearson_valid_only",
        "mae",
        "rmse",
        "invalid_output_rate",
    ),
)


# ==================================================================
# Sentiment
# ==================================================================

SENTIMENT_V1 = BenchmarkTask(
    task_id="sentiment_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="sentiment_v1",
    dataset_split="test",
    expected_examples=50,
    input_fields=("text",),
    gold_field="label",
    metadata_fields=(
        "length_bucket",
        "source_id",
    ),
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə Azərbaycan dilində mətn veriləcək. Mətnin əsas "
            "emosional münasibətini müəyyən et.\n"
            "\n"
            "Kateqoriyalar:\n"
            "{choice_catalog}\n"
            "\n"
            "Cavab kimi yalnız kateqoriyanın adını yaz.\n"
            "Heç bir izah və ya əlavə mətn yazma."
        ),
        user=(
            "Mətn:\n"
            "{text}\n"
            "\n"
            "Sentiment:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=(
            ChoiceSpec(
                "positive",
                (
                    "mətn əsasən müsbət münasibət, razılıq, sevinc, "
                    "tərif və ya məmnunluq ifadə edir"
                ),
            ),
            ChoiceSpec(
                "neutral",
                (
                    "mətn əsasən neytral məlumat verir və aydın "
                    "müsbət və ya mənfi münasibət dominant deyil"
                ),
            ),
            ChoiceSpec(
                "negative",
                (
                    "mətn əsasən mənfi münasibət, narazılıq, tənqid, "
                    "qəzəb, kədər və ya məyusluq ifadə edir"
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
        "macro_f1",
        "per_class_accuracy",
        "invalid_output_rate",
    ),
)


# ==================================================================
# Intent
# ==================================================================

# Every intent carries an Azerbaijani description. Without them the
# task would partly measure prior knowledge of the Banking77 label
# taxonomy instead of understanding of the customer request.
INTENT_CHOICES = (
    ChoiceSpec(
        "card_arrival",
        (
            "sifariş edilmiş kartın nə vaxt çatacağı və ya hələ də "
            "gəlib çatmaması"
        ),
        group="CARD",
    ),
    ChoiceSpec(
        "card_delivery_estimate",
        "kartın çatdırılması üçün təxmini müddət və ya tarix",
        group="CARD",
    ),
    ChoiceSpec(
        "card_not_working",
        "mövcud kartın işləməməsi və ya ödənişi qəbul etməməsi",
        group="CARD",
    ),
    ChoiceSpec(
        "declined_card_payment",
        "kartla edilən ödənişin rədd edilməsi",
        group="CARD",
    ),
    ChoiceSpec(
        "card_payment_not_recognised",
        "hesabda tanınmayan kart ödənişinin görünməsi",
        group="CARD",
    ),
    ChoiceSpec(
        "pending_card_payment",
        "kart ödənişinin gözləmədə qalması və tamamlanmaması",
        group="CARD",
    ),
    ChoiceSpec(
        "transaction_charged_twice",
        "eyni əməliyyat üçün iki dəfə pul tutulması",
        group="ACCOUNT",
    ),
    ChoiceSpec(
        "cash_withdrawal_not_recognised",
        "hesabda tanınmayan nağd pul çıxarışının görünməsi",
        group="ATM",
    ),
    ChoiceSpec(
        "declined_cash_withdrawal",
        "bankomatdan nağd pul çıxarışının rədd edilməsi",
        group="ATM",
    ),
    ChoiceSpec(
        "pending_cash_withdrawal",
        "nağd pul çıxarışının gözləmədə qalması",
        group="ATM",
    ),
    ChoiceSpec(
        "wrong_amount_of_cash_received",
        (
            "bankomatdan gözləniləndən fərqli məbləğdə nağd pul "
            "alınması"
        ),
        group="ATM",
    ),
    ChoiceSpec(
        "pending_transfer",
        "pul köçürməsinin gözləmədə qalması",
        group="TRANSFER",
    ),
    ChoiceSpec(
        "failed_transfer",
        "pul köçürməsinin uğursuz olması və tamamlanmaması",
        group="TRANSFER",
    ),
    ChoiceSpec(
        "declined_transfer",
        "pul köçürməsinin sistem tərəfindən rədd edilməsi",
        group="TRANSFER",
    ),
    ChoiceSpec(
        "transfer_not_received_by_recipient",
        "köçürülmüş pulun alıcı tərəfindən alınmaması",
        group="TRANSFER",
    ),
    ChoiceSpec(
        "transfer_fee_charged",
        "pul köçürməsi üçün haqq tutulması",
        group="FEES",
    ),
    ChoiceSpec(
        "pending_top_up",
        "balans artırma əməliyyatının gözləmədə qalması",
        group="ACCOUNT",
    ),
    ChoiceSpec(
        "top_up_failed",
        "balans artırma əməliyyatının uğursuz olması",
        group="ACCOUNT",
    ),
    ChoiceSpec(
        "top_up_reverted",
        "artırılmış balansın geri qaytarılması və ya itməsi",
        group="ACCOUNT",
    ),
    ChoiceSpec(
        "top_up_by_card_charge",
        "kartla balans artırarkən haqq tutulması",
        group="FEES",
    ),
    ChoiceSpec(
        "exchange_rate",
        (
            "valyuta məzənnəsinin necə müəyyən edilməsi haqqında "
            "ümumi sual"
        ),
        group="OTHER",
    ),
    ChoiceSpec(
        "card_payment_wrong_exchange_rate",
        "kart ödənişində yanlış valyuta məzənnəsinin tətbiqi",
        group="CARD",
    ),
    ChoiceSpec(
        "wrong_exchange_rate_for_cash_withdrawal",
        (
            "nağd pul çıxarışında yanlış valyuta məzənnəsinin "
            "tətbiqi"
        ),
        group="ATM",
    ),
    ChoiceSpec(
        "unable_to_verify_identity",
        "şəxsiyyətin təsdiqlənməsi prosesinin uğursuz olması",
        group="ACCOUNT",
    ),
    ChoiceSpec(
        "verify_my_identity",
        "şəxsiyyətin necə təsdiqlənəcəyi haqqında sual",
        group="ACCOUNT",
    ),
)

INTENT_V1 = BenchmarkTask(
    task_id="intent_v1",
    block=BLOCK,
    language=LANGUAGE,
    dataset_repo=DATASET_REPO,
    dataset_config="intent_v1",
    dataset_split="test",
    expected_examples=50,
    input_fields=("text",),
    gold_field="label",
    metadata_fields=(
        "category",
        "length_bucket",
        "source_id",
    ),
    prompt=PromptSpec(
        version=1,
        system=(
            "Sənə bank müştərisinin Azərbaycan dilində müraciəti "
            "veriləcək. Müraciətin əsas məqsədini aşağıdakı intent "
            "kateqoriyalarından birinə aid et.\n"
            "\n"
            "Intent kateqoriyaları:\n"
            "{choice_catalog}\n"
            "\n"
            "Cavab kimi yalnız intent adını yaz.\n"
            "Heç bir izah, şərh, JSON və ya əlavə mətn yazma."
        ),
        user=(
            "Müştəri müraciəti:\n"
            "{text}\n"
            "\n"
            "Intent:"
        ),
    ),
    answer=AnswerSpec(
        mode=AnswerMode.CHOICE,
        choices=INTENT_CHOICES,
    ),
    generation=GenerationSpec(
        max_output_tokens=24,
    ),
    evaluator="classification_exact",
    primary_metric="accuracy",
    diagnostic_metrics=(
        "macro_f1",
        "per_class_accuracy",
        "group_accuracy",
        "invalid_output_rate",
    ),
)


# ==================================================================
# Span tasks, v2
# ==================================================================

# v1 asked for "the matching text fragment" and left the granularity
# open. Models split into two camps: some answered with the minimal
# span, others copied the whole sentence containing it. Both obeyed
# the instruction, and token F1 punished the second camp for a
# formatting choice the prompt never made - one model reached 0.98
# containment (it found the right place in the text almost every
# time) and still scored below a far weaker model that answered
# briefly but located the answer in only 63% of cases.
#
# v2 states the granularity: copy verbatim, choose the shortest span,
# do not write the whole sentence. The datasets are untouched; only
# the instruction changed, which is why this is a new task id.
#
# v2 reduced the effect but did not remove it. Under this prompt, five
# models still answer at very different lengths, and token F1 ranks
# them almost inversely to their ability to locate the answer:
#
#                          token_f1  containment  length/gold
#   gemini-3.1-flash-lite     0.942        0.825         0.91
#   llama-3.3-70b             0.901        0.800         0.99
#   mistral-small-4           0.844        0.915         1.33
#   qwen2.5-7b                0.725        0.610         1.12
#   gpt-4o-mini               0.664        0.970         2.01
#
# gpt-4o-mini finds the answer in 97% of examples - more than any
# other model - and scores last, because it returns the sentence
# around the span. The metric therefore blends comprehension with
# brevity, and no wording tested so far separates them. Splitting the
# two into separate metrics was considered and deliberately deferred;
# until then, read containment next to the score.

# The instruction states the granularity on purpose. Without it a
# model could answer with the whole sentence containing the answer and
# still obey the instruction.
#
# That was once the whole defence, and it was not enough: one model
# located the answer more often than any other and still ranked last,
# because it returned the sentence around the span. The blend of
# comprehension and brevity is now handled where it belongs - span
# recall and span precision are reported apart, see
# evaluators/span.py - so the wording no longer carries it alone.


UNDERSTANDING_TASKS = (
    EXTRACTIVE_QA_V1,
    QA_ABSTENTION_V1,
    NLI_V1,
    STS_V1,
    SENTIMENT_V1,
    INTENT_V1,
)

# Nothing has been published from this benchmark yet, so there is no
# earlier run for a retired task to keep reproducible. A task id is
# frozen from its first publication, not from its first commit.
RETIRED_UNDERSTANDING_TASKS: tuple[BenchmarkTask, ...] = ()
