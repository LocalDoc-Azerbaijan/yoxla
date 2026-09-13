# YOXLA

Azerbaijani LLM evaluation framework and benchmark.

YOXLA is split in two:

* the **framework** — this Python package: inference, benchmark
  execution, evaluation, scoring and reporting;
* the **benchmark data** —
  [`LocalDoc/YOXLA-Benchmark`](https://huggingface.co/datasets/LocalDoc/YOXLA-Benchmark)
  on Hugging Face, versioned independently.

Updating the framework never changes the benchmark, and updating the
benchmark never requires a new release of the package.

## Install

```bash
pip install yoxla
```

API providers (OpenAI, OpenRouter, Google AI Studio, vLLM,
llama.cpp):

```bash
pip install "yoxla[api]"
```

Local Hugging Face models:

```bash
pip install "yoxla[local]"
```

## Quick start

```bash
yoxla validate-benchmark --block understanding
```

```bash
yoxla run --provider openrouter --model google/gemini-3.1-flash-lite --block understanding
```

A single task, three examples, for a smoke check:

```bash
yoxla run --provider transformers --model Qwen/Qwen3-0.6B --task nli_v1 --limit 3
```

## Blocks

1443 frozen examples across four blocks.

### Understanding — 600

| Task               | Examples | Answer          | Primary metric   |
| ------------------ | -------- | --------------- | ---------------- |
| `extractive_qa_v1` | 200      | text            | `span_recall`    |
| `qa_abstention_v1` | 100      | text or abstain | `overall_score`  |
| `nli_v1`           | 100      | choice          | `accuracy`       |
| `sts_v1`           | 100      | numeric 0–5     | `spearman`       |
| `sentiment_v1`     | 50       | choice          | `accuracy`       |
| `intent_v1`        | 50       | choice          | `accuracy`       |

### Language — 400

| Task                    | Examples | Answer | Primary metric |
| ----------------------- | -------- | ------ | -------------- |
| `minimal_pairs_v1`      | 200      | choice | `accuracy`     |
| `az_tr_interference_v1` | 200      | choice | `accuracy`     |

Both show two sentences differing in one word and ask which one is
Azerbaijani. What differs is what the one word tests. Position is
balanced inside every breakdown of both, so answering "A" throughout
scores 50 on the task and 50 on each breakdown.

`minimal_pairs_v1` corrupts a named rule — vowel harmony, the
question particle, the definite accusative, case government and four
more — which is what makes the label derivable rather than a matter
of opinion.

`az_tr_interference_v1` replaces an Azerbaijani word with a Turkish
one:

```text
A) Uşaqlar məktəbdə çox gözəl danışmaq öyrənirlər.
B) Uşaqlar məktəbdə çox gözəl konuşmak öyrənirlər.
```

That is not a spelling slip. It is the failure mode of a model
trained mostly on the larger neighbouring language, and it is the one
measurement that separates a model which learned Azerbaijani from one
which learned Turkish and is guessing.

The gold comes from outside the project, because "this word is not
Azerbaijani" is a fact about a lexicon and not something derivable.
Three lookups in two independent sources decide every pair, and each
source is used only in the direction it can support:

| | |
| --- | --- |
| the Azerbaijani word | is in the hunspell dictionary and occurs ≥ 200 times in Azerbaijani Wikipedia |
| the Turkish word | is in the Turkish dictionary and **absent** from the Azerbaijani one |
| the corpus | has the Azerbaijani word outnumbering the Turkish one ≥ 50:1 |

Presence in the dictionary proves a form is Azerbaijani; absence
proves nothing, because its verb paradigms have holes — `soruşmaq` is
missing while `danışmaq` is present, and treating absence as proof
would mark correct Azerbaijani as Turkish. The corpus runs the other
way: a form seen tens of thousands of times is in use whatever the
dictionary says, but a low count is not proof either, since Turkish
words appear in Azerbaijani Wikipedia inside quotations and film
titles — `kitap` 166 times, `kent` 297. What separates a pair is the
ratio between its two forms, which runs from 54:1 to 80162:1 across
the set. Sources and thresholds: `data/lexicon/SOURCES.md` in the
dataset repository.

Whether that machinery was worth it is measured rather than argued.
Twenty pairs were written by hand before the sources were in place
and the dictionary rejected five of them: `araba`, `bilgisayar`,
`durum` and `subay` are all Azerbaijani words, and `sormaq` is a real
stem. A quarter of a hand-written table was wrong in a way no reader
would have noticed, and a model asked to produce the same table is
wrong for the same reason — it is a lexicographic fact, not a
judgement.

Every word of the Azerbaijani sentence is checked too, not only the
target, so the sentence a model is asked to prefer is verified
Azerbaijani throughout rather than merely correct in one position.

`per_interference_type_accuracy` splits the score two ways, and the
split matters more than the total:

| type | the pair | what it tests |
| --- | --- | --- |
| `cognate` | `kitab` / `kitap` | which sound shape Azerbaijani uses |
| `distinct_lexeme` | `danışmaq` / `konuşmak` | which word it uses at all |

Only the second is beyond a model that merely spells Azerbaijani
correctly. The set is 163 distinct pairs to 37 cognates, and on a
subset of 37 a single item is worth 2.7 points — read that column as
an indication, not a measurement.

### Knowledge — 150

| Task                  | Examples | Answer            | Primary metric |
| --------------------- | -------- | ----------------- | -------------- |
| `knowledge_choice_v1` | 150      | choice, 1–20      | `accuracy`     |

Facts about Azerbaijan harvested from Wikidata across seven
categories. Twenty candidates per question; the model returns a
number.

A fact Wikidata answers more than one way is dropped at build time.
Nizami Ganjavi carries three death years, so no question is built
from it: a model that knows one of the other two would be marked
wrong by the gold, not by its own error.

An open-answer version was built first and dropped. Sixty-two of the
hundred and fifty answers are multi-word entities — honorific titles
no two people would word alike, names with up to seven equally
correct forms — and every scoring dispute found in review came from
that group. Scoring them needs a judge or a list of accepted
spellings someone keeps guessing at, and neither belongs in a frozen
benchmark. Closing the answer space gives up the distinction between
recall and recognition; what it buys is that a wrong answer is wrong
because the model chose wrong, never because a check disagreed about
spelling.

A closed space is only worth something if the distractors are hard,
which is a property of the data rather than of the code that reads
it — so it is measured. Three cues would let a model answer without
knowing the fact, and each is closed at build time: distractors come
from the same probe, so the wrong *kind* of thing cannot be
eliminated; the answer is never the most-linked option, so "pick the
famous one" fails; a year distractor sits within a decade of the
answer, so knowing the century does not narrow the field. Cue-only
adversaries then run over the finished set and the build fails if any
beats its floor by more than five points.

`modal_answer_share` is the diagnostic to read beside the score. A
model that does not know the answer tends to return the same number
every time — one run sat at 0.7 — and accuracy alone does not show
it.

### RAG — 293

| Task                  | Examples | Answer          | Primary metric |
| --------------------- | -------- | --------------- | -------------- |
| `rag_verification_v1` | 160      | choice          | `accuracy`     |
| `rag_selection_v1`    | 133      | choice, 1–6 or none | `accuracy` |

The two halves of a retrieval system, measured apart:
`rag_selection_v1` asks whether the right passage was picked up,
`rag_verification_v1` asks what the model does with a passage once it
has one. A system can fail at either, and one number would hide which.

#### `rag_verification_v1`

A passage and one claim about it. The model says whether the passage
supports the claim, contradicts it, or is silent about it —
`TƏSDİQ` / `ZİDD` / `YOXDUR`.

The third label is what makes hallucination measurable. Asked only
"true or false", a model has nowhere to put a claim the passage never
addresses, and inventing support for it looks the same as reading
correctly.

Absence is decidable here in a way it is not for a found document,
because each passage is written from a known list of three facts.
Anything outside that list is provably not stated — a claim about a
death year in a passage that states only a birthplace is unsupported
by construction, not by judgement.

Four claims come from every passage, and `per_claim_type_accuracy` is
the number to read rather than the average:

| claim type | the passage | the claim | label |
| --- | --- | --- | --- |
| `supported` | states a fact | restates it | `TƏSDİQ` |
| `contradicted` | states a fact | alters it | `ZİDD` |
| `counterfactual` | states it **altered** | states the true value | `ZİDD` |
| `absent` | never mentions it | asserts it | `YOXDUR` |

The third row is worth the most. The passage says Nizami died in
1195; the claim says 1202, which is true of the world and false of
the passage. A model answering `TƏSDİQ` has read its own memory
instead of the text — the failure that makes a retrieval system
quietly wrong. A model can hold the first two labels and still fail
that column, and one averaged number hides it.

A fact Wikidata answers more than one way is not used here either,
and for a sharper reason than in Knowledge. The theatre and the
university in the harvest were both renamed, so each carries two
names — and a passage built from one of them while a claim asserted
the other produced a `ZİDD` that is not defensible: two periods are
not a contradiction, and a model answering `YOXDUR` would be reading
correctly. Those subjects are skipped, not relabelled.

The generator model wrote the passages and the claims; it never
decided a label. Every label follows from the fact list, and a
generation that did not match the list was rejected rather than
relabelled. The uploader re-derives all four labels from Wikidata and
the finished passage before publishing, and refuses to publish if the
arrangement it expects is not there.

#### `rag_selection_v1`

A question and six short passages, roughly seventy words in all. The
model answers with the number of the passage containing the answer,
or `HEÇ BİRİ` if none of them does.

The whole measurement is in the negatives, and the first attempt at
this set was abandoned for lack of them. Recombining the extractive QA
contexts would have been free, but those questions were written from
those contexts and inherit their wording: the true context had the
highest word overlap with the question in **187 of 200** items, so
"pick the passage sharing the most words" would have solved the task
without reading anything.

Here the passages are written from single facts about a known
subject, so every passage about that subject carries its name and
overlap stops pointing anywhere:

| negative | example, asked where Nizami was born |
| --- | --- |
| same subject, other aspect | his death year |
| same aspect, other subject | another poet's birthplace |
| same subject, same value kind | his death year against his birth year — one word of relation decides |

A fourth kind is deliberately absent. A passage stating an altered
value — "Nizami was born in Baku" — is relevant to the question and
merely false, so counting it as a wrong pick would conflate "not the
right passage" with "the right passage, wrong fact". The second thing
is what `rag_verification_v1` measures.

One item in seven has its answer in none of the passages, which is
what `per_item_type_accuracy` splits out and the number worth reading.
A retrieval system that always returns its best guess fails there and
nowhere else. The share is one in seven rather than more so that every
one of the seven answers is equally likely: at a quarter, answering
`HEÇ BİRİ` to everything scored 20% against a chance of 14%.

**One adversary here cannot be driven to chance, and the floor is
reported rather than hidden.** The gold is the only passage carrying
both the subject and the relation, and a negative carrying both would
be a passage stating the answer. So the gold is held to a *tie* with
the other passages about its subject — a word counter then faces two
or three equals and earns a third or a half, which is a floor near
0.35 rather than the 0.14 that every other adversary in this benchmark
sits at. It is computed over the finished set, printed at build time,
and re-checked by the uploader.

The other cue was found only after the set looked finished. Passages
about an institution run longer than passages about a person, simply
because the name is longer, so "pick the longest" scored 0.254 against
a permutation null of 0.143. The negatives are now matched to the gold
for length, and items where the gold is still the single longest are
dropped.

The set is 133 items rather than a round 200 because of those two
rules. Every item they removed was one where the gold gave itself
away, and loosening them to reach a round number would have bought the
number with the cues.

```bash
yoxla tasks
```

Each task score is normalized to `[0, 100]`, and the block score is
the **macro average over tasks** — not over examples, so the 200 QA
items do not outweigh the 50 intent items.

## Span answers

`extractive_qa_v1` and `qa_abstention_v1` put the answer inside a
passage the model is given, so they are scored on character offsets
rather than on strings. The quote is located in the passage and its
range compared with the gold's — no accepted forms, no normalizer,
nothing to maintain when a new way of writing the same span turns up.
The gold answer occurs exactly once in its own context in all 250
examples, which is what makes the position unambiguous; a run refuses
to start if that stops being true.

Two numbers come out of it, and they are reported apart because they
are different abilities:

| | |
| --- | --- |
| `span_recall` | did the quote cover the answer — **the score** |
| `span_precision` | how much else it dragged in |

Measured over five models, recall sits between 0.83 and 0.85 for the
top three while precision spreads from 0.48 to 0.92. The model that
ranked last on token F1 sits fourth on recall: most of its old deficit
was the length of its quotes, not its reading. Folding the two
together reads that as a comprehension failure, which it is not.

Recall is credited only when the quote stays inside the sentences the
answer occupies — otherwise copying the whole passage would score a
perfect recall. Models stay inside that window 79 to 95 per cent of
the time, so the guard bites only on what it is meant to catch.

`unsupported_rate` comes free from the same machinery: a quote that
cannot be found in the passage was not copied from it.

## How answers are judged

Parsing is strict: only the requested answer format counts.

```text
neutral                  valid
`neutral`   "neutral"    valid   (symmetric wrappers are removed)
Neutral                  valid   (case-insensitive, AZ-aware)
Neutral.                 invalid
The answer is neutral    invalid
```

An invalid answer is a result, never a reason to re-prompt: one
benchmark example is one model attempt. Answer normalization never
folds Azerbaijani letters (`ə → e`, `ş → s`, `ı → i`), because that
would reward models that do not write correct Azerbaijani.

Retries exist only for transport failures (timeouts, 429, 5xx).

## What the benchmark does not measure

Every answer space is closed: a label, a number, or a span quoted
from a passage the model was given. That is what lets every score be
checked without a judge and reproduced from stored output — and it
draws a hard boundary around what the numbers can mean.

**Production is not measured.** No task asks the model to write
Azerbaijani and scores what it wrote. An orthography block did
exactly that — a short text from a handful of facts, scored on
whether the spelling rules survived — and it is gone, along with the
open-answer knowledge task. The same model recognises correct
Azerbaijani grammar at 90.5 and, in the same run, scored 57.7 on
writing it, so the two are not interchangeable and the second number
no longer exists anywhere in the benchmark.

**Recall is not measured, only recognition.** Knowledge asks the
model to pick a fact out of twenty candidates, never to produce it,
and the interference task asks it to recognize a Turkish word rather
than to avoid writing one. A model can score well on both while its
own Azerbaijani drifts toward Turkish, and nothing here would show
it.

**Nothing above sentence level is measured.** No task covers
discourse, long context, multi-turn behaviour, or instruction
following beyond answer format.

A benchmark that scores every one of these would need either a judge
model or human annotation. Both were considered and neither is used:
a judge would put one model's Azerbaijani inside the loop that
measures Azerbaijani, and would end reproducibility — `rescore` would
stop reproducing a number and a frozen manifest would guarantee
nothing.

## Reproducibility

Every run writes:

```text
runs/<run_id>/
├── run.json            model, provider, YOXLA version, benchmark
│                       version, dataset revision and fingerprint,
│                       prompt version and prompt hashes
├── predictions.jsonl   one line per example, raw output included
├── task_scores.json    metrics and normalized score per task
└── summary.json        block and benchmark aggregates
```

Pin the data for a comparable run:

```bash
yoxla run --provider openai --model gpt-5 --block understanding --revision a66c827c36b249a6137eda67a0110da469b4b175
```

`yoxla validate-benchmark` compares the data against the frozen
manifest bundled with the package and warns when the Hub content has
changed.

Raw model output is always stored, so metrics can be recomputed
without paying for inference again:

```bash
yoxla rescore runs/openrouter_some-model_20260818T135510Z
```

`rescore` replays the stored answers through the current evaluators
and rewrites `task_scores.json` and `summary.json`. Model answers are
never modified.

### Span-based tasks

Scoring these on one number used to blend two abilities, and it
ranked models almost inversely. Under `token_f1` gpt-4o-mini came
last while locating the answer more often than any other model,
because it returned the sentence around the span rather than the
span.

The offset metrics separate them — see [Span answers](#span-answers).
The same five runs, rescored:

| model | `span_recall` | `span_precision` | over-sentence | old `token_f1` |
| --- | --- | --- | --- | --- |
| gemini-3.1-flash-lite | 0.849 | 0.915 | 2.0% | 0.942 |
| mistral-small-4 | 0.848 | 0.738 | 2.0% | 0.844 |
| llama-3.3-70b | 0.834 | 0.884 | 2.5% | 0.901 |
| gpt-4o-mini | 0.794 | 0.483 | 10.5% | 0.664 |
| qwen2.5-7b | 0.630 | 0.693 | 4.0% | 0.725 |

The top three find the answer equally well and are separated in the
old number mostly by how tightly they quote — which is now its own
column rather than a hidden term in the score. gpt-4o-mini moves from
last to fourth: its `token_f1` of 0.664 was largely a brevity penalty.

The separation is not total. Recall is forfeited when a quote runs
past the sentence holding the answer, so a model that habitually
returns two sentences — gpt-4o-mini does it in one example out of ten
— still loses recall for it. Without that guard recall would be free:
copy the passage, score 1.0.

`token_f1`, `normalized_em` and `strict_em` stay as diagnostics. They
are what earlier runs were scored on, and keeping them visible is what
makes the change in the headline number explainable.

### Running in parallel

A full run is 1443 requests and by default each waits for the last.
`--workers` puts several in flight:

```bash
yoxla run --provider openrouter --model some/model --block all --workers 8
```

Requests are issued from a pool but **results are consumed in example
order**, so `predictions.jsonl` reads the same whatever the schedule
was, `rescore` still replays it, and an interrupted run still resumes
from the last line on disk. On a synthetic 60-example run the speedup
is close to linear — ×3.9 at four workers, ×9.7 at ten — and the score
is unchanged.

**The default stays 1 because the schedule can move the score.** Under
load a provider answers 429 and times out more often; an example whose
retries run out is recorded as a generation error, and that counts
against the model. A model run on twenty threads can therefore score
below the same model run on one, for a reason that has nothing to do
with the model. The worker count is written into `run.json`, and the
summary prints `generation errors` — check it is zero before comparing
two runs.

Local Transformers models gain nothing here: they serialize on the
device.

### Resuming

`predictions.jsonl` is flushed after every example. If a long API run
dies, continue it with the same run id:

```bash
yoxla run --provider openrouter --model some/model --block understanding --run-id my-run --resume
```

Cached examples are replayed into the evaluators, so a resumed run
scores exactly like an uninterrupted one.

## Providers

Providers describe *how to connect*; the model id is given at
runtime, so a newly released model needs no change to YOXLA.

```bash
yoxla providers
```

```text
openrouter   openai_compatible   OPENROUTER_API_KEY
google       openai_compatible   GEMINI_API_KEY
openai       openai_compatible   OPENAI_API_KEY
vllm         openai_compatible   http://127.0.0.1:8000/v1
llamacpp     openai_compatible   http://127.0.0.1:8080/v1
transformers transformers        local / Hugging Face
```

A custom registry can be supplied with `--config my-providers.yaml`.

### Reasoning models

Benchmark v1 runs with thinking disabled. That has to be stated to
the provider, not assumed: OpenRouter turns reasoning on by default
for models that support it, and some of them default to a high effort
that spends the entire output budget before producing an answer
token — the result is an empty response with `finish_reason: length`.

The `openrouter` provider therefore declares:

```yaml
thinking_control: reasoning
```

which sends `reasoning: {"enabled": false}` (or `true` under
`--thinking`). llama.cpp uses `chat_template_kwargs` for the same
purpose. Providers without a declared control are sent no reasoning
parameter at all.

When a model does return reasoning, it is preserved in `raw_text`
inside `<think>` tags while `text` — the scored field — holds only the
final answer.

Connection smoke test:

```bash
yoxla test --provider openrouter --model google/gemini-3.1-flash-lite
```

## Adding a task

A task is a registry entry, not a code change to the runner. It
declares its dataset, field mapping, prompt, answer space, generation
settings, parser mode and evaluator — see
the modules under `src/yoxla/benchmark/tasks/`.

A task id such as `nli_v1` freezes the whole contract: dataset,
prompt, answer space and evaluator. Any substantial change means a
new id (`nli_v2`), never an edit in place, and the superseded task
stays registered and runnable so an older run can be reproduced.

**That freeze starts at first publication, not at first commit.**
All four blocks are now on the Hub, so the nine task ids are frozen
and a substantial change to any of them means a new id. What was
changed in place before publication was changed at `v1`, which is why
there is no `v2` anywhere. No task is retired.

## Status

Implemented: inference (API, local Transformers, bitsandbytes/Unsloth,
GGUF via llama.cpp), the Understanding, Language, Knowledge and RAG
blocks, deterministic evaluators, scoring, run artifacts, resume and
validation.

Not implemented yet: RAG extraction, Language Under Load, and the
leaderboard.

AZ/TR purity was on that list and is now half done. Recognizing a
Turkish word in an Azerbaijani sentence is measured
(`az_tr_interference_v1`); whether a model's own output stays
Azerbaijani is not, and cannot be without scoring produced text.

Removed rather than postponed: the generation block and the
open-answer knowledge task, both of which needed a judgement about
what counts as the same answer — see
[What the benchmark does not measure](#what-the-benchmark-does-not-measure).

## License

Apache-2.0
