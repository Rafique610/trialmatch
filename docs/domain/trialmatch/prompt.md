# Prompt and answer format

Code: `src/data/prompt.py`. One contract shared by baseline (TRIA-6),
fine-tuning (TRIA-7), evaluation (TRIA-5/8) and the app (TRIA-12).

## Input (what the model sees)

Qwen chat template with two turns:

- **system:** "You are a clinical trial screening assistant. You judge one
  eligibility criterion against a patient note, using only what the note says."
- **user:**
  1. The patient note as numbered sentences, `0. …`, `1. …`. These are the
     dataset's own numbers.
  2. `Criterion type: inclusion|exclusion`
  3. `Criterion: <text>`
  4. The four allowed labels **for that type only**, each with a one-line
     meaning. The exclusion wording spells out that `excluded` means the
     patient *has* the exclusion condition.
  5. The instruction to answer with JSON only, use `[]` when no sentence is
     relevant, and never treat missing information as proof.

Trial title, GPT-4 fields and anything else are never in the prompt. The
inputs are exactly the brief's three: patient description, criterion type,
criterion text.

## Output (what the model must produce)

```json
{"label": "not enough information", "evidence_ids": [0]}
```

- Training target = this JSON from the expert label and expert evidence,
  then the end-of-turn token. Loss is on these answer tokens only (TRIA-7).
- No explanation is trained (ADR 0001). The app builds it from the cited sentences.

## Parsing (`parse_output`)

- Takes the first `{…}` that parses as JSON, so extra chatter around it is tolerated.
- The label is lower-cased and must be in that criterion type's label set.
  Otherwise the answer is a **parse error** and the label is `None`.
- Evidence ids: integers within the note's range are kept (sorted, deduplicated).
  Anything else goes into `dropped_ids` so evaluation can count it.
- No JSON at all → `parse_error = "no JSON object"`. Errors are scored as
  wrong, never mapped to a label.

## Token lengths (`results/truncation_report.json`, Qwen2.5 tokenizer)

| Split | Prompt tokens (median / p95 / max) | Training tokens max | Answer tokens (median / max) |
| --- | --- | --- | --- |
| train | 307 / 494 / 609 | 636 | 15 / 39 |
| val | 349 / 391 / 454 | 470 | 15 / 26 |
| test | 307 / 378 / 513 | 530 | 15 / 29 |

**`MAX_LEN = 1024`.** The longest training example (636 tokens) fits with
room to spare. **0 rows are truncated, so 0 gold evidence sentences are
lost.** If a future note were longer, the rule is: cut note sentences from
the end, never the criterion or the answer, and log every gold evidence
sentence removed. `lengths.py` implements and reports this.

The answers are short (median 15 tokens), so decoding benchmarks (TRIA-9/11)
set output lengths explicitly rather than relying on the natural answer length.
