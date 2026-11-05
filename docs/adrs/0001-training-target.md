# ADR 0001: Train on label + evidence ids; build the explanation from cited sentences

Date: 5 November 2026 · Status: accepted (project owner)

## Context

The brief asks each prediction to return a label, cited sentence ids, the
evidence text and "an explanation based only on the available information".
The dataset has expert labels and expert evidence sentence ids, but no expert
explanation. Its only explanations were written by GPT-4, and the brief
forbids using GPT-4 outputs as inputs or leaking them into evaluation.

## Options

| Option | Model output | Problem |
| --- | --- | --- |
| A. Label + ids | `{"label", "evidence_ids"}` | The model writes no free-text reason |
| B. Label + ids + our own written reason | adds `"explanation"` | No expert text to learn from; the model learns invented text or hallucinates reasons |
| C. GPT-4 explanations as targets | adds `"explanation"` | Forbidden (rubric B2) |

## Decision

Option A. The model is trained to output only `{"label": ..., "evidence_ids": [...]}`.
At inference, the explanation is assembled from the cited sentences, for example:

- `Not included: sentence [2] says "...", which conflicts with "<criterion>".`
- `Not enough information: no sentence in the note addresses this criterion.`

## Consequences

- Every explanation is traceable to sentence ids and cannot contain facts
  missing from the note.
- Shorter targets mean faster training on 6 GB and faster decoding.
- Explanations are templated rather than fluent. This is stated in the report's
  Limitations.
- Decoding benchmarks (TRIA-9, 11) set output lengths explicitly, so short
  answers don't limit them.

## How to revert

Add an `explanation` field to the target in `src/data/prompt.py` (TRIA-4) and
retrain. Option C stays forbidden.
