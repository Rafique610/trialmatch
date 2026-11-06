# ADR 0004: Target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct

Date: 5 November 2026 · Status: proposed (TRIA-2, awaiting owner review)

## Context

We need an open-weight decoder that can be fine-tuned and run on an RTX 3050
with 6 GB, plus a smaller draft model with an identical tokenizer for greedy
speculative decoding. Facts: `docs/research/models.md`, from
`results/model_candidates.json` and `results/tokenizer_compat.json`.

## Options

| | Qwen2.5-1.5B | Qwen2.5-3B | Llama-3.2-1B | Llama-3.2-3B |
| --- | --- | --- | --- | --- |
| Params | 1.54 B | 3.09 B | 1.24 B | 3.21 B |
| Licence | Apache-2.0 | Qwen Research (research/evaluation only — allows our use) | Llama 3.2 Community | Llama 3.2 Community |
| Gated download | no | no | yes | yes |
| fp16 / NF4 weights | 2.88 / 1.04 GB | 5.75 / 1.87 GB | 2.30 / 0.94 GB | 5.98 / 2.05 GB |
| QLoRA training estimate | ≈ 3.3 GB | ≈ 4.6 GB | — | — |
| KV cache / 1k tokens | 28 MB | 36 MB | 32 MB | 112 MB |
| Same-tokenizer draft | Qwen2.5-0.5B (verified) | Qwen2.5-0.5B / 1.5B | none smaller in family | Llama-3.2-1B |
| Context | 32k | 32k | 128k | 128k |

## Decision

Target **Qwen/Qwen2.5-1.5B-Instruct**, draft **Qwen/Qwen2.5-0.5B-Instruct**.

- Fits the GPU with headroom for both training (≈ 3.3 GB QLoRA estimate) and
  inference with both models in fp16 (≈ 3.8 GB plus small caches). Exact
  fp16 token comparisons are possible for Tasks 3 and 5.
- Apache-2.0 and ungated, so the examiner can download it without accepting
  terms or holding an account, and the adapter can be published with no
  usage restriction.
- The draft passes all 8 tokenizer checks (identical vocabulary, special
  tokens, chat template and ids).
- The smallest KV cache of the candidates (GQA with 2 KV heads) keeps
  cached-decoding memory low.
- A 32k context is far more than our notes need (to be confirmed with token
  statistics in TRIA-4).

## Rejected

- **Qwen2.5-3B:** rejected on memory, not licence. Its Qwen Research licence
  limits use to "research or evaluation purposes", which covers this
  assignment, but it would block any later reuse that Apache-2.0 allows. The
  deciding problem is hardware: ≈ 4.6 GB QLoRA leaves little room on a 6 GB
  card that Windows also uses, and speculative decoding with 3B + 0.5B in fp16
  would need ≈ 6.7 GB before any KV cache, more than the card has.
- **Llama-3.2-1B:** no smaller same-tokenizer model to draft with, so Task 5
  is impossible within the family. The gated download hurts reproducibility.
- **Llama-3.2-3B:** gated, 112 MB/1k-token KV cache, ≈ 6 GB in fp16. Too
  large for exact fp16 comparisons next to a draft on this GPU.

## Consequences

- Small models have limited clinical knowledge, so fine-tuning has to do the
  work. The before/after comparison (TRIA-6, TRIA-8) will show how much it does.
- The draft is 3× smaller than the target. That gap is narrow for
  speculative decoding, so a slowdown is possible and will be measured, not
  assumed (TRIA-11).

## How to revert

Change `target_model_id` / `draft_model_id` in `src/config.py`, re-run
`src/speculative/check_tokenizers.py`, and redo TRIA-4 onward.
