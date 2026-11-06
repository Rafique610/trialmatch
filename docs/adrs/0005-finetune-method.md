# ADR 0005: Fine-tune with QLoRA (4-bit NF4 base + LoRA adapters)

Date: 5 November 2026 · Status: proposed (TRIA-2, awaiting owner review)

## Context

Qwen2.5-1.5B-Instruct must be fine-tuned on the 37-patient training split
(ADR 0004) on a 6 GB RTX 3050 (ADR 0003). Literature: `docs/research/literature.md`
[S7, S8]. Memory figures: `docs/research/models.md`.

## Options (memory for Qwen2.5-1.5B, batch 1)

| Method | What trains | Memory | Verdict |
| --- | --- | --- | --- |
| Full fine-tune (fp16 weights + fp32 Adam) | all 1.54 B params | ≈ 16 B/param ≈ 25 GB before activations | does not fit |
| LoRA on fp16 base | 18.5 M adapter params (r=16) | 2.88 GB base + ≈ 0.3 GB optimiser + activations ≈ 4.5–5 GB | fits barely; no room for longer notes |
| **QLoRA** (NF4 base) | same 18.5 M | 1.04 GB base + ≈ 0.3 GB + activations ≈ 3.3 GB | **fits with headroom** |
| Prompt-only (no training) | nothing | inference only | fails the brief's fine-tuning requirement; kept as the baseline |

## Decision

QLoRA via `peft` + `bitsandbytes`:

- NF4 4-bit base with double quantisation, bf16 compute (the RTX 3050 is Ampere, which supports bf16).
- LoRA on q, k, v, o, gate, up, down projections. Rank, alpha and dropout set
  in the training config. At least 2 configurations compared on validation (TRIA-7).
- Loss on the answer tokens only. Prompt tokens are masked with -100, so the
  model learns to answer, not to copy the note.
- Gradient checkpointing, batch 1 with gradient accumulation.

## Consequences

- Training sees a 4-bit base, but the app and benchmarks run the adapter on
  the fp16 base, so the two can differ slightly. TRIA-8 evaluates the
  inference setup actually shipped, and the report states which one.
- The adapter is about 70 MB (18.5 M params in fp32), easy to host on the
  Hugging Face Hub.
- The memory figures here are estimates. TRIA-7 replaces them with measured
  peak memory.

## How to revert

Switch to fp16 LoRA by removing the quantisation config in the TRIA-7 training
config. Only the memory and speed results change.
