# ADR 0003: Train and benchmark on the local RTX 3050

Date: 5 November 2026 · Status: accepted (project owner)

## Context

Hardware: NVIDIA RTX 3050 Laptop GPU (6 GB VRAM), 16 GB RAM, Windows 11. The
brief allows any free environment and asks for an explanation of the compute
and memory limits.

## Options

- Local RTX 3050: always available, same machine as the app and benchmarks;
  6 GB limits model size and sequence length.
- Kaggle T4 (16 GB, free quota): more memory; session limits, and results come
  from a different GPU than the app's.

## Decision

Train, evaluate and benchmark locally. To fit 6 GB: 4-bit base model (QLoRA),
batch size 1 with gradient accumulation, gradient checkpointing, and a max
sequence length chosen from the TRIA-4 token statistics.

Kaggle is an emergency fallback only, if training still runs out of memory.
The same script runs there unchanged.

## Consequences

- All reported runtimes and memory figures come from one GPU, so they are comparable.
- The report states the 6 GB limit and every memory-saving setting used.

## How to revert

Run `task train` on Kaggle with the same config. Record the GPU in the run
metadata and this ADR.
