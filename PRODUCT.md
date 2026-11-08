# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Next.js + TypeScript in `web/`, managed with pnpm (project rule `docs/invariants/frontend.md`), talking to the FastAPI backend in `src/app/` (TRIA-12).

## Users

Primary (confirmed 8 Nov 2026): the **assignment evaluator**, an instructor in the demo and defence of GenAI Assignment 2. They paste an unseen patient note and criteria, need labels and evidence fast, then compare no-cache, KV-cache and speculative decoding live and download a report. Engineering panels are first-class, not hidden.
Secondary: a clinical research coordinator checking one patient against a trial protocol's criteria.

## Product Purpose

TrialMatch screens one patient note against clinical-trial criteria, one criterion at a time. For each criterion it gives a label from four (inclusion: included / not included / not enough information / not applicable; exclusion: excluded / not excluded / not enough information / not applicable) and the numbered note sentences that support it. A fine-tuned Qwen2.5-1.5B does the work. An explicit rule then aggregates the labels into a preliminary outcome. Success: the evaluator can check every answer against the note and trust the numbers on screen.

## Positioning

Every label is tied to the exact sentences the model cited, and every speed number is measured live on the evaluator's own input by decoders the team wrote: its own KV cache, tiled attention and speculative decoding.

## Operating Context

A laptop with an RTX 3050 6 GB. A demo video and a live defence. One request decodes at a time (≈ 0.7 s per criterion). Started with one command (`task app`).

## Capabilities and Constraints

- Endpoints: `/screen`, `/benchmark`, `/attention-bench`, `/report`, `/health`, plus validation examples.
- The overall outcome is a stated rule (ADR 0002), not a prediction. It must be shown as a rule, with the criteria that drove it.
- The explanation is built from cited sentences, never generated (ADR 0001).
- An unreadable model answer is shown as such and never guessed.
- Limits: note ≤ 20,000 characters, 1–30 criteria, prompt ≤ 3,000 tokens.

## Brand Commitments

- Name: TrialMatch.
- Persistent banner: "Educational screening prototype — not a clinical enrollment decision."

## Evidence on Hand

- Real validation notes and criteria in `data/processed/val.jsonl`.
- Measured results in `results/` (test accuracy 0.709 and macro-F1 0.566 vs 0.039 / 0.077 untrained; benchmarks in `results/bench/`).
- No users, testimonials or clinical validation exist, and none may be implied.

## Product Principles

1. Evidence before verdict: a label is only as good as the sentence it points to.
2. Real numbers only: everything on screen comes from the backend or a measurement.
3. Never collapse uncertainty: all four labels stay visible; "not enough information" is a first-class answer.
4. The rule is visible: the overall outcome always shows its rule and drivers.

## Accessibility & Inclusion

WCAG 2.1 AA contrast, full keyboard use, visible focus, and reduced-motion support. Label states are never conveyed by colour alone.
