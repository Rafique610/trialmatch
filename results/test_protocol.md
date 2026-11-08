# Test protocol (frozen before the test split is run)

Written 8 November 2026, at code commit `c205d7c`, before any model output on
`data/processed/test.jsonl` existed. The test split was not used for any
choice made before this point: prompt, checkpoint, decoding and metrics were
all fixed on train and validation (TRIA-4 to TRIA-7).

## Data

- `data/processed/test.jsonl`: 8 patients, 206 criteria (75 inclusion, 131 exclusion).
  *(Corrected after the run: the first draft copied the validation counts, 66/140, by mistake. No frozen setting changed.)*
  sha256 `fc79450838dd79a18b9b455e512944b314fbbf8e8f634dd0630afa241937309f`.
- Split rule and patient ids: `results/split_manifest.json` (seed 42, 37/8/8 by patient).
- Known before the run (from the split counts in TRIA-3): the test split has **0**
  gold `excluded` rows. Its per-class F1 is therefore null, and it is left out of
  the exclusion macro-F1 (rule in `docs/domain/trialmatch/eval.md`).

## Models (both run once, same rows, same decoding)

| Run folder | Model |
| --- | --- |
| `results/baseline_test/` | `Qwen/Qwen2.5-1.5B-Instruct`, no adapter |
| `results/finetuned_test/` | same base + adapter `checkpoints/final` = `Aduhere/trialmatch-qwen2.5-1.5b-qlora` (run C_r16_weighted, epoch 4), sha256 `6bb84be733cdefae53f7bcdfaf599736f65cd2215ec7466e2d7076450288ccc5` |

The checkpoint was chosen by validation macro-F1 (`results/finetune_runs.md`).

## Inference (identical for both)

- Prompt: `src/data/prompt.py` (system message + numbered sentences + criterion), chat template, generation prompt.
- Precision fp16, greedy decoding, `max_new_tokens=64`, batch 8, left padding, `transformers.generate`.
  *(Correction, 8 Nov 2026, after the run: `generate` also applied `repetition_penalty=1.1` from Qwen's
  `generation_config.json`, which the call did not override. Both test runs used it, so the comparison is like for like.
  The test was **not** re-run. On validation, pure greedy (`--repetition-penalty 1.0`) gives **identical labels on all 206 rows** for both models; only cited evidence changes (fine-tuned evidence micro-F1 0.508 → 0.547, base 0.134 → 0.091). See `finetune.md` → *Decoding note*.)*
- Answer parsing: `parse_output` in `src/data/prompt.py`. Unparseable output counts as a `parse_error`, never guessed.

## Metrics

From `src/eval`, unchanged since TRIA-5: accuracy, macro-F1 (gold-present labels),
per-class P/R/F1 and confusion matrices, overall plus inclusion and exclusion;
evidence micro/macro P/R/F1 with the empty-set rules in `eval.md`.

## Commands

```bash
uv run python -m src.finetune.predict --split test --run baseline_test
uv run python -m src.finetune.predict --split test --run finetuned_test --adapter checkpoints/final
```

Each is run once. Whatever comes out is reported, and nothing is re-tuned afterwards.
