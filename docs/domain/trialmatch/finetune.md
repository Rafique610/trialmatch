# Fine-tuning (TRIA-7)

Method decided in ADR 0005 (QLoRA). Code: `src/finetune/train.py` (one run)
and `src/finetune/sweep.py` (queue + comparison). Configs: `configs/*.json`.

## Mechanism

1. **Base model frozen in 4-bit.** `Qwen/Qwen2.5-1.5B-Instruct` loaded with
   bitsandbytes NF4, double quantisation, bf16 compute.
   `prepare_model_for_kbit_training` turns on gradient checkpointing
   (non-reentrant) and keeps layer norms in fp32.
2. **LoRA adapters** on `q, k, v, o, gate, up, down` projections. Only these train.
3. **Examples.** Chat-template prompt (`src/data/prompt.py`) followed by the
   gold JSON answer and `<|im_end|>`. Prompt tokens get label `-100`, so the
   loss is cross-entropy on the answer tokens only. Max length 1024 (nothing
   truncated, see `prompt.md`).
4. **Loop.** Plain PyTorch, batch 1, gradient accumulation 8, AdamW
   (no weight decay), gradient clipping at 1.0, 5 % linear warm-up then
   cosine decay, training order shuffled each epoch with seed 42.
5. **Optional label weighting** (run C). Each example's loss is multiplied by
   the inverse square root of its (criterion type, label) frequency in the
   training split, scaled to mean 1 and capped at 5. Rare labels (`excluded`
   8 rows, `not included` 13) count more.
6. **Every epoch:** teacher-forced validation loss; save the adapter to
   `checkpoints/<run>/epoch_<k>/`; greedy-decode the validation split with the
   in-memory 4-bit model (repetition penalty 1.1, see *Decoding note*) and score it with `src/eval` →
   `results/train_<run>/epoch_<k>/`.
7. **Selection.** Best run and epoch by validation macro-F1. Test is never
   used. The chosen adapter is then re-scored on validation with the shipped
   inference setup (fp16 base + adapter, `src/finetune/predict.py`) before TRIA-8.

## Runs (5 epochs each)

| Run | Rank / alpha | LR | Label weighting | Question it answers |
| --- | --- | --- | --- | --- |
| A_r16 | 16 / 32 | 2e-4 | no | default QLoRA setting |
| B_r8 | 8 / 16 | 2e-4 | no | are smaller adapters enough? |
| C_r16_weighted | 16 / 32 | 2e-4 | yes | does up-weighting rare labels help macro-F1? |
| D_r16_lr5e-5 | 16 / 32 | 5e-5 | no | does a gentler learning rate overfit less? |

All runs use dropout 0.05, gradient accumulation 8 and seed 42.

## Measured during the smoke test (16 rows, 1 epoch)

About 0.9 s per training example; peak GPU memory 2.85 GB in training and
3.37 GB for fp16 + adapter inference (ADR 0005 estimated ≈ 3.3 GB for
training). The adapter saved from 4-bit training loads into the fp16 base
and gave the same labels on the 8 smoke rows.

## Outputs

- `results/train_<run>/log.jsonl`: loss and learning rate per optimiser step (training curve)
- `results/train_<run>/summary.json`: per-epoch validation loss, accuracy, macro-F1, evidence F1, time, peak memory
- `results/train_<run>/epoch_<k>/`: validation predictions and metrics per epoch
- `results/finetune_runs.md` + `results/finetune_val_curves.png`: comparison across runs (`task train-report`)
- `results/finetune_queue.log`: start, end and exit code of each run
- `checkpoints/`: adapters (git-ignored). On 10 Nov 2026 the 19 epoch checkpoints other than the chosen one (`C_r16_weighted/epoch_4`, identical to `final/`) were deleted to free disk. Per-epoch metrics stay in `results/train_*/`; `task train` regenerates any checkpoint (about 1 h per config). The chosen one is public at
  https://huggingface.co/Aduhere/trialmatch-qwen2.5-1.5b-qlora (`task adapter-upload`,
  `scripts/upload_adapter.py`, which writes the model card). `task adapter-download` fetches it into
  `checkpoints/final/` without a login (`settings.adapter_repo_id`).

## Results

Sweep ran 7 Nov 22:58 → 8 Nov 03:17 (4 h 19 min, all exit 0). The first
epoch of A took 20 min while the laptop was in use; later epochs took
8.5–9.5 min. Full table: `results/finetune_runs.md`; curves:
`results/finetune_val_curves.png`.

Best epoch of each run (validation, 4-bit in-training scoring):

| Run | Best epoch | Acc | Macro-F1 | Incl F1 | Excl F1 | Evidence micro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| A_r16 | 5 | 0.738 | 0.442 | 0.362 | 0.445 | 0.548 |
| B_r8 | 5 | 0.743 | 0.443 | 0.375 | 0.427 | 0.555 |
| **C_r16_weighted** | **4** | **0.757** | **0.572** | 0.391 | **0.622** | 0.517 |
| D_r16_lr5e-5 | 5 | 0.757 | 0.528 | 0.367 | 0.560 | 0.575 |

What the sweep shows:

- **Label weighting gives the biggest gain** (C vs A: macro-F1 +0.13),
  almost all of it on exclusion, where the rare `excluded` and
  `not applicable` labels start being predicted.
- **Rank 8 is as good as rank 16** (B ≈ A), so the adapter size is not the
  bottleneck.
- **The gentler LR (D) also helps macro-F1** and has the lowest validation
  loss. With 2e-4, val loss bottoms at epoch 2 and then rises while
  macro-F1 keeps improving. The model grows more confident on the frequent
  labels and still learns the rare ones, so val loss is not the selection
  metric here.
- Parse errors are 0 everywhere except D epochs 4–5 (1 row).

**Chosen adapter: `checkpoints/C_r16_weighted/epoch_4`** (best val macro-F1).

### Re-scored with the shipped setup (fp16 base + adapter)

`results/finetuned_val/`:

| Model (val, 206 criteria) | Acc | Macro-F1 | Incl F1 | Excl F1 | Evidence micro-F1 | Parse err |
| --- | --- | --- | --- | --- | --- | --- |
| Untrained, fp16 | 0.068 | 0.081 | – | – | 0.134 | 0 |
| C epoch 4, 4-bit (in training) | 0.757 | 0.572 | 0.391 | 0.622 | 0.517 | 0 |
| C epoch 4, fp16 + adapter | 0.738 | 0.527 | 0.365 | 0.574 | 0.508 | 0 |

The fp16 and 4-bit runs disagree on 20 of 206 labels. The macro-F1 gap
comes mostly from exclusion `excluded`, which has only **2** gold rows in
val. fp16 predicts it 6 times (recall 1.0, precision 0.33); 4-bit predicts
it once (precision 1.0, recall 0.5). With so few rows, one prediction moves
that class's F1 by about 0.2, so the gap is within noise for a split this
small. fp16 inference: 97 s for 206 rows, peak 3.48 GB.

## Decoding note: repetition penalty

Found in TRIA-11: Qwen's `generation_config.json` sets `repetition_penalty: 1.1`, and
`transformers.generate` applies it unless the call overrides it. Ours didn't, so **every
`generate`-based evaluation (TRIA-6 baseline, TRIA-7 epoch scoring and re-score, TRIA-8 test)
was greedy with a 1.1 repetition penalty**, not pure greedy as first written. Base and fine-tuned
models always got the same setting, so every comparison is like for like. `predict.py` now takes
`--repetition-penalty` (default 1.1, so the recorded runs reproduce) and writes it into `meta.json`.

Measured on validation only (the test split was not reopened):

| Model (val, 206) | Penalty | Acc | Macro-F1 | Evidence micro P / R / F1 | Ids cited |
| --- | --- | --- | --- | --- | --- |
| Untrained | 1.1 (`baseline_val`) | 0.068 | 0.081 | 0.088 / 0.278 / 0.134 | 363 |
| Untrained | 1.0 (`baseline_val_greedy`) | 0.068 | 0.081 | 0.053 / 0.339 / 0.091 | 739 |
| Fine-tuned | 1.1 (`finetuned_val`) | 0.738 | 0.527 | 0.444 / 0.591 / 0.508 | 153 |
| Fine-tuned | 1.0 (`finetuned_val_greedy`) | 0.738 | 0.527 | 0.497 / 0.609 / 0.547 | 141 |

- **No label changes** (0/206 rows for either model), so accuracy, macro-F1 and the checkpoint
  choice are unaffected.
- **Only the evidence lists change** (46 rows fine-tuned, 115 untrained). The penalty discourages
  tokens that already appear in the prompt, which includes the sentence numbers. For the fine-tuned
  model, pure greedy cites a little less and more precisely (+0.04 F1). The untrained model, without
  the penalty, cites twice as many sentences (−0.04 F1).
- The app and our own decoders (TRIA-9/11) use pure greedy, so their evidence matches the
  `*_greedy` rows. The test numbers remain the penalty-1.1 results, reported as such.
