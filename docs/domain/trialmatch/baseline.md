# Baseline: untrained Qwen2.5-1.5B-Instruct on validation

Run: `task baseline` → `results/baseline_val/` (`meta.json`,
`predictions.jsonl`, `metrics.json`, `per_record.jsonl`, confusion PNGs).

## Settings (from `meta.json`)

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16, no adapter |
| Prompt | `src/data/prompt.py`, the same prompt fine-tuning uses |
| Decoding | greedy, `max_new_tokens=64`, batch 8, left padding, `transformers.generate`, **`repetition_penalty=1.1`** (Qwen's `generation_config` default, not overridden; found in TRIA-11). Pure greedy on the same split gives identical labels: `results/baseline_val_greedy/` |
| Split | val: 8 patients, 206 criteria |
| Time / memory | 206 rows in 50 s; peak GPU memory 3.21 GB (RTX 3050 6 GB) |

## Results

| Subset | n | Accuracy | Macro-F1 | Parse errors | Evidence micro-F1 | Evidence macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 206 | 0.068 | 0.081 | 0 | 0.134 | 0.100 |
| inclusion | 66 | 0.136 | 0.102 | 0 | 0.177 | 0.090 |
| exclusion | 140 | 0.036 | 0.022 | 0 | 0.122 | 0.104 |

## What goes wrong

| | Gold (expert) | Predicted |
| --- | --- | --- |
| inclusion | included 35 · not enough information 27 · not included 3 · not applicable 1 | **not applicable 58** · included 8 |
| exclusion | not excluded 101 · not enough information 26 · not applicable 11 · excluded 2 | **excluded 131** · not applicable 5 · not excluded 4 |

1. **The format is learned, the task is not.** All 206 answers are valid
   JSON, each wrapped in a markdown code fence, which the parser tolerates.
2. **Label collapse.** The model gives nearly one label per criterion type,
   whatever the note says.
3. **Exclusion polarity is inverted.** It answers `excluded` for 131 of 140
   rows, even though the label meanings are spelled out in the prompt. Most of
   those rows are expert `not excluded`. A small model doesn't reliably
   handle "the patient has the excluded condition" versus "the criterion is
   about exclusion".
4. **Over-citing evidence.** On average it cites 1.8 sentences. Sometimes it
   cites the whole note (`[0…8]`) for rows where the experts cited none.

These are the "before" numbers for the before/after comparison. They are not
tuned. The prompt was fixed in TRIA-4 and not changed after seeing these results.
