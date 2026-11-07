# Evaluation

Code: `src/eval/metrics.py` (maths), `src/eval/run.py` (files and plots).
Every model run (baseline, fine-tuned, val, test) is scored the same way:

```bash
task eval -- results/<run>
```

## Run folder contract

| File | Written by | Content |
| --- | --- | --- |
| `meta.json` | predictor (TRIA-6/8) | `split`, model id, adapter, decoding settings, date, hardware |
| `predictions.jsonl` | predictor | one row per annotation: `annotation_id`, `raw_output`, `label` (or null), `evidence_ids`, `parse_error`, `dropped_ids` |
| `metrics.json` | `task eval` | everything below, plus `meta` |
| `per_record.jsonl` | `task eval` | gold vs predicted label and evidence per row, `correct`, raw output (for error analysis) |
| `confusion_{overall,inclusion,exclusion}.png` | `task eval` | confusion matrices |

Gold always comes from `data/processed/<split>.jsonl`, never from the run
folder. A gold row without a prediction is an error, never a silent skip.

## Label metrics (reported overall, inclusion-only, exclusion-only)

- **Accuracy:** correct / all rows. Parse errors count as wrong.
- **Per-class precision / recall / F1 / support**, from scikit-learn.
- **Macro-F1 (headline):** mean F1 over the labels present in the gold data
  of that subset. A label with no gold rows (e.g. `excluded` on test, where
  it has 0 rows) is reported as **undefined** (`null`), not 0. That avoids
  punishing or rewarding the model for a class it was never asked about.
- **`macro_f1_all_labels`:** the same mean over every label of the subset,
  with absent labels scored 0. Reported for transparency; the paper states
  which one is quoted.
- **Confusion matrix:** rows are expert labels, columns are predictions. An
  extra `parse_error` column counts unusable answers.
- "Overall" pools both label sets (6 labels + parse_error). The inclusion
  and exclusion views use their own 4 labels.

## Evidence metrics (sentence-id sets)

Per row, G = expert sentence ids and P = predicted ids. Parse errors predict P = ∅.

- **Micro:** TP = Σ|G∩P|, FP = Σ|P−G|, FN = Σ|G−P| pooled over rows, then P/R/F1.
  Empty sets simply add nothing.
- **Macro:** per-row P/R/F1, averaged over the rows where each is defined:

| Case | Precision | Recall | F1 | Counted as |
| --- | --- | --- | --- | --- |
| G = ∅, P = ∅ | undefined | undefined | undefined | `both_empty`: a correct abstention, left out of the macro averages |
| G = ∅, P ≠ ∅ | 0 | undefined | 0 | `gold_empty_pred_nonempty`: the model cited sentences where experts cited none |
| G ≠ ∅, P = ∅ | undefined | 0 | 0 | `gold_nonempty_pred_empty`: the model missed all evidence |
| G ≠ ∅, P ≠ ∅ | normal | normal | normal | `both_nonempty` |

- **`macro_empty_as_correct`:** the alternative convention, where a
  both-empty row scores 1/1/1. Reported because 60 % of gold sets are empty,
  so this choice moves the numbers a lot.
- **`evidence_when_gold_nonempty`:** scores restricted to rows where experts
  cited evidence, to show retrieval quality on its own.
- An empty evidence set is never read as proof of eligibility or
  ineligibility. It only means no sentence was cited.

## Checks (`tests/test_metrics.py`)

Hand-worked examples for the label and evidence maths; gold-as-prediction on
val scores exactly 1.0 everywhere; a run with missing predictions is rejected.
