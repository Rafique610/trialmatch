# Dataset: TrialGPT Criterion Annotations

`ncbi/TrialGPT-Criterion-Annotations`, pinned to revision
`1cfcafde94a1560a33b4addc1638664fe28fc059` (`src/config.py`). One split
(`train`) of **1,015 rows from 53 patients**. Each row pairs one patient with
one trial criterion. Patients appear in 5 to 43 rows.

## Columns

| Column | Kept? | Why |
| --- | --- | --- |
| `annotation_id`, `patient_id`, `trial_id`, `trial_title` | kept | identifiers and trial context |
| `criterion_type` (`inclusion` / `exclusion`) | kept | model input |
| `criterion_text` | kept | model input |
| `note` | kept, parsed to `sentences` | model input. Already numbered `0. … \n1. …`; we parse those numbers, never re-split |
| `expert_eligibility` | kept as `label` | gold answer |
| `expert_sentences` (JSON list) | kept as `evidence_ids` | gold evidence |
| `gpt4_eligibility`, `gpt4_sentences`, `gpt4_explanation` | **dropped** | GPT-4 predictions; the brief bans them as inputs and from evaluation |
| `explanation_correctness` | **dropped** | an expert grade of GPT-4's explanation, still about GPT-4's answer |
| `training` | **dropped** | the original authors' split flag; the brief prescribes our own split |

Dropping happens in `src/data/load.py` through an allow-list (`KEEP`). A
test asserts that no dropped or `gpt4*` key survives.

## Labels (whole dataset)

| Inclusion | n | Exclusion | n |
| --- | --- | --- | --- |
| included | 150 | excluded | 10 |
| not included | 23 | not excluded | 484 |
| not enough information | 174 | not enough information | 119 |
| not applicable | 9 | not applicable | 46 |

610 of 1,015 rows (60 %) have **no** expert evidence (`[]`).

## Split (`results/split_manifest.json`)

Sorted unique patient ids, `numpy.random.default_rng(42).shuffle`, first 37
train, next 8 val, last 8 test.

| Split | Patients | Annotations | Empty evidence |
| --- | --- | --- | --- |
| train | 37 | 603 | 349 |
| val | 8 | 206 | 116 |
| test | 8 | 206 | 145 |

Per-split label counts, per-patient counts, annotation ids and file hashes
are in the manifest.

## Known consequences

- **The test split has no `excluded` row** (train 8, val 2). Per-class
  scores for `excluded` are undefined on test. The report must say so rather
  than print 0 or NaN without comment.
- The rare classes (`not included` 13 train / 3 val / 7 test; `excluded`
  8 / 2 / 0; `not applicable` small) make macro-F1 volatile with 8 patients
  per split.
- With 60 % empty gold evidence, the empty-set scoring rule (TRIA-5) strongly
  affects evidence metrics.
