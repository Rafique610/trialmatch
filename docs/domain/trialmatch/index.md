# TrialMatch: project facts

GenAI Assignment 2. Given a patient note and one trial criterion (inclusion or
exclusion), predict the expert label and cite the patient sentences that
support it. Brief: `GenAI_Assignment#2.pdf`. **Deadline: 10 November 2026**
(2-hour grace, then zero).

## Board

Synverk project **TrialMatch**, key `TRIA`. Every card links a flow:

| Flow | Cards |
| --- | --- |
| TrialMatch · the whole pipeline | TRIA-1, 2, 14, 15 |
| Data · from raw dataset to training examples | TRIA-3, 4 |
| Train + evaluate · before and after fine-tuning | TRIA-5 to 8 |
| KV cache · decode without recomputing the past | TRIA-9 |
| Attention · reference vs tiled online softmax | TRIA-10 |
| Speculative decoding · draft guesses, target checks | TRIA-11 |
| App · one screening request, end to end | TRIA-12, 13 |

## Stack

| Thing | Choice |
| --- | --- |
| Python | 3.11 via `uv` (`pyproject.toml` + `uv.lock`) |
| PyTorch | CUDA 12.8 build from the PyTorch index |
| Models | Target `Qwen/Qwen2.5-1.5B-Instruct`, draft `Qwen/Qwen2.5-0.5B-Instruct` (confirmed in TRIA-2) |
| Fine-tuning | QLoRA (4-bit base + LoRA adapters) via `peft` + `bitsandbytes` |
| Backend | FastAPI in `src/app/` |
| Frontend | Next.js + pnpm in `web/` |
| Paper | IEEE LaTeX in `report/`, compiled in Overleaf |
| Commands | `Taskfile.yml` (`task --list`) |

## Hardware

NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM; 16 GB system RAM; Windows 11. All
training and benchmarks run here (ADR 0003).

## Folders

| Folder | Feature | Card |
| --- | --- | --- |
| `src/research/` | model facts for the comparison | TRIA-2 |
| `src/data/` | load, clean, split, prompt | TRIA-3, 4 |
| `src/eval/` | metrics | TRIA-5 |
| `src/finetune/` | predict + train | TRIA-6, 7 |
| `src/kvcache/` | cache-aware greedy decoder | TRIA-9 |
| `src/attention/` | reference + tiled attention | TRIA-10 |
| `src/speculative/` | speculative decoding | TRIA-11 |
| `src/app/` | FastAPI backend | TRIA-12 |
| `web/` | browser UI | TRIA-13 |
| `report/` | IEEE paper | TRIA-14 |
| `results/` | manifests, metrics, benchmark logs (committed) | all |
| `data/`, `checkpoints/` | downloaded and produced files (git-ignored) | all |

## Decisions

- ADR 0001: training target is label + evidence ids; explanation built from cited sentences.
- ADR 0002: overall patient outcome is a stated rule over criterion labels.
- ADR 0003: train and benchmark on the local RTX 3050.
- ADR 0004: target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct (tokenizers verified identical).
- ADR 0005: QLoRA (NF4 base + LoRA r≈16 on all projections), loss on answer tokens only.

## Research

`docs/research/literature.md` (13 sources, S1–S13) and `docs/research/models.md`
(the comparison table). Raw facts are in `results/model_candidates.json` and
`results/tokenizer_compat.json`; regenerate them with `task research`.

## Data

1,015 criterion rows from 53 patients, split 37/8/8 (603/206/206 rows). See
`dataset.md` for the columns kept and dropped, label counts and known
consequences. Built with `task data`. Processed files go to
`data/processed/{train,val,test}.jsonl` (git-ignored); the manifest is
`results/split_manifest.json` (committed).

Record shape: `annotation_id, patient_id, trial_id, trial_title,
criterion_type, criterion_text, sentences[], label, evidence_ids[]`.

## Prompt

The input/answer contract, parser and token lengths are in `prompt.md`.
Answer = `{"label", "evidence_ids"}`; `MAX_LEN = 1024` (longest example 636
tokens, nothing truncated).

## Phase status

| Card | Status |
| --- | --- |
| TRIA-1 Setup | done |
| TRIA-2 Research + model choice | done |
| TRIA-3 Data + split | done |
| TRIA-4 Prompt + lengths | done |
| TRIA-5 Evaluation metrics | done |
| TRIA-6 Baseline on val | done |
| TRIA-7 Fine-tune sweep | done |
| TRIA-8 Final test | done |
| TRIA-9 KV cache | done |
| TRIA-10 Tiled attention | done |
| TRIA-11 Speculative decoding | done |
| TRIA-12 App backend | done |
| TRIA-13 App frontend | done |
| TRIA-14 Paper | done |
| TRIA-15 Submission | in review |

## Baseline

`task baseline` → `results/baseline_val/`. The untrained model scores val
accuracy 0.068 and macro-F1 0.081: correct JSON, but label collapse and
inverted exclusion polarity. Details in `baseline.md`. The predictor
`src/finetune/predict.py` (`--split`, `--run`, `--adapter`, `--limit`) is
reused by TRIA-7 and TRIA-8.

## Fine-tuning

`task train` runs the 4-config QLoRA sweep; `task train-report` compares
the runs. The chosen adapter is `checkpoints/C_r16_weighted/epoch_4`
(label-weighted, r16, lr 2e-4), picked by val macro-F1 and never by test.
With fp16 + the adapter it scores val acc 0.738, macro-F1 0.527, evidence
micro-F1 0.508 (`results/finetuned_val/`). Published as
`Aduhere/trialmatch-qwen2.5-1.5b-qlora` (`task adapter-download` → `checkpoints/final/`).
Details in `finetune.md`.

## Final test

Run once (`task final-test`) under the frozen protocol `results/test_protocol.md`.
Test: base acc 0.039 / macro-F1 0.077 → fine-tuned acc 0.709 / macro-F1 0.566,
evidence micro-F1 0.157 → 0.579, 0 parse errors. Table: `results/test_comparison.md`
(built by `src/eval/compare.py`). Error analysis: `results/error_analysis.md` (10 cases).
The largest error source is the "not mentioned" convention, which the gold labels apply
inconsistently.

## KV cache

`src/kvcache/`: our own forward over the Qwen2 layers, with `KVCache` (preallocated, `crop` for
rollback) and a pluggable `attn_fn` (TRIA-10 plugs in here). `greedy_cached` equals full
recompute token for token. Benchmark: `results/bench/kvcache.json`. GPU tests are marked `gpu`
(`task test-gpu`), and `task test` skips them. Details in `kvcache.md`.

## Attention

`src/attention/reference.py` (the single reference, also used by the KV runner) and
`src/attention/tiled.py` (online softmax, causal block skip). Use `attn_fn=tiled` in `Runner.forward`.
Benchmark: `results/bench/attention.json`. Flash SDPA is not in the Windows PyTorch build. Details in `attention.md`.

## Speculative decoding

`src/speculative/decode.py` (`speculative_greedy(target, draft, ids, n, gamma, eos)`) on two `Runner`s.
The draft weights live in `models/Qwen2.5-0.5B-Instruct` (git-ignored; the C: drive was full) and are found
by `settings.model_source`. Qwen's `generation_config` has `repetition_penalty 1.1`: pass
`repetition_penalty=1.0` to `generate` when pure greedy is meant. Details in `speculative.md`.

## App backend

`src/app/` (FastAPI): `main.create_app(engine=None)` (tests inject a stand-in engine),
`engine.Engine` (merged adapter, lazy draft, GPU lock), `screening` (split / prompt / parse /
explain / ADR 0002 rule), `bench`, `report`. `task app` → :8000/docs. CORS allows the Vite dev
server on :5173 (TRIA-13). Details in `app.md`.

## App frontend

`web/` (Next.js 16, pnpm): `app/page.tsx` (Screen) and `app/benchmarks/page.tsx`. Shared state lives in
`lib/workbench.tsx` (one request per criterion, then `/aggregate`); the typed client is `lib/api.ts`.
Components are in `components/workbench/*` and `components/bench/*`. Styles: `app/globals.css` (tokens) and
`app/form.css`. The design system is in root `DESIGN.md`. Capture states with `scripts/ui_screens.py`.

## Paper

`report/` (IEEEtran) is built by `task paper`. `report/make_tables.py` turns `results/` into `report/tables/*.tex`, `numbers.tex` (the numbers used in the prose) and the figures. Tectonic lives in `.tools/` (git-ignored). Checklist and owner checks: `paper.md`.

## Submission

The README's *Submission* section lists every link. The fresh-clone path (clone → `uv sync` → `task data` → `task adapter-download` → `task app`) was verified on 9 Nov 2026: identical split manifest, 178 tests pass, the app answers. The demo script is in `docs/demo_script.md`, defence notes are in `docs/defense.md`, and the paper's video link is the `\VideoURL` macro in `report/main.tex`.

## Evaluation

`task eval -- results/<run>` scores a run folder (`meta.json` +
`predictions.jsonl`). The metric conventions, especially for empty evidence
sets and absent classes, are in `eval.md`.
