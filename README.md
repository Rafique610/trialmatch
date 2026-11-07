# TrialMatch

Criterion-level clinical trial screening with a fine-tuned open-weight
Transformer. Given a patient note and one inclusion or exclusion criterion, the
model predicts the expert label (including *not enough information* and
*not applicable*) and cites the patient sentences that support it. The project
also implements and benchmarks KV-cached decoding, tiled online-softmax
attention, and greedy speculative decoding, and serves everything in a
browser app.

> **Educational screening prototype, not a clinical enrollment decision tool.**

GenAI Assignment 2 · dataset: [TrialGPT Criterion Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations)

## Status

| Part | Card | State |
| --- | --- | --- |
| Project setup | TRIA-1 | done |
| Research and model choice | TRIA-2 | done |
| Dataset and 37/8/8 split | TRIA-3 | done |
| Prompt format and token lengths | TRIA-4 | done |
| Evaluation metrics | TRIA-5 | done |
| Baseline, training, final test | TRIA-6 to 8 | not started |
| KV cache, tiled attention, speculative decoding | TRIA-9 to 11 | not started |
| App | TRIA-12, 13 | not started |
| Paper and submission | TRIA-14, 15 | not started |

## Requirements

- Windows/Linux with an NVIDIA GPU (developed on an RTX 3050 Laptop, 6 GB) and a driver supporting CUDA 12.8+
- [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev/)

## Setup

```bash
task setup   # uv sync: Python 3.11 + GPU PyTorch + libraries
task gpu     # should print: cuda True <your GPU name>
task test    # runs the test suite
```

`task --list` shows every command. Commands not built yet print which card adds them.

## Data and split

```bash
task data    # downloads the dataset (pinned revision), drops GPT-4 columns,
             # splits patients 37/8/8 with numpy default_rng(42)
```

| Split | Patients | Criterion rows |
| --- | --- | --- |
| train | 37 | 603 |
| val | 8 | 206 |
| test | 8 | 206 |

- Manifest with patient ids, label counts and file hashes: [results/split_manifest.json](results/split_manifest.json).
- Processed rows are written to `data/processed/*.jsonl` (not committed; `task data` rebuilds them identically).
- Leakage tests run with `task test` (`tests/test_split.py`).
- Column choices and label counts: [docs/domain/trialmatch/dataset.md](docs/domain/trialmatch/dataset.md).

### Prompt and answer

The model reads the numbered note, the criterion type and the criterion text,
and answers with JSON only:

```json
{"label": "not enough information", "evidence_ids": [0]}
```

`task data` also measures token lengths: the longest example is 636 tokens,
the maximum length is 1024, and **no note is truncated** (0 gold evidence
sentences lost). See [results/truncation_report.json](results/truncation_report.json)
and [docs/domain/trialmatch/prompt.md](docs/domain/trialmatch/prompt.md).

## Evaluation

```bash
task eval -- results/<run>   # needs meta.json + predictions.jsonl in that folder
```

Writes `metrics.json`, `per_record.jsonl` and three confusion-matrix PNGs
(overall, inclusion, exclusion) into the run folder. The metrics are:

- **labels:** accuracy, macro-F1, per-class precision/recall/F1, confusion matrices
- **evidence:** sentence-id micro and macro precision/recall/F1, with explicit
  rules for empty gold or predicted sets

Unparseable answers are counted as errors, never guessed. Conventions:
[docs/domain/trialmatch/eval.md](docs/domain/trialmatch/eval.md).

## Models

| Role | Model | Why |
| --- | --- | --- |
| Target (fine-tuned) | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache-2.0, ungated, 2.88 GB fp16 / 1.04 GB 4-bit, fits a 6 GB GPU |
| Draft (speculative decoding) | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Identical tokenizer to the target (verified) |

```bash
task research   # re-pulls model facts -> results/model_candidates.json
                # and checks tokenizers -> results/tokenizer_compat.json
```

Comparison and reasoning: [docs/research/models.md](docs/research/models.md),
[docs/research/literature.md](docs/research/literature.md),
[ADR 0004](docs/adrs/0004-model-choice.md), [ADR 0005](docs/adrs/0005-finetune-method.md).

## Layout

| Folder | What it holds |
| --- | --- |
| `src/config.py` | All settings (model ids, paths, split seed); override with `TM_*` env vars |
| `src/research/` | Model fact collection for the comparison |
| `src/data/` | Dataset loading, patient-grouped split, prompt format |
| `src/eval/` | Label and evidence metrics |
| `src/finetune/` | Baseline prediction and QLoRA training |
| `src/kvcache/` | Our KV-cached greedy decoder and no-cache reference |
| `src/attention/` | Reference and tiled online-softmax attention |
| `src/speculative/` | Greedy speculative decoding |
| `src/app/` | FastAPI backend |
| `web/` | Browser UI (Next.js) |
| `report/` | IEEE LaTeX paper |
| `results/` | Split manifest, metrics, benchmark logs |
| `docs/` | Rules, project facts, ADRs (start at `docs/docs/index.md`) |

## AI use

Built with help from Claude (Anthropic) as a coding assistant. All code is
reviewed, tested, and understood by the author. Full appendix in the paper.
