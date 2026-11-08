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
| Untrained baseline (val) | TRIA-6 | done |
| QLoRA training sweep (val) | TRIA-7 | done |
| Final test + error analysis | TRIA-8 | done |
| KV cache (own decoder, equivalence, benchmark) | TRIA-9 | done |
| Tiled online-softmax attention | TRIA-10 | done |
| Speculative decoding | TRIA-11 | done |
| App backend (FastAPI) | TRIA-12 | in review |
| App frontend | TRIA-13 | not started |
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

## Results so far (validation, 206 criteria)

| Model | Accuracy | Macro-F1 | Evidence micro-F1 |
| --- | --- | --- | --- |
| Qwen2.5-1.5B-Instruct, untrained (greedy, fp16) | 0.068 | 0.081 | 0.134 |
| + QLoRA adapter (run C, epoch 4; greedy, fp16) | 0.738 | 0.527 | 0.508 |

These runs used `transformers.generate`, which also applied Qwen's default repetition penalty
(1.1). Pure greedy gives the same labels; only the cited evidence shifts (fine-tuned 0.508 → 0.547).
See the decoding note in [finetune.md](docs/domain/trialmatch/finetune.md).

```bash
task train         # 4 QLoRA configs x 5 epochs (~4.5 h on an RTX 3050)
task train-report  # comparison table + curves from finished runs
```

The fine-tuned adapter is public on the Hugging Face Hub:
[Aduhere/trialmatch-qwen2.5-1.5b-qlora](https://huggingface.co/Aduhere/trialmatch-qwen2.5-1.5b-qlora).
To use it without training:

```bash
task adapter-download   # -> checkpoints/final/ (no login needed)
uv run python -m src.finetune.predict --split val --run finetuned_val --adapter checkpoints/final
```

The label-weighted run won on validation macro-F1. Run comparison and
analysis: [docs/domain/trialmatch/finetune.md](docs/domain/trialmatch/finetune.md).

```bash
task baseline   # re-runs the untrained model on val (~1 min on an RTX 3050)
```

The untrained model answers in valid JSON, but collapses to one label per
criterion type and reads exclusion backwards. See
[docs/domain/trialmatch/baseline.md](docs/domain/trialmatch/baseline.md).

## Final test (206 criteria, run once)

Settings were frozen in [results/test_protocol.md](results/test_protocol.md) before the run.
Decoding was greedy with Qwen's default repetition penalty 1.1, for both models (see the correction note in the protocol).

| Model | Accuracy | Macro-F1 | Incl macro-F1 | Excl macro-F1 | Evidence micro-F1 |
| --- | --- | --- | --- | --- | --- |
| Untrained | 0.039 | 0.077 | 0.099 | 0.000 | 0.157 |
| Fine-tuned | **0.709** | **0.566** | 0.541 | 0.487 | **0.579** |

The test split has no gold `excluded` rows, so that class drops out of macro-F1.
Full table: [results/test_comparison.md](results/test_comparison.md). Ten errors
traced to sentence ids: [results/error_analysis.md](results/error_analysis.md).

```bash
task final-test
```

## KV cache (TRIA-9)

Our own greedy decoder with an explicit KV cache: RoPE positions, causal mask with
cache offset, grouped-query attention. It matches full recomputation token for token
(64/32, 256/64, 1024/128). It decodes about 40 ms/token at any prompt length, against
58–875 ms without the cache, which is 20× faster at 2048 tokens. Details:
[docs/domain/trialmatch/kvcache.md](docs/domain/trialmatch/kvcache.md).

```bash
task bench       # equivalence + benchmark -> results/bench/kvcache.json, kvcache.png
task test-gpu    # token-equality tests on the real model (~2.5 min)
```

## Tiled attention (TRIA-10)

Attention in key/value blocks with a running max, sum and output (online softmax), in pure
PyTorch. It matches the full reference to 1.8e-6 (fp32) / 4.9e-4 (fp16) over a 144-config grid.
Swapped into all 28 layers of the model, it gives identical tokens. At 8,192 tokens it needs
26 MB of working memory where the reference needs 9.3 GB. It's slower than PyTorch's fused
kernels, because a Python loop can't keep tiles in on-chip memory. Details:
[docs/domain/trialmatch/attention.md](docs/domain/trialmatch/attention.md).

## Speculative decoding (TRIA-11)

Our own greedy draft-and-verify loop: the 0.5B model guesses γ tokens, the 1.5B model checks them
in one pass, and both caches roll back to what was accepted. Its output is identical to the 1.5B
model's own greedy output (all task prompts at every γ; the one free-text mismatch is an exact fp16
tie). On this laptop it is **slower** (0.59–0.97×) even though 74–91% of guesses are accepted. Our
runner makes a 0.5B token cost 85% of a 1.5B token, and the doc measures why. Details:
[docs/domain/trialmatch/speculative.md](docs/domain/trialmatch/speculative.md).

## App backend (TRIA-12)

```bash
task app        # API on http://localhost:8000 — open /docs to try every endpoint
```

- **`POST /screen`:** takes any patient note plus inclusion/exclusion criteria, with a choice of
  decoder (no cache, KV cache, speculative). It returns, per criterion, the label, the cited
  sentences, an explanation built from them, and a missing-info flag, plus a preliminary overall
  outcome from an explicit rule (not a clinical decision).
- **`POST /benchmark`:** runs the same input through all three decoders live.
- **`POST /attention-bench`:** a live run of the attention comparison.
- **`POST /report`:** downloads an HTML/Markdown report with provenance.

Details: [docs/domain/trialmatch/app.md](docs/domain/trialmatch/app.md).

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
