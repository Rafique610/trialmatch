"""Upload the chosen LoRA adapter to the Hugging Face Hub as a public model repo, with a model card.
Needs `hf auth login` first (the token never goes through this script).
Run: uv run python -m scripts.upload_adapter <user>/<repo> [--adapter checkpoints/C_r16_weighted/epoch_4]
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from src.config import settings

CHOSEN = "checkpoints/C_r16_weighted/epoch_4"


def card(repo_id: str, adapter: Path) -> str:
    m = json.loads((settings.results_dir / "finetuned_val" / "metrics.json").read_text(encoding="utf-8"))
    run = json.loads((settings.results_dir / "train_C_r16_weighted" / "summary.json").read_text(encoding="utf-8"))["config"]
    o, ev = m["overall"]["labels"], m["overall"]["evidence"]["micro"]
    return f"""---
base_model: {settings.target_model_id}
library_name: peft
license: apache-2.0
tags: [lora, qlora, clinical-trials, trialgpt]
---

# TrialMatch: Qwen2.5-1.5B-Instruct QLoRA adapter

A LoRA adapter for `{settings.target_model_id}` that labels one patient note against one
clinical-trial criterion. It returns JSON `{{"label": ..., "evidence_ids": [...]}}`, where
`evidence_ids` are the numbered note sentences that support the label. It was trained on the
expert annotations of the TrialGPT Criterion Annotations dataset (revision
`{settings.dataset_revision[:12]}`), with a 37/8/8 patient-grouped split and seed 42.

Code, prompt format and evaluation: https://github.com/Rafique610/trialmatch

**Research and coursework use only. Not for clinical decisions.**

## Training

QLoRA (NF4 4-bit base, bf16 compute). LoRA r={run['rank']}, alpha={run['alpha']},
dropout={run['dropout']}, on q,k,v,o,gate,up,down. lr {run['lr']}, gradient accumulation
{run['grad_accum']}, inverse-sqrt label weighting. Chosen at epoch {adapter.name.split('_')[-1]} of
{run['epochs']} by validation macro-F1 (the test split was not used for selection). Hardware: one RTX 3050 6 GB.

## Validation results (206 criteria, fp16 base + adapter, greedy)

| Metric | Value |
| --- | --- |
| Accuracy | {o['accuracy']} |
| Macro-F1 | {o['macro_f1']} |
| Evidence micro-F1 | {ev['f1']} |
| Parse errors | {o['parse_errors']} |

The untrained base model scores accuracy 0.068 and macro-F1 0.081 on the same split.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("{settings.target_model_id}", dtype="float16", device_map="cuda")
model = PeftModel.from_pretrained(base, "{repo_id}")
```

Use the prompt built by `src/data/prompt.py` in the repo; the adapter expects that exact format.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--adapter", default=CHOSEN)
    args = ap.parse_args()
    adapter = Path(args.adapter)
    (adapter / "README.md").write_text(card(args.repo_id, adapter), encoding="utf-8")
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=adapter, commit_message=f"Adapter from {adapter.as_posix()}")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
