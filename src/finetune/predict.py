"""Run a model (optionally with a LoRA adapter) over a split and write a scored run folder.

Reused for the baseline (TRIA-6), checkpoint selection (TRIA-7) and the final test (TRIA-8).
Decoding is greedy; batches are left-padded. Every setting is written to meta.json.
Run: uv run python -m src.finetune.predict --split val --run baseline_val [--adapter DIR] [--limit N]
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from src.data.prompt import messages, parse_output
from src.eval.run import evaluate, read_jsonl


def load(adapter: str | None):
    tok = AutoTokenizer.from_pretrained(settings.target_model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(settings.target_model_id, dtype=torch.float16, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


@torch.inference_mode()
def generate(tok, model, records: list[dict], batch_size: int, max_new_tokens: int) -> list[str]:
    outputs = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [tok.apply_chat_template(messages(r), add_generation_prompt=True, tokenize=False) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
            temperature=None, top_p=None, top_k=None,  # greedy: clear the sampling defaults in generation_config
        )
        outputs += tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--run", required=True, help="folder name under results/")
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder (omit for the base model)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    args = ap.parse_args()

    records = read_jsonl(settings.data_dir / "processed" / f"{args.split}.jsonl")[: args.limit]
    run_dir = settings.results_dir / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load(args.adapter)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    raw = generate(tok, model, records, args.batch_size, args.max_new_tokens)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    rows = []
    for r, text in zip(records, raw):
        parsed = parse_output(text, r["criterion_type"], len(r["sentences"]))
        rows.append({"annotation_id": r["annotation_id"], "raw_output": text, "label": parsed["label"],
                     "evidence_ids": parsed["evidence_ids"], "parse_error": parsed["parse_error"],
                     "dropped_ids": parsed["dropped_ids"]})
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")

    meta = {
        "split": args.split, "rows": len(records), "limit": args.limit,
        "model": settings.target_model_id, "adapter": args.adapter, "precision": "fp16",
        "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                     "padding": "left", "implementation": "transformers generate"},
        "prompt": "src/data/prompt.py", "seconds": round(seconds, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    m = evaluate(run_dir)
    for subset in ("overall", "inclusion", "exclusion"):
        lab, ev = m[subset]["labels"], m[subset]["evidence"]
        print(f"{subset:9} n={lab['n']:3} acc={lab['accuracy']} macroF1={lab['macro_f1']} parse_err={lab['parse_errors']} "
              f"evidence micro F1={ev['micro']['f1']} macro F1={ev['macro']['f1']}")
    print(f"{len(records)} rows in {seconds:.0f}s, peak GPU {meta['peak_gpu_memory_gb']} GB -> {run_dir}")


if __name__ == "__main__":
    main()
