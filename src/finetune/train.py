"""QLoRA fine-tuning with a plain PyTorch loop (ADR 0005). One config file = one run.

Per epoch: train on the train split, teacher-forced val loss, save the adapter,
then greedy-decode val with the in-memory 4-bit model and score it (TRIA-5).
Everything lands in results/train_<name>/ and checkpoints/<name>/epoch_<k>/.
Run: uv run python -m src.finetune.train configs/<name>.json [--smoke]
"""

import argparse
import json
import math
import random
import time
from collections import Counter

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import settings
from src.data.lengths import MAX_LEN
from src.data.prompt import messages, parse_output, target
from src.eval.run import evaluate, read_jsonl
from src.finetune.predict import generate

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, rec: dict) -> dict:
    """Prompt tokens are masked with -100 so the loss covers only the JSON answer + end token."""
    prompt = tok(tok.apply_chat_template(messages(rec), add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
    answer = tok(target(rec) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (prompt + answer)[:MAX_LEN]
    return {"input_ids": ids, "labels": ([-100] * len(prompt) + answer)[:MAX_LEN]}


def label_weights(train: list[dict]) -> dict:
    """Inverse-sqrt frequency per (criterion type, label), scaled to mean 1 over the training rows, capped at 5."""
    counts = Counter((r["criterion_type"], r["label"]) for r in train)
    per_type = Counter(r["criterion_type"] for r in train)
    raw = {k: math.sqrt(per_type[k[0]] / counts[k]) for k in counts}
    mean = sum(raw[(r["criterion_type"], r["label"])] for r in train) / len(train)
    return {k: min(v / mean, 5.0) for k, v in raw.items()}


def load_model(cfg: dict):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(settings.target_model_id, quantization_config=bnb,
                                                 dtype=torch.bfloat16, device_map="cuda")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                      target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    return get_peft_model(model, lora)


@torch.no_grad()
def val_loss(model, examples: list[dict]) -> float:
    model.eval()
    total = tokens = 0
    for ex in examples:
        ids = torch.tensor([ex["input_ids"]], device="cuda")
        labels = torch.tensor([ex["labels"]], device="cuda")
        n = int((labels[:, 1:] != -100).sum())
        total += model(input_ids=ids, labels=labels).loss.item() * n
        tokens += n
    return total / tokens


def score_epoch(model, gen_tok, val: list[dict], run_dir, epoch: int, cfg: dict) -> dict:
    model.eval()
    model.config.use_cache = True
    raw = generate(gen_tok, model, val, batch_size=8, max_new_tokens=64)
    model.config.use_cache = False
    out = run_dir / f"epoch_{epoch}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, text in zip(val, raw):
        p = parse_output(text, r["criterion_type"], len(r["sentences"]))
        rows.append({"annotation_id": r["annotation_id"], "raw_output": text, "label": p["label"],
                     "evidence_ids": p["evidence_ids"], "parse_error": p["parse_error"], "dropped_ids": p["dropped_ids"]})
    (out / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    meta = {"split": "val", "limit": len(val) if cfg.get("smoke") else None, "model": settings.target_model_id,
            "adapter": f"checkpoints/{cfg['name']}/epoch_{epoch}", "precision": "nf4 (in-training)",
            "decoding": {"strategy": "greedy", "max_new_tokens": 64, "batch_size": 8, "padding": "left"}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return evaluate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="16 train rows, 8 val rows, 1 epoch")
    args = ap.parse_args()
    cfg = json.loads(open(args.config, encoding="utf-8-sig").read())
    if args.smoke:
        cfg |= {"name": cfg["name"] + "_smoke", "epochs": 1, "smoke": True}

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train = read_jsonl(settings.data_dir / "processed" / "train.jsonl")
    val = read_jsonl(settings.data_dir / "processed" / "val.jsonl")
    if args.smoke:
        train, val = train[:16], val[:8]

    tok = AutoTokenizer.from_pretrained(settings.target_model_id)
    gen_tok = AutoTokenizer.from_pretrained(settings.target_model_id, padding_side="left")
    weights = label_weights(train) if cfg["class_weighting"] else None
    train_ex = [encode(tok, r) | {"weight": weights[(r["criterion_type"], r["label"])] if weights else 1.0} for r in train]
    val_ex = [encode(tok, r) for r in val]

    model = load_model(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    steps_total = math.ceil(len(train_ex) / cfg["grad_accum"]) * cfg["epochs"]
    warmup = max(1, int(0.05 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))

    run_dir = settings.results_dir / f"train_{cfg['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "log.jsonl").open("w", encoding="utf-8")
    summary = {"config": cfg, "trainable_params": trainable, "train_rows": len(train_ex), "epochs": []}
    torch.cuda.reset_peak_memory_stats()
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        model.config.use_cache = False
        order = list(range(len(train_ex)))
        random.shuffle(order)
        t0, running = time.perf_counter(), []
        for i, idx in enumerate(order, 1):
            ex = train_ex[idx]
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            labels = torch.tensor([ex["labels"]], device="cuda")
            loss = model(input_ids=ids, labels=labels).loss
            (loss * ex["weight"] / cfg["grad_accum"]).backward()
            running.append(loss.item())
            if i % cfg["grad_accum"] == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                log.write(json.dumps({"epoch": epoch, "step": step, "loss": sum(running) / len(running), "lr": sched.get_last_lr()[0]}) + "\n")
                log.flush()
                running = []
        train_seconds = time.perf_counter() - t0
        vloss = val_loss(model, val_ex)
        ckpt = settings.checkpoints_dir / cfg["name"] / f"epoch_{epoch}"
        model.save_pretrained(ckpt)
        m = score_epoch(model, gen_tok, val, run_dir, epoch, cfg)
        row = {"epoch": epoch, "train_seconds": round(train_seconds), "val_loss": round(vloss, 4),
               "val_accuracy": m["overall"]["labels"]["accuracy"], "val_macro_f1": m["overall"]["labels"]["macro_f1"],
               "val_incl_macro_f1": m["inclusion"]["labels"]["macro_f1"], "val_excl_macro_f1": m["exclusion"]["labels"]["macro_f1"],
               "val_evidence_micro_f1": m["overall"]["evidence"]["micro"]["f1"], "val_parse_errors": m["overall"]["labels"]["parse_errors"],
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
        summary["epochs"].append(row)
        print(json.dumps(row), flush=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.close()


if __name__ == "__main__":
    main()
