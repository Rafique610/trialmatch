"""KV-cache equivalence check + benchmark (TRIA-9) -> results/bench/kvcache.json, kvcache.png.

Prompts are real validation prompts (chat template), concatenated and cut to the target length.
Run: uv run python -m src.kvcache.bench [--repeats 5]
"""

import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from src.data.prompt import messages
from src.eval.run import read_jsonl
from src.kvcache.decode import greedy_cached, greedy_hf, greedy_no_cache
from src.kvcache.model import Runner

EQUIV = [(64, 32), (256, 64), (1024, 128)]
BENCH_IN, BENCH_OUT = [128, 512, 1024, 2048], 64
OUT = settings.results_dir / "bench"


def load():
    tok = AutoTokenizer.from_pretrained(settings.target_model_id)
    model = AutoModelForCausalLM.from_pretrained(settings.target_model_id, dtype=torch.float16, device_map="cuda").eval()
    return tok, model


def prompt_ids(tok, length: int) -> torch.Tensor:
    ids = []
    for r in read_jsonl(settings.data_dir / "processed" / "val.jsonl"):
        ids += tok(tok.apply_chat_template(messages(r), add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
        if len(ids) >= length:
            return torch.tensor(ids[:length], device="cuda")
    raise ValueError(length)


@torch.no_grad()
def first_divergence(model, ids, ref: list[int], got: list[int]) -> dict | None:
    i = next((j for j, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    if i is None:
        return None
    logits = model(input_ids=torch.cat([ids, torch.tensor(ref[:i], device="cuda", dtype=ids.dtype)])[None]).logits[0, -1].float()
    top = logits.topk(2)
    return {"step": i, "ref_token": ref[i], "got_token": got[i], "ref_top1_minus_top2_logit": round(float(top.values[0] - top.values[1]), 4),
            "got_token_rank_in_ref": int((logits > logits[got[i]]).sum())}


def equivalence(tok, model, runner) -> list[dict]:
    rows = []
    for n_in, n_out in EQUIV:
        ids = prompt_ids(tok, n_in)
        cached = greedy_cached(runner, ids, n_out)["tokens"]
        own_nc = greedy_no_cache(runner, ids, n_out)["tokens"]
        hf = greedy_hf(model, ids, n_out)["tokens"]
        rows.append({"input_len": n_in, "output_len": n_out,
                     "cached_equals_own_no_cache": cached == own_nc, "cached_equals_hf_no_cache": cached == hf,
                     "divergence_vs_own": first_divergence(model, ids, own_nc, cached),
                     "divergence_vs_hf": first_divergence(model, ids, hf, cached)})
        print(rows[-1], flush=True)
    return rows


def timed(fn, repeats: int) -> dict:
    fn()  # warm-up
    runs = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        r = fn()
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        runs.append(r)
    stat = lambda k: {"mean": round(statistics.mean(x[k] for x in runs), 4), "std": round(statistics.stdev(x[k] for x in runs), 4)}
    out = {"prefill_ms": stat("prefill_ms"), "decode_ms": stat("decode_ms"), "decode_tok_s": stat("decode_tok_s"),
           "total_tok_s": stat("total_tok_s"), "peak_gb": round(max(x["peak_gb"] for x in runs), 3)}
    if "cache_bytes" in runs[0]:
        out["cache_bytes_per_step"] = runs[0]["cache_bytes"]
    return out


def as_metrics(r: dict, n: int) -> dict:
    p, d = r["prefill_s"], r["decode_s"]
    m = {"prefill_ms": p * 1000, "decode_ms": d * 1000, "decode_tok_s": (n - 1) / d, "total_tok_s": n / (p + d)}
    if "cache_bytes" in r:
        m["cache_bytes"] = r["cache_bytes"]
    return m


def hf_generate(model, ids, n) -> dict:
    """Library reference with its own cache; only total time is comparable."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(ids[None], attention_mask=torch.ones_like(ids)[None], max_new_tokens=n, min_new_tokens=n, do_sample=False,
                   temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    torch.cuda.synchronize()
    return {"prefill_s": 0.0, "decode_s": time.perf_counter() - t0}


def benchmark(tok, model, runner, repeats: int) -> list[dict]:
    rows = []
    for n_in in BENCH_IN:
        ids = prompt_ids(tok, n_in)
        row = {"input_len": n_in, "output_len": BENCH_OUT}
        row["no_cache"] = timed(lambda: as_metrics(greedy_no_cache(runner, ids, BENCH_OUT), BENCH_OUT), repeats)
        row["cached"] = timed(lambda: as_metrics(greedy_cached(runner, ids, BENCH_OUT), BENCH_OUT), repeats)
        hf = timed(lambda: as_metrics(hf_generate(model, ids, BENCH_OUT), BENCH_OUT), repeats)
        row["hf_generate_cached"] = {"total_ms": hf["decode_ms"], "total_tok_s": hf["total_tok_s"], "peak_gb": hf["peak_gb"]}
        row["decode_speedup"] = round(row["no_cache"]["decode_ms"]["mean"] / row["cached"]["decode_ms"]["mean"], 2)
        print({k: row[k] for k in ("input_len", "decode_speedup")}, row["cached"]["decode_tok_s"], row["no_cache"]["decode_tok_s"], flush=True)
        rows.append(row)
    return rows


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["input_len"] for r in rows]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
    for key, name in (("no_cache", "no cache (recompute)"), ("cached", "KV cache (own)")):
        a.errorbar(x, [r[key]["decode_ms"]["mean"] / (BENCH_OUT - 1) for r in rows], [r[key]["decode_ms"]["std"] / (BENCH_OUT - 1) for r in rows], marker="o", label=name, capsize=3)
        b.plot(x, [r[key]["peak_gb"] for r in rows], marker="o", label=name)
    a.set(title=f"Decode latency per token ({BENCH_OUT} new tokens)", xlabel="input tokens", ylabel="ms / token")
    b.set(title="Peak GPU memory", xlabel="input tokens", ylabel="GB")
    for r in rows:
        cb = r["cached"]["cache_bytes_per_step"]
        c.plot(range(r["input_len"], r["input_len"] + len(cb)), [v / 1024**2 for v in cb], label=f"input {r['input_len']}")
    c.set(title="KV cache size while decoding", xlabel="sequence length", ylabel="MB")
    for ax in (a, b, c):
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "kvcache.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = load()
    runner = Runner(model)
    result = {"model": settings.target_model_id, "precision": "fp16", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "repeats": args.repeats, "warmup": 1, "equivalence": equivalence(tok, model, runner),
              "benchmark": benchmark(tok, model, runner, args.repeats)}
    (OUT / "kvcache.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(result["benchmark"])
    print(f"-> {OUT / 'kvcache.json'}")


if __name__ == "__main__":
    main()
