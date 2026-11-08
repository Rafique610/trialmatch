"""Attention error grid + speed/memory benchmark + Qwen plug-in check (TRIA-10).
-> results/bench/attention_errors.json, attention.json, attention.png
Run: uv run python -m src.attention.bench [--repeats 5] [--skip-model]
"""

import argparse
import itertools
import json
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.attention.reference import attention
from src.attention.tiled import tiled_attention
from src.config import settings

OUT = settings.results_dir / "bench"
H, D = 12, 128  # Qwen2.5-1.5B query heads and head dim
LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
BACKENDS = {"sdpa_math": SDPBackend.MATH, "sdpa_mem_efficient": SDPBackend.EFFICIENT_ATTENTION, "sdpa_flash": SDPBackend.FLASH_ATTENTION}


def rand(tq, tk, d, dtype, h=H, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(1, h, t, d, device="cuda", dtype=dtype, generator=g) for t in (tq, tk, tk)]


def errors() -> list[dict]:
    rows = []
    for seq, d, causal, dtype, blocks in itertools.product([1, 7, 64, 129, 512, 2048], [32, 64, 128], [True, False],
                                                           [torch.float32, torch.float16], [(64, 64), (48, 40)]):
        q, k, v = rand(seq, seq, d, dtype, h=4)
        ref = attention(q, k, v, causal=causal).float()
        got = tiled_attention(q, k, v, causal=causal, block_q=blocks[0], block_k=blocks[1]).float()
        sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=causal).float()
        rows.append({"seq": seq, "d": d, "causal": causal, "dtype": str(dtype).split(".")[-1], "block_q": blocks[0], "block_k": blocks[1],
                     "tiled_vs_ref_max": (got - ref).abs().max().item(), "tiled_vs_ref_mean": (got - ref).abs().mean().item(),
                     "sdpa_vs_ref_max": (sdpa - ref).abs().max().item(), "sdpa_vs_ref_mean": (sdpa - ref).abs().mean().item()})
    return rows


def measure(fn, repeats: int) -> dict:
    try:
        fn()
        torch.cuda.synchronize()
        times, peaks = [], []
        for _ in range(repeats):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            peaks.append((torch.cuda.max_memory_allocated() - base) / 1024**2)
            del out
        return {"ms": round(statistics.mean(times), 3), "ms_std": round(statistics.stdev(times), 3), "extra_mb": round(max(peaks), 1)}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "out of memory"}
    except RuntimeError as e:  # backend not compiled in / not supported on this GPU
        return {"error": str(e).splitlines()[0][:160]}


def sdpa_with(backend):
    def run(q, k, v):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return run


def timing(repeats: int) -> list[dict]:
    rows = []
    for n in LENGTHS:
        q, k, v = rand(n, n, D, torch.float16)
        row = {"seq": n, "inputs_mb": round(3 * q.numel() * 2 / 1024**2, 1),
               "reference": measure(lambda: attention(q, k, v), repeats),
               "tiled_64": measure(lambda: tiled_attention(q, k, v, block_q=64, block_k=64), repeats),
               "tiled_128": measure(lambda: tiled_attention(q, k, v, block_q=128, block_k=128), repeats)}
        for name, b in BACKENDS.items():
            row[name] = measure(lambda: sdpa_with(b)(q, k, v), repeats)
        print({k2: (v2.get("ms"), v2.get("extra_mb"), v2.get("error")) if isinstance(v2, dict) else v2 for k2, v2 in row.items()}, flush=True)
        rows.append(row)
        del q, k, v
    return rows


def block_sweep(repeats: int) -> list[dict]:
    q, k, v = rand(2048, 2048, D, torch.float16)
    return [{"block": b, **measure(lambda: tiled_attention(q, k, v, block_q=b, block_k=b), repeats)} for b in (16, 32, 64, 128, 256, 512)]


def in_model() -> dict:
    """Bonus: swap the tiled kernel into every layer of the real model (TRIA-9 runner) and compare."""
    from src.kvcache import bench as kv
    from src.kvcache.decode import greedy_cached
    from src.kvcache.model import Runner, attention as ref_attn, tiled

    tok, model = kv.load()
    r = Runner(model)
    ids = kv.prompt_ids(tok, 512)
    a = r.forward(ids, r.new_cache(512), n_logits=512)
    b = r.forward(ids, r.new_cache(512), n_logits=512, attn_fn=tiled)
    ta, tb = greedy_cached(r, ids, 64, attn_fn=ref_attn)["tokens"], greedy_cached(r, ids, 64, attn_fn=tiled)["tokens"]
    res = {"prompt_tokens": 512, "new_tokens": 64, "prefill_logits_max_abs_diff": round((a - b).abs().max().item(), 4),
           "prefill_argmax_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item(), "greedy_tokens_equal": ta == tb}
    del model
    torch.cuda.empty_cache()
    return res


def plot(rows: list[dict], sweep: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["reference", "tiled_64", "tiled_128", *BACKENDS]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.2))
    for name in names:
        pts = [(r["seq"], r[name]) for r in rows if "ms" in r[name]]
        if pts:
            a.plot([p[0] for p in pts], [p[1]["ms"] for p in pts], marker="o", label=name)
            b.plot([p[0] for p in pts], [p[1]["extra_mb"] for p in pts], marker="o", label=name)
    for ax, title, unit in ((a, "Time (causal, 12 heads, d=128, fp16)", "ms"), (b, "Peak extra memory over inputs", "MB")):
        ax.set(title=title, xlabel="sequence length", ylabel=unit, xscale="log", yscale="log")
        ax.minorticks_off()
        ax.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
        ax.legend(fontsize=8)
    ok = [s for s in sweep if "ms" in s]
    c.plot([s["block"] for s in ok], [s["ms"] for s in ok], marker="o", color="tab:orange", label="time (ms)")
    c2 = c.twinx()
    c2.plot([s["block"] for s in ok], [s["extra_mb"] for s in ok], marker="s", color="tab:green", label="extra MB")
    c.set(title="Tiled block size at seq 2048", xlabel="block size (Br = Bc)", ylabel="ms", xscale="log")
    c2.set_ylabel("MB")
    c.minorticks_off()
    c.set_xticks([s["block"] for s in ok], [str(s["block"]) for s in ok])
    fig.tight_layout()
    fig.savefig(OUT / "attention.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw attention.png from attention.json")
    args = ap.parse_args()
    if args.plot_only:
        r = json.loads((OUT / "attention.json").read_text(encoding="utf-8"))
        return plot(r["timing"], r["block_sweep_seq2048"])
    OUT.mkdir(parents=True, exist_ok=True)
    errs = errors()
    summary = {}
    for dt in ("float32", "float16"):
        sel = [e for e in errs if e["dtype"] == dt]
        summary[dt] = {"configs": len(sel), "tiled_vs_ref_max": max(e["tiled_vs_ref_max"] for e in sel),
                       "tiled_vs_ref_mean": statistics.mean(e["tiled_vs_ref_mean"] for e in sel),
                       "sdpa_vs_ref_max": max(e["sdpa_vs_ref_max"] for e in sel), "sdpa_vs_ref_mean": statistics.mean(e["sdpa_vs_ref_mean"] for e in sel)}
    (OUT / "attention_errors.json").write_text(json.dumps({"summary": summary, "rows": errs}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "repeats": args.repeats, "warmup": 1,
              "shape": {"batch": 1, "heads": H, "head_dim": D, "dtype": "float16", "causal": True},
              "timing": timing(args.repeats), "block_sweep_seq2048": block_sweep(args.repeats)}
    print(result["block_sweep_seq2048"], flush=True)
    if not args.skip_model:
        result["in_model"] = in_model()
        print(result["in_model"], flush=True)
    (OUT / "attention.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(result["timing"], result["block_sweep_seq2048"])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
