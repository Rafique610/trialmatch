"""Speculative decoding benchmark (TRIA-11) -> results/bench/speculative.json, speculative.png.

Target Qwen2.5-1.5B-Instruct (base, then with the fine-tuned adapter merged), draft Qwen2.5-0.5B-Instruct.
Baseline = our plain cached greedy decoder (TRIA-9) on the same prompts. Two workloads:
  task     the real screening prompts (validation), short JSON answers, stop at <|im_end|>
  summary  "summarise this note" on the same notes, free text, base target only (where drafts agree more)
Run: uv run python -m src.speculative.bench [--repeats 3] [--n 20]
"""

import argparse
import json
import statistics

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from src.data.prompt import messages
from src.eval.run import read_jsonl
from src.kvcache.decode import _sync, greedy_cached
from src.kvcache.model import Runner
from src.speculative.decode import speculative_greedy

OUT = settings.results_dir / "bench"
GAMMAS = [1, 2, 3, 4, 6, 8]


def EOS(tok) -> int:
    return tok.convert_tokens_to_ids("<|im_end|>")


def load(adapter: bool):
    tok = AutoTokenizer.from_pretrained(settings.target_model_id)
    target = AutoModelForCausalLM.from_pretrained(settings.target_model_id, dtype=torch.float16, device_map="cuda").eval()
    if adapter:
        target = merge_adapter(target)
    draft = AutoModelForCausalLM.from_pretrained(settings.model_source(settings.draft_model_id), dtype=torch.float16, device_map="cuda").eval()
    return tok, target, draft


def merge_adapter(target):
    path = settings.checkpoints_dir / "final"
    return PeftModel.from_pretrained(target, path if path.exists() else settings.adapter_repo_id).merge_and_unload().eval()


def prompts(tok, n: int, kind: str = "task") -> list[torch.Tensor]:
    rows = read_jsonl(settings.data_dir / "processed" / "val.jsonl")
    if kind == "summary":  # one prompt per distinct patient note (many rows share a note)
        rows = list({r["patient_id"]: r for r in rows}.values())[:n]
    else:
        rows = rows[: n * 5 : 5]  # spread over patients and trials
    out = []
    for r in rows:
        msgs = messages(r) if kind == "task" else [
            {"role": "user", "content": "Summarise this patient note in plain English in 3-4 sentences:\n\n" + " ".join(r["sentences"])}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out.append(torch.tensor(tok(text, add_special_tokens=False)["input_ids"], device="cuda"))
    return out


def run_set(fn, ps, repeats: int) -> dict:
    fn(ps[0])  # warm-up
    times, results = [], None
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        t0 = _sync()
        results = [fn(p) for p in ps]
        times.append(_sync() - t0)
    tokens = sum(len(r["tokens"]) for r in results)
    return {"seconds": statistics.mean(times), "seconds_std": statistics.stdev(times) if repeats > 1 else 0.0,
            "tokens": tokens, "tok_s": tokens / statistics.mean(times), "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "results": results}


def workload(name: str, tok, target, draft, ps, max_new: int, repeats: int) -> dict:
    t, d, eos = Runner(target), Runner(draft), EOS(tok)
    base = run_set(lambda p: greedy_cached(t, p, max_new, eos=eos), ps, repeats)
    step_ms = statistics.mean(r["decode_s"] / max(1, len(r["tokens"]) - 1) for r in base["results"]) * 1000
    row = {"workload": name, "prompts": len(ps), "max_new_tokens": max_new,
           "mean_prompt_tokens": round(statistics.mean(p.shape[0] for p in ps), 1),
           "baseline": {k: round(v, 4) for k, v in base.items() if k != "results"} | {"target_decode_ms_per_token": round(step_ms, 2)},
           "gammas": []}
    print(name, "baseline", row["baseline"], flush=True)
    for g in GAMMAS:
        s = run_set(lambda p: speculative_greedy(t, d, p, max_new, g, eos=eos), ps, repeats)
        rounds = [x for r in s["results"] for x in r["rounds"]]
        proposed, accepted = sum(x["proposed"] for x in rounds), sum(x["accepted"] for x in rounds)
        same = sum(a["tokens"] == b["tokens"] for a, b in zip(s["results"], base["results"]))
        hist = [sum(1 for x in rounds if x["accepted"] == k) for k in range(g + 1)]
        row["gammas"].append({
            "gamma": g, "seconds": round(s["seconds"], 4), "seconds_std": round(s["seconds_std"], 4), "tok_s": round(s["tok_s"], 2),
            "speedup": round(base["seconds"] / s["seconds"], 3), "peak_gb": round(s["peak_gb"], 3),
            "acceptance_rate": round(accepted / proposed, 4), "mean_accepted_per_round": round(accepted / len(rounds), 3),
            "tokens_per_round": round(s["tokens"] / (len(rounds) + len(ps)), 3), "rounds": len(rounds), "accepted_histogram": hist,
            "draft_ms_per_round": round(1000 * sum(r["draft_s"] for r in s["results"]) / len(rounds), 2),
            "verify_ms_per_round": round(1000 * sum(r["verify_s"] for r in s["results"]) / len(rounds), 2),
            "outputs_identical_to_greedy": f"{same}/{len(ps)}"})
        print(name, row["gammas"][-1], flush=True)
    return row


@torch.no_grad()
def hf_assisted_check(tok, target, draft, ps, max_new: int) -> dict:
    """Reference only. The model's generation_config sets repetition_penalty=1.1, so it is reset to 1.0 here
    to compare pure greedy with pure greedy."""
    t, eos = Runner(target), EOS(tok)
    kw = dict(do_sample=False, max_new_tokens=max_new, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    res = {"prompts": len(ps), "hf_assisted_equals_hf_greedy": 0, "ours_equals_hf_greedy": 0, "ours_speculative_equals_hf_assisted": 0}
    d = Runner(draft)
    for p in ps:
        mask = torch.ones_like(p)[None]
        plain = target.generate(p[None], attention_mask=mask, **kw)[0, p.shape[0]:].tolist()
        assisted = target.generate(p[None], attention_mask=mask, assistant_model=draft, **kw)[0, p.shape[0]:].tolist()
        ours = greedy_cached(t, p, max_new, eos=eos)["tokens"]
        spec = speculative_greedy(t, d, p, max_new, 4, eos=eos)["tokens"]
        res["hf_assisted_equals_hf_greedy"] += assisted == plain
        res["ours_equals_hf_greedy"] += ours == plain[: len(ours)]
        res["ours_speculative_equals_hf_assisted"] += spec == assisted[: len(spec)]
    return res


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
    for r in rows:
        gs = [x["gamma"] for x in r["gammas"]]
        a.plot(gs, [x["speedup"] for x in r["gammas"]], marker="o", label=r["workload"])
        b.plot(gs, [x["acceptance_rate"] for x in r["gammas"]], marker="o", label=r["workload"])
        c.plot(gs, [x["tokens_per_round"] for x in r["gammas"]], marker="o", label=r["workload"])
    a.axhline(1, color="grey", ls="--", lw=1)
    a.set(title="Speed-up vs plain greedy (target only)", xlabel="γ (draft tokens per round)", ylabel="×")
    b.set(title="Draft acceptance rate", xlabel="γ", ylabel="accepted / proposed", ylim=(0, 1))
    c.set(title="Tokens emitted per target pass", xlabel="γ", ylabel="tokens")
    for ax in (a, b, c):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "speculative.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok, target, draft = load(adapter=False)
    task_ps, sum_ps = prompts(tok, args.n), prompts(tok, args.n, "summary")
    rows = [workload("task, base target", tok, target, draft, task_ps, 64, args.repeats),
            workload("summary, base target", tok, target, draft, sum_ps, 128, args.repeats)]
    check = hf_assisted_check(tok, target, draft, task_ps[:5], 64)
    print("hf assisted", check, flush=True)
    target = merge_adapter(target)
    rows.insert(0, workload("task, fine-tuned target", tok, target, draft, task_ps, 64, args.repeats))
    result = {"target": settings.target_model_id, "draft": settings.draft_model_id, "adapter": settings.adapter_repo_id,
              "precision": "fp16", "gpu": torch.cuda.get_device_name(0), "repeats": args.repeats, "warmup": 1,
              "hf_assisted_reference": check, "workloads": rows}
    (OUT / "speculative.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(rows)
    print(f"-> {OUT / 'speculative.json'}")


if __name__ == "__main__":
    main()
