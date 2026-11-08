"""Live benchmarks for the app: the same prompt through every decoder, and a small attention run.
Every number comes from running it now on this GPU (core rule: real data only)."""

import torch

from src.app.screening import MAX_PROMPT_TOKENS, InputError, split_note
from src.attention.bench import measure, rand
from src.attention.reference import attention
from src.attention.tiled import tiled_attention

MODES = ("no_cache", "kv_cache", "speculative")


def decoders(engine, note: str, ctype: str, ctext: str, gamma: int) -> dict:
    sentences = split_note(note)
    if not sentences:
        raise InputError("the note has no sentences after splitting")
    ids = engine.prompt_ids(sentences, ctype, ctext)
    if ids.shape[0] > MAX_PROMPT_TOKENS:
        raise InputError(f"note + criterion is {ids.shape[0]} tokens; the limit on this GPU is {MAX_PROMPT_TOKENS}")
    engine.decode(ids, "kv_cache")  # warm-up, not reported
    runs = {m: engine.decode(ids, m, gamma) for m in MODES}
    ref = runs["kv_cache"]["tokens"]
    rows = []
    for m, r in runs.items():
        row = {"mode": m, "same_tokens_as_kv_cache": r["tokens"] == ref, "answer": r["text"],
               "speedup_vs_no_cache": round((runs["no_cache"]["prefill_ms"] + runs["no_cache"]["decode_ms"]) / (r["prefill_ms"] + r["decode_ms"]), 2)}
        rows.append(row | {k: v for k, v in r.items() if k not in ("tokens", "text")})
    return {"prompt_tokens": int(ids.shape[0]), "gamma": gamma, "runs": rows, "provenance": engine.provenance}


def attention_live(engine, seq_lengths: list[int], block: int) -> dict:
    rows = []
    with engine.lock:
        for n in seq_lengths:
            q, k, v = rand(n, n, 128, torch.float16)
            err = (tiled_attention(q, k, v, block_q=block, block_k=block).float() - attention(q, k, v).float()).abs().max().item()
            rows.append({"seq": n, "max_abs_error_tiled_vs_reference": err,
                         "reference": measure(lambda: attention(q, k, v), 3),
                         "tiled": measure(lambda: tiled_attention(q, k, v, block_q=block, block_k=block), 3),
                         "torch_sdpa": measure(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), 3)})
            del q, k, v
    return {"heads": 12, "head_dim": 128, "dtype": "float16", "causal": True, "block": block, "repeats": 3, "rows": rows}
