"""Greedy decoders (TRIA-9). All take a 1-D tensor of prompt ids and return n new token ids plus timings.

- greedy_no_cache: every step re-runs the whole sequence (prompt + tokens so far) from an empty cache.
- greedy_cached:   prefill the prompt once, then feed only the newest token and reuse stored K/V.
- greedy_hf:       independent reference, the library forward with use_cache=False, full recompute.
"""

import time

import torch

from src.kvcache.model import Runner, attention


def _sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def greedy_no_cache(runner: Runner, ids: torch.Tensor, n: int, eos: int | None = None, attn_fn=attention) -> dict:
    seq, out, t0 = ids, [], _sync()
    prefill = None
    while len(out) < n and (not out or out[-1] != eos):
        cache = runner.new_cache(seq.shape[0])
        nxt = runner.forward(seq, cache, attn_fn=attn_fn)[-1].argmax()
        out.append(int(nxt))
        seq = torch.cat([seq, nxt[None]])
        if prefill is None:
            prefill = _sync() - t0
    total = _sync() - t0
    return {"tokens": out, "prefill_s": prefill, "decode_s": total - prefill}


def greedy_cached(runner: Runner, ids: torch.Tensor, n: int, eos: int | None = None, attn_fn=attention) -> dict:
    cache = runner.new_cache(ids.shape[0] + n)
    t0 = _sync()
    nxt = runner.forward(ids, cache, attn_fn=attn_fn)[-1].argmax()  # prefill: whole prompt, once
    prefill = _sync() - t0
    out, cache_bytes = [int(nxt)], [cache.nbytes]
    while len(out) < n and out[-1] != eos:
        nxt = runner.forward(nxt[None], cache, attn_fn=attn_fn)[-1].argmax()  # decode: 1 token, position = cache.length
        out.append(int(nxt))
        cache_bytes.append(cache.nbytes)
    total = _sync() - t0
    return {"tokens": out, "prefill_s": prefill, "decode_s": total - prefill, "cache_bytes": cache_bytes}


@torch.no_grad()
def greedy_hf(model, ids: torch.Tensor, n: int) -> dict:
    seq, out = ids[None], []
    for _ in range(n):
        nxt = model(input_ids=seq, use_cache=False).logits[0, -1].argmax()
        out.append(int(nxt))
        seq = torch.cat([seq, nxt[None, None]], dim=1)
    return {"tokens": out}
