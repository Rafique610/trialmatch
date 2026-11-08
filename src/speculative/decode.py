"""Greedy speculative decoding with our own KV caches (TRIA-11). Built on the TRIA-9 Runner.

Invariant between rounds: `out` holds every emitted token. The last one (`pending`) has been emitted
but not yet run through the target, so the target cache covers prompt + out[:-1].

One round:
  1. Draft: run whatever the draft cache is missing (1–2 tokens), then propose γ tokens greedily, one at a time.
  2. Target: ONE forward over [pending, d1..dγ] -> its own greedy choice after each prefix, p0..pγ.
  3. Accept d1..dn while d_i == p_{i-1}; then emit p_n (the correction, or a bonus token if all γ matched).
  4. Roll back: crop the target cache to prompt + accepted tokens, and the draft cache the same way,
     so the rejected guesses are forgotten and positions stay consistent.
The output equals the target's own greedy decoding: every emitted token is a target argmax given
exactly the tokens before it.
"""

import time

import torch

from src.kvcache.model import Runner


def _sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def speculative_greedy(target: Runner, draft: Runner, ids: torch.Tensor, n: int, gamma: int, eos: int | None = None) -> dict:
    total = ids.shape[0] + n + gamma + 1
    tc, dc = target.new_cache(total), draft.new_cache(total)
    t0 = _sync()
    out = [int(target.forward(ids, tc)[-1].argmax())]  # target prefill -> first token (same as plain greedy)
    draft.forward(ids, dc)  # draft prefill; its next-token guess is not needed (the target already chose)
    prefill_s = _sync() - t0
    seq = ids.tolist() + out
    rounds, draft_s, verify_s = [], 0.0, 0.0
    while len(out) < n and out[-1] != eos:
        g = min(gamma, n - len(out))  # never draft past the length limit
        # 1. draft proposes g tokens
        t1 = _sync()
        nxt = draft.forward(torch.tensor(seq[dc.length:], device=ids.device), dc)[-1].argmax()
        guesses = [int(nxt)]
        while len(guesses) < g:
            nxt = draft.forward(nxt[None], dc)[-1].argmax()
            guesses.append(int(nxt))
        t2 = _sync()
        # 2. target scores pending + guesses in one pass
        base = tc.length
        block = torch.tensor([seq[-1], *guesses], device=ids.device)
        choice = target.forward(block, tc, n_logits=g + 1).argmax(-1).tolist()
        t3 = _sync()
        # 3. accept the matching prefix, then the target's own token
        k = 0
        while k < g and guesses[k] == choice[k]:
            k += 1
        new = guesses[:k] + [choice[k]]
        if eos in new:
            new = new[: new.index(eos) + 1]
        new = new[: n - len(out)]
        # 4. roll back both caches to what is actually emitted (all but the new pending token)
        tc.crop(base + len(new))
        dc.crop(min(dc.length, base + len(new)))
        out += new
        seq += new
        rounds.append({"proposed": g, "accepted": k})
        draft_s += t2 - t1
        verify_s += t3 - t2
    total_s = _sync() - t0
    return {"tokens": out, "rounds": rounds, "prefill_s": prefill_s, "draft_s": draft_s, "verify_s": verify_s, "total_s": total_s}
