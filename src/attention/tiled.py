"""Tiled attention with an online softmax (TRIA-10), pure PyTorch, no fused kernel.

For each query block (Br rows) we stream over key/value blocks (Bc columns) and keep, per query row:
  m  running max of the scores seen so far
  l  running sum of exp(score - m)
  O  running sum of exp(score - m) * V
When a new block raises the max to m_new, the old l and O are rescaled by exp(m_old - m_new), so the
result equals softmax over all keys without ever holding the [Tq, Tk] score matrix: peak extra memory
is one [Br, Bc] tile per head. Under `causal`, key blocks entirely in the future are skipped.
Same arguments and masking conventions as src.attention.reference.attention.
"""

import math

import torch

from src.attention.reference import allowed


def tiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, q_offset: int = 0,
                    key_mask: torch.Tensor | None = None, block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    tq, tk, scale = q.shape[-2], k.shape[-2], 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty(*q.shape[:-1], v.shape[-1], dtype=q.dtype, device=q.device)
    for q0 in range(0, tq, block_q):
        qb = q[..., q0:q0 + block_q, :].float() * scale
        br = qb.shape[-2]
        m = torch.full((*qb.shape[:-1], 1), float("-inf"), device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros(*qb.shape[:-1], v.shape[-1], device=q.device)
        last_visible = q_offset + q0 + br - 1  # the latest key any row of this block may see
        for k0 in range(0, tk, block_k):
            if causal and k0 > last_visible:
                break  # this and every later key block is in the future for the whole query block
            s = qb @ k[..., k0:k0 + block_k, :].float().transpose(-1, -2)
            keep = allowed(br, s.shape[-1], q_offset, causal, key_mask, q.device, q0, k0)
            if keep is not None:
                s = s.masked_fill(~keep, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # still no visible key yet
            p = torch.exp(s - m_safe)
            alpha = torch.exp(m - m_safe)  # rescale what was accumulated under the old max (0 when m was -inf)
            l = alpha * l + p.sum(-1, keepdim=True)
            acc = alpha * acc + p @ v[..., k0:k0 + block_k, :].float()
            m = m_new
        out[..., q0:q0 + br, :] = (acc / torch.where(l == 0, torch.ones_like(l), l)).to(q.dtype)
    return out
