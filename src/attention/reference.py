"""Reference scaled dot-product attention (TRIA-10). Materialises the full [Tq, Tk] score matrix.

Shapes: q [..., Tq, d], k and v [..., Tk, d]. Query i sits at absolute position q_offset + i
(q_offset = number of cached keys before the queries, TRIA-9), so under `causal` it may see keys 0..q_offset + i.
`key_mask`: bool, True = keep, broadcastable to [..., Tq, Tk] (e.g. [B, 1, 1, Tk] for padding).
Rows with every key masked return 0 instead of NaN. Math in fp32, output in q's dtype.
"""

import math

import torch


def allowed(tq: int, tk: int, q_offset: int, causal: bool, key_mask, device, q0: int = 0, k0: int = 0) -> torch.Tensor | None:
    """Bool [.., tq, tk] of visible (query q0+i, key k0+j) pairs, or None when nothing is masked."""
    keep = None
    if causal:
        qpos = q_offset + q0 + torch.arange(tq, device=device)
        keep = (k0 + torch.arange(tk, device=device))[None, :] <= qpos[:, None]
    if key_mask is not None:
        m = key_mask[..., q0:q0 + tq, :] if key_mask.shape[-2] > 1 else key_mask
        m = m[..., k0:k0 + tk]
        keep = m if keep is None else keep & m
    return keep


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, q_offset: int = 0,
              key_mask: torch.Tensor | None = None) -> torch.Tensor:
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    keep = allowed(q.shape[-2], k.shape[-2], q_offset, causal, key_mask, q.device)
    if keep is not None:
        scores = scores.masked_fill(~keep, float("-inf"))
    m = scores.amax(-1, keepdim=True)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)  # fully masked row: avoid -inf - -inf
    p = torch.exp(scores - m)  # max-subtraction: every exponent <= 0, so no overflow
    l = p.sum(-1, keepdim=True)
    out = (p @ v.float()) / torch.where(l == 0, torch.ones_like(l), l)
    return out.to(q.dtype)
