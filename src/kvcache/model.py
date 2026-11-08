"""Own forward pass for Qwen2 decoders with an explicit KV cache (TRIA-9).

We reuse the model's weights and its linear / RMSNorm / MLP modules, but the control flow is ours:
RoPE positions, the causal mask with a cache offset, grouped-query attention and the cache itself.
`attn_fn` is pluggable so TRIA-10 can swap in tiled attention, and `KVCache.crop` lets TRIA-11 roll back.
Batch size 1, no padding.
"""

import torch

from src.attention import reference
from src.attention.tiled import tiled_attention


class KVCache:
    """Preallocated per-layer K/V buffers [kv_heads, max_len, head_dim]; `length` = tokens stored."""

    def __init__(self, layers: int, kv_heads: int, head_dim: int, max_len: int, dtype, device):
        self.k = [torch.empty(kv_heads, max_len, head_dim, dtype=dtype, device=device) for _ in range(layers)]
        self.v = [torch.empty_like(t) for t in self.k]
        self.length = 0

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = start + k.shape[1]
        self.k[layer][:, start:end] = k
        self.v[layer][:, start:end] = v
        return self.k[layer][:, :end], self.v[layer][:, :end]

    def crop(self, n: int) -> None:
        self.length = n

    @property
    def nbytes(self) -> int:
        """Bytes actually in use: 2 (K and V) x layers x kv_heads x head_dim x length x element size."""
        k = self.k[0]
        return 2 * len(self.k) * k.shape[0] * k.shape[2] * self.length * k.element_size()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_offset: int) -> torch.Tensor:
    """Causal reference attention (src.attention.reference). q [H, Tq, d], k/v [H, Tk, d], Tk = q_offset + Tq."""
    return reference.attention(q, k, v, causal=True, q_offset=q_offset)


def tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_offset: int) -> torch.Tensor:
    """Causal tiled online-softmax attention (TRIA-10), same interface, for `attn_fn=tiled`."""
    return tiled_attention(q, k, v, causal=True, q_offset=q_offset)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Runner:
    def __init__(self, model):
        hf = model.get_base_model() if hasattr(model, "get_base_model") else model  # PEFT: LoRA layers stay active
        cfg = hf.config
        self.embed, self.layers, self.norm, self.lm_head = hf.model.embed_tokens, hf.model.layers, hf.model.norm, hf.lm_head
        self.n_heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta") or cfg.rope_theta
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=hf.device) / self.head_dim))
        self.dtype, self.device = hf.dtype, hf.device

    def new_cache(self, max_len: int) -> KVCache:
        return KVCache(len(self.layers), self.kv_heads, self.head_dim, max_len, self.dtype, self.device)

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: KVCache, n_logits: int = 1, attn_fn=attention) -> torch.Tensor:
        """Run new tokens `ids` [T] at positions cache.length .. cache.length + T - 1.
        Writes their K/V into the cache and returns logits [n_logits, vocab] for the last n_logits positions."""
        start, t = cache.length, ids.shape[0]
        cos, sin = self.rope(torch.arange(start, start + t, device=self.device))
        h = self.embed(ids)  # [T, hidden]
        group = self.n_heads // self.kv_heads
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_proj(x).view(t, self.n_heads, self.head_dim).transpose(0, 1)
            k = a.k_proj(x).view(t, self.kv_heads, self.head_dim).transpose(0, 1)
            v = a.v_proj(x).view(t, self.kv_heads, self.head_dim).transpose(0, 1)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            k_all, v_all = cache.write(i, k, v, start)
            k_all, v_all = k_all.repeat_interleave(group, 0), v_all.repeat_interleave(group, 0)  # GQA: 2 KV heads -> 12
            o = attn_fn(q, k_all, v_all, start).transpose(0, 1).reshape(t, -1)
            h = h + a.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        cache.length = start + t
        return self.lm_head(self.norm(h[-n_logits:])).float()
