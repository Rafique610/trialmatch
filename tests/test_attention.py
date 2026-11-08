"""Tiled online-softmax attention vs the reference and torch SDPA (TRIA-10). CPU, so it runs anywhere.

Tolerances (max abs error): fp32 1e-5; fp16 inputs 2e-3 (both implementations accumulate in fp32;
the gap is fp16 rounding of inputs/outputs). Measured errors over a larger GPU grid: results/bench/attention_errors.json.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

from src.attention.reference import attention
from src.attention.tiled import tiled_attention

TOL = {torch.float32: 1e-5, torch.float16: 2e-3}
SEQS, DIMS = [1, 7, 64, 129, 512], [32, 64, 128]


def qkv(b, h, tq, tk, d, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, h, t, d, generator=g).to(dtype) for t in (tq, tk, tk)]


@pytest.mark.parametrize("seq,d,causal,dtype", list(itertools.product(SEQS, DIMS, [True, False], [torch.float32, torch.float16])))
def test_tiled_matches_reference_and_sdpa(seq, d, causal, dtype):
    q, k, v = qkv(2, 3, seq, seq, d, dtype)
    ref = attention(q, k, v, causal=causal)
    got = tiled_attention(q, k, v, causal=causal, block_q=48, block_k=40)  # blocks that do not divide most lengths
    sdpa = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal).to(dtype)
    assert (got.float() - ref.float()).abs().max() <= TOL[dtype]
    assert (ref.float() - sdpa.float()).abs().max() <= TOL[dtype]


@pytest.mark.parametrize("block_q,block_k", [(1, 1), (16, 64), (64, 16), (129, 129), (512, 512)])
def test_block_sizes(block_q, block_k):
    q, k, v = qkv(1, 2, 129, 129, 64, torch.float32)
    assert (tiled_attention(q, k, v, block_q=block_q, block_k=block_k) - attention(q, k, v)).abs().max() <= 1e-5


@pytest.mark.parametrize("tq,offset", [(1, 99), (5, 95), (37, 63)])
def test_queries_after_a_cache(tq, offset):
    """Decode / chunked prefill: fewer queries than keys, positions shifted by the cache length."""
    q, k, v = qkv(1, 2, tq, offset + tq, 64, torch.float32)
    full_q = torch.cat([torch.zeros(1, 2, offset, 64), q], dim=2)
    expected = attention(full_q, k, v)[:, :, offset:]
    assert (tiled_attention(q, k, v, q_offset=offset, block_q=8, block_k=16) - expected).abs().max() <= 1e-5
    assert (attention(q, k, v, q_offset=offset) - expected).abs().max() <= 1e-5


def test_padding_mask_and_fully_masked_rows():
    q, k, v = qkv(2, 2, 10, 10, 32, torch.float32)
    mask = torch.ones(2, 1, 1, 10, dtype=torch.bool)
    mask[1, ..., 6:] = False  # second sequence: last 4 keys are padding
    ref = attention(q, k, v, causal=False, key_mask=mask)
    sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert (tiled_attention(q, k, v, causal=False, key_mask=mask, block_q=3, block_k=4) - ref).abs().max() <= 1e-5
    assert (ref - sdpa).abs().max() <= 1e-5
    none = torch.zeros(2, 1, 1, 10, dtype=torch.bool)  # every key masked: defined as 0, not NaN
    for f in (attention, tiled_attention):
        out = f(q, k, v, causal=False, key_mask=none)
        assert not out.isnan().any() and out.abs().max() == 0


def test_large_scores_stay_finite():
    """Max-subtraction: scores in the thousands would overflow a naive exp."""
    q, k, v = qkv(1, 1, 64, 64, 64, torch.float32)
    q = q * 300
    ref, got = attention(q, k, v), tiled_attention(q, k, v, block_q=16, block_k=16)
    assert torch.isfinite(ref).all() and (got - ref).abs().max() <= 1e-4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU and the downloaded 1.5B model")
def test_tiled_inside_real_model_gives_same_tokens():
    from src.kvcache import bench
    from src.kvcache.decode import greedy_cached
    from src.kvcache.model import Runner, tiled

    tok, model = bench.load()
    r = Runner(model)
    ids = bench.prompt_ids(tok, 256)
    assert greedy_cached(r, ids, 32, attn_fn=tiled)["tokens"] == greedy_cached(r, ids, 32)["tokens"]
