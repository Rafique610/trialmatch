"""KV cache tests. The CPU tests use a tiny random Qwen2; the GPU test uses the real 1.5B model."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.kvcache.decode import greedy_cached, greedy_hf, greedy_no_cache
from src.kvcache.model import KVCache, Runner


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def test_forward_matches_library_logits(tiny):
    ids = torch.randint(0, 97, (20,))
    r = Runner(tiny)
    mine = r.forward(ids, r.new_cache(20), n_logits=20)
    with torch.no_grad():
        ref = tiny(input_ids=ids[None]).logits[0]
    assert torch.allclose(mine, ref, atol=1e-4)


def test_chunked_prefill_equals_full(tiny):
    """Feeding the prompt in two chunks through the cache gives the same last logits: positions and mask offsets are right."""
    ids = torch.randint(0, 97, (30,))
    r = Runner(tiny)
    full = r.forward(ids, r.new_cache(30))
    cache = r.new_cache(30)
    r.forward(ids[:17], cache)
    assert torch.allclose(r.forward(ids[17:], cache), full, atol=1e-4)


@pytest.mark.parametrize("n_in,n_out", [(8, 16), (40, 24), (120, 32)])
def test_cached_equals_no_cache_cpu(tiny, n_in, n_out, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    ids = torch.randint(0, 97, (n_in,))
    r = Runner(tiny)
    cached = greedy_cached(r, ids, n_out)
    assert cached["tokens"] == greedy_no_cache(r, ids, n_out)["tokens"] == greedy_hf(tiny, ids, n_out)["tokens"]
    assert cached["cache_bytes"][-1] == 2 * 2 * 2 * 16 * (n_in + n_out - 1) * 4  # 2 x layers x kv_heads x head_dim x len x fp32


def test_cache_crop():
    c = KVCache(1, 1, 2, 10, torch.float32, "cpu")
    c.length = 7
    c.crop(4)
    assert c.length == 4 and c.nbytes == 2 * 1 * 1 * 2 * 4 * 4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU and the downloaded 1.5B model")
@pytest.mark.parametrize("n_in,n_out", [(64, 32), (256, 64), (1024, 128)])
def test_cached_equals_reference_real_model(n_in, n_out):
    from src.kvcache import bench

    tok, model = _real()
    ids = bench.prompt_ids(tok, n_in)
    r = Runner(model)
    cached = greedy_cached(r, ids, n_out)["tokens"]
    assert cached == greedy_no_cache(r, ids, n_out)["tokens"]
    assert cached == greedy_hf(model, ids, n_out)["tokens"]


_cache = {}


def _real():
    if not _cache:
        from src.kvcache import bench

        _cache["m"] = bench.load()
    return _cache["m"]
