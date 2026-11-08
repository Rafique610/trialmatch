"""Speculative decoding must give exactly the target's own greedy tokens (TRIA-11)."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.kvcache.decode import greedy_cached
from src.kvcache.model import Runner
from src.speculative.decode import speculative_greedy

V = 61


def tiny(seed: int, layers: int = 2):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def models():
    target = tiny(0)
    noisy = tiny(0)
    with torch.no_grad():  # a draft that agrees with the target only part of the time
        for p in noisy.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return {"target": target, "same": target, "noisy": noisy, "unrelated": tiny(1, layers=1)}


@pytest.fixture(autouse=True)
def no_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


@pytest.mark.parametrize("draft", ["same", "noisy", "unrelated"])
@pytest.mark.parametrize("gamma", [1, 2, 4, 6])
@pytest.mark.parametrize("seed", range(5))
def test_equals_target_greedy(models, draft, gamma, seed):
    ids = torch.randint(0, V, (5 + 7 * seed,), generator=torch.Generator().manual_seed(seed))
    t, d = Runner(models["target"]), Runner(models[draft])
    want = greedy_cached(t, ids, 40)["tokens"]
    got = speculative_greedy(t, d, ids, 40, gamma)
    assert got["tokens"] == want
    acc = sum(r["accepted"] for r in got["rounds"])
    if draft == "same":
        assert acc == sum(r["proposed"] for r in got["rounds"])  # identical draft: every guess accepted (bonus path)


def test_unrelated_draft_still_progresses(models):
    t, d = Runner(models["target"]), Runner(models["unrelated"])
    r = speculative_greedy(t, d, torch.arange(10), 30, 4)
    assert len(r["tokens"]) == 30 and len(r["rounds"]) <= 29  # at least one token per round even with 0 accepted


def test_eos_stops_like_greedy(models):
    t = Runner(models["target"])
    ids = torch.arange(12)
    plain = greedy_cached(t, ids, 40)["tokens"]
    eos = plain[7]  # pretend the 8th generated token is EOS
    want = plain[: plain.index(eos) + 1]
    for gamma in (2, 5):
        assert speculative_greedy(t, Runner(models["noisy"]), ids, 40, gamma, eos=eos)["tokens"] == want


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU and the downloaded models")
def test_real_pair_equals_target_greedy():
    from src.speculative import bench

    tok, target, draft = bench.load(adapter=False)
    t, d = Runner(target), Runner(draft)
    for ids in bench.prompts(tok, 5):
        want = greedy_cached(t, ids, 48, eos=bench.EOS(tok))["tokens"]
        for gamma in (2, 4, 6):
            assert speculative_greedy(t, d, ids, 48, gamma, eos=bench.EOS(tok))["tokens"] == want
