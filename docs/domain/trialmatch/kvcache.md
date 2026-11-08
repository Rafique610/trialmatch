# KV cache (TRIA-9)

Code: `src/kvcache/model.py` (own forward + cache), `src/kvcache/decode.py` (decoders),
`src/kvcache/bench.py` (equivalence + benchmark), `tests/test_kvcache.py`.
Not `generate(use_cache=True)`: the cache tensors, positions, mask and attention are ours.

## What is ours and what is reused

`Runner.forward` runs the Qwen2 stack itself, layer by layer. From the library it reuses
only the weights and the small modules: embedding, `q/k/v/o_proj`, the RMSNorms, the MLP
and `lm_head`. Everything that decides how tokens see each other is our code:

| Piece | Where | How |
| --- | --- | --- |
| KV cache | `KVCache` | Preallocated `[kv_heads=2, max_len, head_dim=128]` K and V buffers per layer (28 layers). `length` counts stored tokens; `crop(n)` rolls back (used by TRIA-11). |
| Positions (RoPE) | `Runner.rope` | New tokens get absolute positions `cache.length … cache.length + T − 1`. θ = 1e6 from the config, `inv_freq = θ^(−2i/d)`, rotation `x·cos + rotate_half(x)·sin` on q and k **before** k is stored, so cached keys never need re-rotating. |
| Causal mask with a cache | `attention` | Query *i* sits at absolute position `q_offset + i` and may see keys `0 … q_offset + i`. During decode (`Tq = 1`) that is every cached key, so the mask is all-true. During prefill it is the usual lower triangle. |
| Grouped-query attention | `Runner.forward` | Only the 2 KV heads are cached; they are repeated ×6 to meet the 12 query heads at attention time. That is why the cache is 6× smaller than full multi-head would be. |
| Attention | `attention` (pluggable `attn_fn`) | Reference: scores in fp32, `−inf` on future keys, softmax, `@ V`. TRIA-10 swaps in tiled attention here. |

## Decoders

- `greedy_no_cache`: every step builds an empty cache and runs the **whole** sequence
  (prompt + tokens so far). Step *t* costs a forward over `L + t` tokens, so the total is O(n·L) tokens of work.
- `greedy_cached`: **prefill** runs the prompt once and fills the cache. Each **decode** step then feeds only
  the newest token (1 query) against all cached keys and appends its K/V. Total work is L + n tokens.
- `greedy_hf`: an independent reference, the library forward with `use_cache=False` (full recompute).

## Cache size

bytes = 2 (K, V) × 28 layers × 2 KV heads × 128 head_dim × sequence length × 2 bytes (fp16)
= **28 672 bytes per token (28 KB)**. It grows linearly, O(n): 1024 tokens → 28 MB, 2048 → 56 MB.
`KVCache.nbytes` reports this after every step, and the right panel of `results/bench/kvcache.png` plots it.

## Equivalence

`results/bench/kvcache.json` → `equivalence`. Prompts are real validation prompts, cut to length. fp16, greedy.

| Input / output tokens | Cached = own no-cache | Cached = library no-cache | First divergence |
| --- | --- | --- | --- |
| 64 / 32 | yes | yes | none |
| 256 / 64 | yes | yes | none |
| 1024 / 128 | yes | yes | none |

All 224 generated tokens match exactly. No mismatch occurred, so no near-tie needed explaining.

## Benchmark

`task bench` → `results/bench/kvcache.json`, `kvcache.png`. RTX 3050 6 GB laptop GPU, fp16,
64 new tokens, 1 warm-up and 5 timed repeats per setting, `torch.cuda.synchronize()` around every
timed phase, peak memory reset before each run. Mean ± std:

| Input tokens | Prefill ms | Decode ms/token, no cache | Decode ms/token, KV cache | Decode speed-up | Decode tok/s, cache | Peak GB, no cache / cache | Cache at end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 50 ± 6 | 58.5 | 50.6 ± 6.8 | 1.2× | 20.0 ± 2.6 | 2.90 / 2.90 | 5.2 MB |
| 512 | 150 ± 1 | 163.6 | 40.6 ± 0.3 | 4.0× | 24.6 ± 0.2 | 2.95 / 2.94 | 15.7 MB |
| 1024 | 327 ± 0 | 346.4 | 38.7 ± 0.2 | 9.0× | 25.9 ± 0.2 | 3.05 / 3.04 | 29.7 MB |
| 2048 | 808 ± 3 | 874.7 | 43.1 ± 0.3 | 20.3× | 23.2 ± 0.1 | 3.41 / 3.38 | 57.7 MB |

How to read it:

- **Without the cache, each new token costs about one prefill.** Decode ms/token tracks prefill
  time and grows with the prompt (58 → 875 ms). With the cache it stays flat at about 40 ms
  whatever the prompt length, so the speed-up grows with the prompt: 1.2× at 128 tokens, 20× at 2048.
- **Cached decode is bound by memory and launch overhead, not compute.** Each step reads all
  3 GB of fp16 weights (≈ 16 ms at the GPU's memory bandwidth) and launches 28 layers' worth of
  small kernels from Python. So one token costs about the same at any length.
- **Memory hardly differs between the two.** The weights (≈ 2.9 GB) dominate. The cache adds 28 KB
  per token (58 MB at 2 k tokens). Peak memory grows with input length because prefill
  materialises the `12 × L × L` fp32 attention scores, the same in both modes; avoiding that is
  what TRIA-10's tiled attention is for.
- The 128-token row has a larger spread (± 6.8 ms): it ran first, and the GPU clocks were still settling.
- For context, the library's `generate` with its own cache ran at 12.7 / 22.1 / 21.3 / 15.8 tok/s
  end to end (prefill included). Our cached decoder reaches 20.0 / 23.6 / 23.2 / 18.2 tok/s end to end,
  so it is in the same range. It's not meant as a speed contest: the point here is the cache, not kernels.

## Numerical differences

Our attention computes scores and softmax in fp32. The library uses fp16 SDPA kernels.
On a short prompt the largest logit difference is 0.055 and every argmax matches.
Cached and uncached decoding run *exactly the same* arithmetic per token in our code except
the attention's key length, so they can only differ through floating-point ordering. If a
greedy choice ever flips, `bench.first_divergence` logs the first divergent step and the
reference's top-1 minus top-2 logit gap, so a near-tie can be told apart from a real bug.
An fp32 re-test is not possible on this GPU: the 1.5B model is 6.2 GB in fp32, more than the 6 GB card.

## Tests

`task test` (CPU, tiny random Qwen2):
- forward equals library logits (atol 1e-4);
- prefill in two chunks equals prefill in one, which checks positions and mask offsets;
- cached = no-cache = library tokens at 3 length combos;
- cache byte count and `crop`.

`task test-gpu` (real 1.5B model, fp16): cached = own no-cache = library no-cache, token
for token, at 64/32, 256/64 and 1024/128 input/output tokens.
