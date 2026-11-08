# Attention: reference vs tiled online softmax (TRIA-10)

Code: `src/attention/reference.py`, `src/attention/tiled.py`, `src/attention/bench.py`,
`tests/test_attention.py`. Pure PyTorch, no fused kernel. Results: `results/bench/attention.json`,
`attention_errors.json`, `attention.png` (`task bench`).

## Reference

`attention(q, k, v, causal, q_offset, key_mask)` builds the full score matrix
`S = QKᵀ/√d` in fp32 and sets masked entries to `−inf`. Then it subtracts each row's max before
`exp`, so every exponent is ≤ 0 and nothing overflows, normalises and multiplies by V.

- **Causal mask:** query *i* sits at absolute position `q_offset + i`, so it works after a KV cache too (TRIA-9).
- **Padding:** `key_mask`, True = keep.
- **Rows where every key is masked** return 0 instead of NaN: the max is replaced by 0 and the
  sum by 1 for those rows.

The KV-cache runner (`src/kvcache/model.py`) now calls this same reference, so attention has
one source of truth.

## Tiled with an online softmax

For each block of `Br` queries, stream over blocks of `Bc` keys, keeping three running values per query row:

```
m  = running max of scores seen so far        (starts at −inf)
l  = running Σ exp(score − m)                 (starts at 0)
O  = running Σ exp(score − m) · V             (starts at 0)

for each key block:
    S      = Q_blk K_blkᵀ / √d   (masked → −inf)
    m_new  = max(m, rowmax(S))
    P      = exp(S − m_new)
    α      = exp(m − m_new)       # rescales everything accumulated under the old max
    l      = α·l + rowsum(P)
    O      = α·O + P V_blk
    m      = m_new
output = O / l
```

Why it's exact: after any number of blocks, `O/l` equals softmax-weighted V over the keys seen
so far. When a new block raises the max, multiplying the old `l` and `O` by
`exp(m_old − m_new)` puts them in the new frame, and the identity is exact. Only one `Br × Bc` tile of
scores exists at a time, never `Tq × Tk`.

- **Causal block skip:** a key block that starts after the last position any query in the
  current block may see is skipped, along with every block after it (`break`). For a
  square causal problem that roughly halves the work.
- **Edge cases:** a row with no visible key yet has `m = −inf`; it is replaced by 0 inside the
  `exp` so `−inf − (−inf)` never happens, `α = exp(−inf) = 0`, and an all-masked row ends with `l = 0` → output 0.
- Block sizes don't need to divide the sequence length (the last block is just shorter).

## Correctness

`tests/test_attention.py` (CPU), 70 tests plus 1 GPU test:

| Test | Grid | Tolerance (max abs) |
| --- | --- | --- |
| tiled vs reference vs `F.scaled_dot_product_attention` | seq {1, 7, 64, 129, 512} × d {32, 64, 128} × causal/non-causal × fp32/fp16, batch 2, 3 heads, blocks 48×40 | fp32 1e-5, fp16 2e-3 |
| block sizes | (1,1), (16,64), (64,16), (129,129), (512,512) at seq 129 | 1e-5 |
| queries after a cache | 1, 5, 37 queries after 99, 95, 63 cached keys | 1e-5 |
| padding mask + fully masked rows | partial padding vs SDPA; all-masked → exactly 0, no NaN | 1e-5 |
| large scores | q × 300 (scores in the thousands) | finite, 1e-4 |
| in the real model (GPU) | tiled in all 28 layers, 256-token prompt, 32 greedy tokens | identical tokens |

The measured error grid (`attention_errors.json`, GPU, 144 configs: seq up to 2048, both block
shapes, 4 heads):

| dtype | tiled vs reference, max / mean | torch SDPA vs reference, max / mean |
| --- | --- | --- |
| fp32 | 1.8e-6 / 2.7e-8 | 2.5e-6 / 7.5e-8 |
| fp16 | 4.9e-4 / 2.4e-8 | 2.0e-3 / 1.9e-5 |

The tiled version stays *closer* to the fp32 reference than PyTorch's fp16 SDPA does. It
accumulates in fp32 and only rounds once at the end. The fp16 maximum (4.9e-4) is one fp16
rounding step at the output's magnitude.

**Inside the model** (bonus): the tiled kernel replaced attention in all 28 layers of
Qwen2.5-1.5B on a 512-token validation prompt. Prefill logits differ by at most 0.13, every
argmax agrees, and 64 greedy tokens are identical. The 0.13 is fp16 rounding of each layer's
attention output, compounded over 28 layers; it never changes a choice.

## Benchmark

1 sequence, 12 heads, d = 128 (Qwen's shape), causal, fp16, 1 warm-up and 5 timed repeats, CUDA-synchronised,
RTX 3050 6 GB laptop GPU. "Extra MB" = peak memory above the q, k, v inputs.

| Seq | Reference ms / MB | Tiled 64 ms / MB | Tiled 128 ms / MB | SDPA math ms / MB | SDPA mem-efficient ms / MB |
| --- | --- | --- | --- | --- | --- |
| 256 | 4.6 / 9 | 14.6 / 3 | 5.4 / 6 | 5.9 / 13 | 0.38 / 0.8 |
| 512 | 18.5 / 36 | 47.5 / 4 | 18.4 / 7 | 4.9 / 40 | 0.36 / 1.5 |
| 1024 | 12.8 / 145 | 178 / 5 | 49.6 / 8 | 11.9 / 136 | 2.4 / 3 |
| 2048 | 40.6 / 580 | 606 / 8 | 157 / 11 | 42.7 / 496 | 1.8 / 6 |
| 4096 | 153 / 2 320 | 2 277 / 14 | 645 / 17 | 136 / 1 888 | 7.3 / 12 |
| 8192 | 5 087 / **9 280** | 9 932 / **26** | 2 652 / **29** | 1 464 / 7 360 | 26.9 / 24 |

SDPA's flash backend is **not available**: PyTorch's Windows wheels are built without it
("No available kernel"). Mem-efficient is the fused tiled kernel we can compare against.

Block-size sweep at seq 2048: 16 → 9 529 ms / 6.5 MB, 32 → 2 268 / 7.0, 64 → 603 / 8.3,
128 → 188 / 11.3, 256 → 57 / 21.1, 512 → 30 / 60.4 MB.

### What the numbers show

1. **Memory is the win.** The reference and SDPA-math need `12 × L × L` scores: 580 MB at 2 k,
   9.3 GB at 8 k. Tiled needs 26–29 MB at 8 k, about **350× less**, and grows linearly, not quadratically.
   At 8 k the reference needs more than the card's 6 GB. Windows quietly spilled it into shared
   system RAM over PCIe, which is why it jumps from 153 ms to 5 s (33×) between 4 k and 8 k.
   On Linux it would just run out of memory. The tiled version never gets near that.
2. **Our Python tiling is slower than everything fused.** At 2 k: tiled-128 157 ms vs reference
   41 ms vs mem-efficient 1.8 ms. Reasons:
   - **Kernel launches:** every tile costs about 10 small CUDA kernels launched from a Python loop.
     At 2 k with 64-blocks that's about 500 tiles × 10 launches, and each launch carries
     microseconds of overhead the GPU spends idle.
   - **No control of fast on-chip memory (SRAM):** a fused kernel keeps Q, K, V, `m`, `l`, `O`
     for a tile in shared memory and registers for the whole inner loop. Ours writes every
     intermediate (`S`, `P`, `α`, the new `O`) back to GPU main memory (HBM) and reads it again,
     a round trip per operation.
   - **Small matmuls:** a 64 × 128 × 64 matmul can't fill the GPU, so tensor cores sit mostly idle.
   - **No fusion of exp, sum and rescale:** these are separate passes over memory.

   SDPA-mem-efficient does the same tiling *inside one kernel*, so it gets both the low memory
   **and** the speed: 8 k in 27 ms with 24 MB. That's the point of FlashAttention-style kernels:
   the algorithm saves memory, and the fusion is what makes it fast.
3. **Block size trades speed for memory.** Bigger tiles mean fewer launches and bigger matmuls:
   16 → 512 is 317× faster, but tile memory grows with `Br × Bc` (6.5 → 60 MB). In real kernels
   the block size is set by how much SRAM each GPU core has.
4. **Short sequences:** below about 1 k the reference is fine. Its score matrix is small, and the
   single big matmul beats any loop.

## In the app

The app keeps the library's / reference attention for speed. The tiled kernel is a
correctness and memory demonstration: `Runner.forward(..., attn_fn=tiled)` runs the whole
model with it and gives identical tokens.
