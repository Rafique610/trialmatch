# Speculative decoding (TRIA-11)

Code: `src/speculative/decode.py` (`speculative_greedy`), `src/speculative/bench.py`,
`tests/test_speculative.py`. Results: `results/bench/speculative.json`, `speculative.png` (`task bench`).
Built on our own runner and KV cache (TRIA-9). It does not use `generate(assistant_model=…)`, which
appears only as a cross-check.

Target: Qwen2.5-1.5B-Instruct (28 layers), base or with the fine-tuned adapter merged.
Draft: Qwen2.5-0.5B-Instruct (24 layers). They share the same tokenizer (TRIA-2, `results/tokenizer_compat.json`).

## Algorithm (greedy)

Between rounds, every emitted token is in `out`. The last one (*pending*) has not been run
through the target yet, so the target cache covers `prompt + out[:-1]`.

1. **Draft:** feed whatever the draft cache is missing (1 token, or 2 after a fully accepted
   round), then propose γ tokens greedily, one forward each, with the draft's own cache.
2. **Verify in one pass:** the target runs `[pending, d1 … dγ]` (γ + 1 tokens) in **one** forward
   pass, with positions continuing from its cache, and returns its greedy choice after each prefix: `p0 … pγ`.
3. **Accept:** keep `d1 … dk` while `d_i = p_(i−1)`. Then emit `p_k`: the target's correction at
   the first mismatch, or a **bonus** token if all γ guesses matched. Every round emits at least
   one token, so it always makes progress.
4. **Roll back:** `KVCache.crop` cuts the target cache to `prompt + emitted − 1` and the draft
   cache to the same length (or less), so rejected guesses are forgotten and positions stay
   consistent. EOS and the length limit cut the round's tokens short.

Every emitted token is the target's argmax given exactly the tokens before it, so the output
equals the target's plain greedy output. The speed-up can only come from step 2: one target pass
emits up to γ + 1 tokens.

## Correctness

| Check | Result |
| --- | --- |
| CPU, tiny random models, fp32: draft identical / noisy / unrelated × γ ∈ {1, 2, 4, 6} × 5 prompts, 40 tokens | 60/60 identical to target greedy; identical draft accepts 100% (bonus path); unrelated draft still emits ≥ 1 token per round |
| CPU: EOS mid-round | stops exactly where greedy stops |
| GPU, real pair, 5 task prompts × γ ∈ {2, 4, 6} | identical to our cached greedy (TRIA-9) |
| Benchmark, task prompts, base and fine-tuned target, γ ∈ {1, 2, 3, 4, 6, 8} | 20/20 identical at every γ |
| Benchmark, free-text summaries (8 notes, 128 tokens) | 8/8 at γ = 2, 3, 8; 7/8 at γ = 1, 4, 6 (one exact tie, below) |
| Library cross-check, 5 prompts, γ = 4 | ours = library assisted = library greedy, 5/5 |

**The one mismatch is an exact tie.** On one patient note, at step 91 of 107, the target's fp16
logits for " given" and " provided" are *exactly equal* (top-1 minus top-2 = 0.0000).
- Plain decoding runs 1 token per forward; verification runs γ + 1 tokens per forward.
- The fp16 matrix kernels for those two shapes round differently, so the tie breaks one way in
  one mode and the other way in the other.
- Nothing differs before the tie, and every other choice in every run matches.

In fp32 (the CPU tests) every case is exact. The real pair cannot be re-run in fp32 on this
GPU (1.5B + 0.5B in fp32 ≈ 8 GB > 6 GB).

**Note on the library cross-check.** Qwen's `generation_config.json` sets `repetition_penalty: 1.1`,
and `generate` applies it unless told otherwise. The cross-check passes `repetition_penalty=1.0`
so that greedy is compared with greedy. (An earlier run without it matched only 2/5, which is how this was found.)

## Benchmark

RTX 3050 6 GB laptop GPU, fp16, 1 warm-up and 3 timed repeats per setting, synchronised timings.
"Speed-up" = plain cached greedy time (TRIA-9 decoder, target only, same prompts) ÷ speculative time.
Peak memory with both models resident: **3.85 GB** (plain greedy: 3.84 GB; the 0.5B draft
adds ≈ 1 GB of weights but stays resident in both runs here).

**Task prompts (20 screening prompts, ≈ 339 tokens, JSON answer), fine-tuned target.**
Plain greedy: 19.8 tok/s, 46.5 ms per target step.

| γ | Acceptance | Accepted / round | Tokens / target pass | Draft ms / round | Verify ms / round | tok/s | Speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.87 | 0.87 | 1.74 | 40 | 47 | 18.4 | 0.93× |
| 2 | 0.77 | 1.54 | 2.15 | 78 | 47 | 16.5 | 0.83× |
| 3 | 0.81 | 2.43 | 2.67 | 117 | 47 | 16.3 | 0.82× |
| 4 | 0.74 | 2.98 | 3.11 | 159 | 47 | 15.8 | 0.80× |
| 6 | 0.65 | 3.90 | 3.46 | 229 | 50 | 13.6 | 0.69× |
| 8 | 0.59 | 4.75 | 3.73 | 309 | 49 | 11.7 | 0.59× |

**Task prompts, base target** (longer, chattier answers: 583 tokens vs 339): acceptance 0.79–0.91,
speed-up 0.97× (γ = 1) → 0.82× (γ = 8). **Free-text summaries, base target** (8 notes, ≤ 128 tokens):
acceptance 0.48–0.82, speed-up 0.96× → 0.62×. Full rows are in `speculative.json`.

## Why it is slower here (measured, not assumed)

The draft guesses well: 74–91% of guesses are accepted at γ ≤ 4. And the verify pass works as
promised: checking γ + 1 tokens costs **47–50 ms at every γ, the same as one plain decode step**
(46.5 ms), because the step is bound by reading weights and launching kernels, not by arithmetic.
That part works.

The problem is the draft's cost. With our runner, a step costs about the same per **layer**
whatever the layer's width, because of 28 vs 24 Python-launched layers of small kernels. So a
0.5B draft token costs **≈ 40 ms against 47 ms for a 1.5B target token**: the draft is 85% as
expensive as the target, not the ≈ 33% its size suggests.

Cost model per round, in target-step units: `γ·c + 1`, with `c = 40/46.5 = 0.85`. It yields
`tokens per pass` tokens. With the measured tokens per pass, `speed-up ≈ tokens/(γ·c + 1)`:

| γ | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| model (c = 0.85) | 0.94× | 0.79× | 0.75× | 0.70× | 0.56× | 0.48× |
| measured | 0.93× | 0.83× | 0.82× | 0.80× | 0.69× | 0.59× |
| same acceptance, if c were 0.1 | 1.58× | 1.79× | 2.05× | **2.22×** | 2.16× | 2.07× |

The model gets the trend right. The measured numbers sit a little above it because the draft's
per-token cost drops slightly inside a round. With a draft costing 10% of the target, as with a
7B target and a 0.5B draft or a fused-kernel runtime, the same acceptance rates would give
about 2×. The bottleneck is the target/draft cost ratio on this setup, not the idea.

Other measured factors:
- **The fine-tuned target lowers acceptance** (0.59–0.87 vs 0.79–0.91 for the base target).
  The fine-tuned model writes our compact JSON, which the untuned draft doesn't expect: field
  order, label wording, short evidence lists.
- **Short outputs:** task answers are ≈ 17 tokens, so prefill and the first token are a large
  fixed share of the time, and there are only a few rounds to gain in.
- **Memory:** both models must be resident (+1 GB). That fits on 6 GB here, but it would rule
  out a bigger draft.

**Choice for the app:** plain cached greedy is the default. Speculative decoding (γ = 1, 0.93×) is
offered as a mode, with its acceptance stats shown live, as the assignment asks.
