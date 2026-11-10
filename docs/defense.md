# Defence prep (TRIA-15, rubric H3)

How to explain each part, where the code is, what the examiner may ask, and live changes to practise.
Numbers come from `results/` (paper tables); re-check them in `report/tables/numbers.tex` if the results change.

## 1. The 60-second pitch

- **Task:** one patient note plus one trial criterion goes in; the model outputs a JSON label (4 options per criterion type) and the ids of the sentences that support it.
- **Model:** Qwen2.5-1.5B-Instruct, QLoRA fine-tuned on the expert annotations only, with a patient-grouped 37/8/8 split (seed 42).
- **Test results:** accuracy 0.039 → **0.709**, macro-F1 0.077 → **0.566**, evidence micro-F1 0.157 → **0.579**, 0 parse errors.
- **Our own inference:**
  - **KV cache:** identical tokens to full recomputation, up to 20× faster decoding at 2,048 tokens.
  - **Tiled online-softmax attention:** the same result to 1.8e-6 in fp32, about 300× less memory at 8,192 tokens.
  - **Speculative decoding:** identical output, but slower here; the draft costs 85% of a target step.
- **App:** FastAPI + Next.js. It shows evidence highlighting, a rule-based preliminary outcome and live benchmarks.

## 2. Data and evaluation: likely questions

- **Why split by patient?** The same patient appears in many rows. A row-level split would leak that patient's note into test (`src/data/split.py`, manifest `results/split_manifest.json`).
- **Why drop the GPT-4 columns?** They are another model's answers. Training on them would copy GPT-4, and evaluating against them would score agreement with GPT-4, not with experts (`KEEP` allow-list in `src/data/load.py`).
- **Why macro-F1?** The labels are very imbalanced: *not excluded* dominates. Accuracy rewards always guessing it; macro-F1 weights every label equally.
- **Empty evidence?** Gold empty and prediction empty counts as a correct abstention, excluded from the macro average (`src/eval/metrics.py`, `docs/domain/trialmatch/eval.md`).
- **"Your friend got 73%."** With n = 206 the 95% interval is about ±6 points. We selected on macro-F1; run D had higher accuracy on validation (0.762) but lower macro-F1.
- **Repetition penalty?** `generate` applied Qwen's default of 1.1 during evaluation. We measured pure greedy on validation: no label changed (`report/tables/penalty.tex`).

## 3. Fine-tuning (`src/finetune/train.py`)

- **4-bit base:** NF4 + double quantisation, bf16 compute; `prepare_model_for_kbit_training` turns on gradient checkpointing.
- **LoRA:** r16, α32, dropout 0.05 on q, k, v, o, gate, up and down projections; 18.5 M trainable parameters.
- **`encode()`:** prompt tokens get label −100, so the loss is computed on the answer tokens only.
- **`label_weights()`:** inverse-sqrt frequency per (type, label), mean 1, capped at 5. This was run C, the winner.
- **Selection:** by validation macro-F1 per epoch; the test split is never used. Validation loss bottomed at epoch 2 while macro-F1 kept rising, so loss was the wrong selection signal.

## 4. KV cache (`src/kvcache/model.py`, `src/kvcache/decode.py`)

| Lines | What to say |
| --- | --- |
| model.py 15–30 | `KVCache`: per layer, preallocated K/V buffers `[2 kv heads, max_len, 128]` plus `length`. `write` copies new keys into `[start:end]`; `crop(n)` only moves `length`, which is how speculative decoding rolls back. |
| 33–36 | Size = 2 (K, V) × 28 layers × 2 heads × 128 × length × 2 bytes = 28 KB/token. GQA is why it's small. |
| 61–62, 68–71 | RoPE: `inv_freq = θ^(−2i/d)`, θ = 1e6. Positions come from the cache length. |
| 77–78 | **The key line:** new tokens sit at absolute positions `cache.length … cache.length+T−1`. Rotating at 0 instead would break everything. |
| 87 | RoPE is applied to q and k *before* k is stored, so cached keys never need re-rotating. |
| 88 | Store the new K/V and get back all keys so far. |
| 89 | GQA: repeat the 2 KV heads ×6 to meet the 12 query heads. |
| 90 | Attention with `q_offset = start`: query i may see keys 0…start+i (mask in `src/attention/reference.py: allowed`). |
| 93 | Advance the cache length once, after all layers. |
| decode.py `greedy_cached` | Prefill once with the whole prompt, then feed one token per step. `greedy_no_cache` rebuilds an empty cache every step (the reference). |

- **Proof it works:** `tests/test_kvcache.py`: chunked prefill = full prefill; cached = no-cache = library tokens. On the GPU, real-model equality at 64/32, 256/64, 1024/128.
- **Why faster?** Without the cache each step re-processes L+t tokens, so the cost grows with the prompt. With the cache each step processes 1 token against stored keys: about 40 ms regardless of length, because decode is bound by reading the weights.

## 5. Tiled attention (`src/attention/tiled.py`)

| Lines | What to say |
| --- | --- |
| 24–29 | For each query block: running max `m = −∞`, sum `l = 0`, output `acc = 0`, per row. |
| 30–33 | **Causal block skip:** if a key block starts after the last position this query block may see, stop. That roughly halves the work. |
| 34–37 | Scores for one tile only, `Br × Bc`; future keys and padding masked to −∞. |
| 38–39 | New max; `m_safe` guards a row with no visible key yet (avoids `−∞ − −∞ = NaN`). |
| 40–41 | `p = exp(s − m_new)`; `alpha = exp(m_old − m_new)` rescales everything accumulated under the old max. |
| 42–44 | `l = α·l + Σp`, `acc = α·acc + p·V`. |
| 45 | Output `acc / l`; a fully masked row returns 0. |

- **Why it's exact:** after any number of blocks, `acc/l` equals the softmax over the keys seen so far. Multiplying by `exp(m_old − m_new)` converts the old sums to the new max exactly.
- **Why slower than PyTorch:** a Python loop launches many small kernels, and every intermediate goes to GPU main memory. A fused kernel (FlashAttention) keeps a tile in on-chip SRAM and does it in one launch.
- **Why less memory:** never more than one tile of scores, `O(L)` instead of `O(L²)`. At 8,192 tokens: 9,280 MB vs 29 MB.
- **Errors:** 1.8e-6 (fp32) and 4.9e-4 (fp16) over 144 configurations. That's closer than PyTorch's fp16 SDPA (2.0e-3), because we accumulate in fp32.

## 6. Speculative decoding (`src/speculative/decode.py`)

| Lines | What to say |
| --- | --- |
| 33–34 | Prefill both models; the target's first token is emitted (same as plain greedy). |
| 39 | Never draft past the length limit. |
| 42–46 | Draft feeds whatever its cache is missing (1 or 2 tokens), then proposes γ tokens greedily. |
| 50–51 | **One target pass** over `[pending, d1…dγ]`, returning the target's choice after each prefix. |
| 54–57 | Accept while `guess[i] == choice[i]`; then emit the target's own token (a correction, or a bonus if all γ matched). |
| 62–63 | **Rollback:** crop both caches to the emitted length, so rejected guesses vanish. |

- **Why identical to greedy:** every emitted token is the target's argmax given exactly its prefix.
- **Why slower here:** cost scales with the number of layers in our runner (24 vs 28), so a draft token costs 40 ms against 47 ms for the target. Cost model: speed-up ≈ tokens per pass / (γ·c + 1) with c = 0.85. With c = 0.1, the same acceptance would give about 2×.
- **The one mismatch:** a free-text prompt hit an exact fp16 tie (gap 0.0000). The 1-token and multi-token kernels round differently.

## 7. Live changes to practise (each takes under 2 minutes)

1. **Change γ.** Two places:
   - In the app: Screen → **Advanced** → Speculative → γ − / +, then **Check eligibility**. The row detail shows acceptance and γ.
   - On Benchmarks → Speed: set γ, then **Compare the three methods**.

   **Expected:** bigger γ gives more tokens per pass but lower acceptance and a lower speed-up; the answer stays identical.
2. **Change the tile size.** Benchmarks → Memory → **Block size for tiling** 32 / 64 / 128 / 256 → **Compare attention methods**. **Expected:** small blocks are slower (more launches) but use less memory. The result never changes (look at the error column).
3. **Change a label rule.** `src/app/screening.py`, `aggregate()` lines 73–82.
   - **Example:** treat "not applicable" as needs review.
   - **How:** add `or r["label"] == "not applicable"` to the `unknown` condition on line 76.
   - **Check:** run `uv run pytest -q tests/test_app.py`; one outcome test will fail, which shows the rule is tested. Update its expectation, restart `task app` and screen again.
4. **Change the cache code and watch the tests catch it.** In `src/kvcache/model.py` line 78, change `torch.arange(start, …)` to `torch.arange(0, …)`. Run `uv run pytest -q tests/test_kvcache.py`: the chunked-prefill and equality tests fail. Undo it.
5. **Run an unseen case.** Paste a new note and type 3–4 criteria (one inclusion, one exclusion, one the note can't answer). Point out the cited sentences, the "can't tell" answer, and the outcome rule.

## 8. Honest limitations to say before being asked

- Small, narrow training data (603 rows, 37 patients, ER vignettes).
- No *excluded* rows in test.
- Expert labels are inconsistent about unmentioned conditions.
- Our attention and speculative code show correctness and memory behaviour, not speed.
- The outcome is a rule, not a prediction.
- It's an educational prototype only.
