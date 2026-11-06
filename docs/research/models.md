# Model candidates

Facts come from each model's `config.json` and its Hugging Face Hub metadata,
pulled by `uv run python -m src.research.model_facts` into
`results/model_candidates.json`. Llama 3.2 repos are gated, so their
`config.json` was read from the ungated `unsloth/` copies (same file); the
licence and gate status come from the official `meta-llama/` repos.

Hardware: RTX 3050 Laptop GPU, **6 GB VRAM** (Windows and the display also use
some of it), 16 GB RAM.

## Architecture (all decoder-only, RoPE, SwiGLU MLP, RMSNorm, grouped-query attention)

| Model | Params | Layers | Q / KV heads | Head dim | Hidden | Vocab | Context | Licence | Gated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 0.49 B | 24 | 14 / 2 | 64 | 896 | 151,936 | 32,768 | Apache-2.0 | no |
| **Qwen2.5-1.5B-Instruct** | **1.54 B** | 28 | 12 / 2 | 128 | 1,536 | 151,936 | 32,768 | **Apache-2.0** | no |
| Qwen2.5-3B-Instruct | 3.09 B | 36 | 16 / 2 | 128 | 2,048 | 151,936 | 32,768 | Qwen Research¹ | no |
| Llama-3.2-1B-Instruct | 1.24 B | 16 | 32 / 8 | 64 | 2,048 | 128,256 | 131,072 | Llama 3.2 Community | yes |
| Llama-3.2-3B-Instruct | 3.21 B | 28 | 24 / 8 | 128 | 3,072 | 128,256 | 131,072 | Llama 3.2 Community | yes |

All five tie the input embedding and output head (`tie_word_embeddings: true`).

¹ The Hub lists this licence as "other". Its `LICENSE` file grants use "for
non-commercial purposes only" and defines that as "research or evaluation
purposes only". This assignment qualifies, so the licence does not exclude
the 3B model here; memory does (ADR 0004).

## Memory

| Model | fp16 weights | 4-bit (NF4) weights | KV cache per 1k tokens (fp16) | LoRA r=16 trainable params |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 0.92 GB | 0.42 GB | 12 MB | 8.8 M |
| **Qwen2.5-1.5B** | **2.88 GB** | **1.04 GB** | **28 MB** | **18.5 M (1.2 %)** |
| Qwen2.5-3B | 5.75 GB | 1.87 GB | 36 MB | 29.9 M |
| Llama-3.2-1B | 2.30 GB | 0.94 GB | 32 MB | 11.3 M |
| Llama-3.2-3B | 5.98 GB | 2.05 GB | 112 MB | 24.3 M |

Formulas (computed from the configs, not measured):

- fp16 weights = params × 2 bytes.
- NF4 weights = (params − embedding) × 0.5 bytes + embedding × 2 bytes.
  bitsandbytes leaves the embedding / tied output head in 16-bit.
- KV cache = 2 (K and V) × layers × KV heads × head dim × 2 bytes × tokens.
- LoRA r=16 on q, k, v, o, gate, up, down: r × (in + out) per matrix, × layers.

### QLoRA training estimate for the 6 GB card (batch 1, 1,024 tokens, gradient checkpointing)

Estimate only. TRIA-7 measures the real peak with `torch.cuda.max_memory_allocated`.

| Part | Qwen2.5-1.5B | Qwen2.5-3B |
| --- | --- | --- |
| NF4 weights | 1.04 GB | 1.87 GB |
| LoRA params + grads + Adam states (16 B/param, fp32) | 0.30 GB | 0.48 GB |
| Output logits, fp32, 1,024 × 151,936 (+ their gradient) | ≈ 1.2 GB | ≈ 1.2 GB |
| Checkpointed activations + one recomputed layer | ≈ 0.3 GB | ≈ 0.5 GB |
| CUDA context / allocator overhead | ≈ 0.5 GB | ≈ 0.5 GB |
| **Total** | **≈ 3.3 GB** | **≈ 4.6 GB** |

The 1.5B model leaves ~2.5 GB headroom. The 3B model would fit only with
little room for longer notes, and Windows already holds part of the 6 GB.
Qwen's large 152k vocabulary makes the logits, not the weights, the biggest
activation cost, so longer sequences hurt more than a bigger model.

### Inference with both models resident (speculative decoding)

The 1.5B target in fp16 (2.88 GB) plus the 0.5B draft in fp16 (0.92 GB) is
about 3.8 GB before KV caches (28 + 12 MB per 1k tokens). This fits in 6 GB
without quantisation. That matters for Tasks 3 and 5: fp16 keeps the
cached/uncached and speculative/greedy token comparisons free of 4-bit
dequantisation noise.

## Draft model compatibility

Speculative decoding compares draft and target token ids, so the tokenizers
must be identical. `uv run python -m src.speculative.check_tokenizers`
(→ `results/tokenizer_compat.json`) passes all 8 checks for Qwen2.5-0.5B vs
1.5B Instruct:

- identical vocabulary
- identical tokenizer length (151,665)
- identical special tokens and EOS (`<|im_end|>`)
- identical chat template
- identical ids on clinical sample text
- lossless round trip
- identical chat-formatted ids

The config `vocab_size` (151,936) is larger than the tokenizer (151,665)
because Qwen pads its embedding matrix. Both models pad the same way, so
their logits have the same width.

Llama-3.2-1B can draft for Llama-3.2-3B, but that pair needs the 3B target,
which is the tight option on this GPU.

## Choice

**Target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct.** See ADR 0004.
