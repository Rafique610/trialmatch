# Literature review

Each source: what it says, its limitation, and how this project uses it.
Numbers quoted from papers are those papers' own results. Our results come
only from `results/`.

## 1. Clinical eligibility reasoning

### [S1] Jin et al., *Matching patients to clinical trials with large language models*, Nature Communications 15:9074, 2024 (TrialGPT). doi:10.1038/s41467-024-53081-z

- **Says:** a zero-shot GPT-4 pipeline in three stages. TrialGPT-Retrieval
  finds candidate trials, TrialGPT-Matching judges each criterion and cites
  patient sentences, and TrialGPT-Ranking aggregates the judgements. On 1,015
  expert-checked patient–criterion pairs, criterion-level accuracy was 87.3 %.
  This is the dataset we must use.
- **Limitation:** it relies on a proprietary, paid API, which the brief forbids
  here. It is also zero-shot, so it never adapts the model to the annotation
  style. Its explanations are GPT-4 text, which we must not train on.
- **Used for:** the task definition (criterion-level label + evidence sentence
  ids), the label sets, and the sentence-numbered patient input. The trial-level
  aggregation idea becomes our explicit outcome rule (ADR 0002). We replace
  GPT-4 with a small, locally fine-tuned open model.

### [S2] Wornow et al., *Zero-Shot Clinical Trial Patient Matching with LLMs*, arXiv:2402.05125, 2024

- **Says:** an LLM checks free-text patient histories against free-text
  inclusion criteria with no task-specific training. It reports state-of-the-art
  results on the 2018 n2c2 cohort-selection benchmark. Clinicians judged most of
  its justifications coherent.
- **Limitation:** it covers inclusion criteria only, uses large hosted models,
  and is zero-shot. It doesn't test whether a small fine-tuned model can do the same.
- **Used for:** support for criterion-by-criterion prompting and
  evidence-backed justification. Also a contrast point: we ask whether
  supervised adaptation of a 1.5B model closes the gap on our split.

## 2. Transformer architecture

### [S3] Vaswani et al., *Attention Is All You Need*, NeurIPS 2017, arXiv:1706.03762

- **Says:** scaled dot-product attention, `softmax(QKᵀ/√d)V`, with causal masking for decoders.
- **Limitation:** time and memory grow quadratically with sequence length, and
  generating one token at a time re-reads all past keys and values.
- **Used for:** the reference attention in TRIA-10, and the baseline the KV
  cache and tiled attention improve on.

### [S4] Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023, arXiv:2305.13245

- **Says:** grouped-query attention shares each key/value head across several
  query heads, shrinking the KV cache with little quality loss.
- **Limitation:** quality depends on how many KV groups are kept. It must be
  built into the model, not added afterwards.
- **Used for:** explaining why Qwen2.5-1.5B's cache is small: 2 KV heads
  versus 12 query heads gives 28 MB per 1k tokens, against 112 MB for
  Llama-3.2-3B (`docs/research/models.md`). The TRIA-9 decoder must repeat KV
  heads correctly.

### [S5] Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864, 2021

- **Says:** rotary embeddings encode position by rotating queries and keys, so
  attention depends on relative position.
- **Limitation:** quality degrades past the trained context unless the scaling is adjusted.
- **Used for:** TRIA-9 position handling. A cached decode step must be rotated
  at its true position (`past_length + t`), not at 0.

### [S6] Qwen Team, *Qwen2.5 Technical Report*, arXiv:2412.15115, 2024

- **Says:** the Qwen2.5 family (0.5B–72B) is decoder-only with GQA, SwiGLU,
  RoPE and RMSNorm, and shares one tokenizer across sizes. Instruct versions
  are released under open licences.
- **Limitation:** it reports general and medical-adjacent benchmarks, not
  clinical eligibility. The small models' clinical knowledge is untested for our task.
- **Used for:** the model choice (ADR 0004). The shared tokenizer makes the
  0.5B model a valid draft for the 1.5B (proved in `results/tokenizer_compat.json`).

## 3. Supervised adaptation

### [S7] Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022, arXiv:2106.09685

- **Says:** freeze the pretrained weights and learn a low-rank update `BA`
  per weight matrix. This matches full fine-tuning on many tasks while
  training a small fraction of the parameters.
- **Limitation:** rank and target modules are hyperparameters. Low rank can
  underfit tasks far from pretraining.
- **Used for:** the adapters in TRIA-7. Rank 16 on all attention and MLP
  projections is 18.5 M trainable parameters for Qwen2.5-1.5B (1.2 %).

### [S8] Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023, arXiv:2305.14314

- **Says:** keep the frozen base in 4-bit NormalFloat (NF4) with double
  quantisation and train LoRA adapters on top. Quality stays close to 16-bit
  fine-tuning at a fraction of the memory.
- **Limitation:** dequantising weights slows each step. The 4-bit base adds
  noise compared with the fp16 model used at inference.
- **Used for:** training on a 6 GB GPU (ADR 0005). NF4 weights for the 1.5B
  model are 1.04 GB, against 2.88 GB in fp16.

## 4. Efficient inference

### [S9] Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023, arXiv:2211.05102

- **Says:** splits generation into a compute-bound prefill phase and a
  memory-bandwidth-bound decode phase. The KV cache's size and layout dominate
  decode cost.
- **Limitation:** written for large TPU deployments, not one consumer GPU.
- **Used for:** the TRIA-9 benchmark design, which times prefill and decode
  separately and reports peak memory.

### [S10] Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023, arXiv:2309.06180

- **Says:** stores the KV cache in fixed-size blocks like virtual-memory pages
  to avoid fragmentation when serving many requests.
- **Limitation:** its gains come from batching many requests. For one request
  at a time on one GPU, a contiguous cache is simpler.
- **Used for:** a design alternative we considered and rejected for TRIA-9.
  It frames why cache growth matters.

### [S11] Milakov & Gimelshein, *Online normalizer calculation for softmax*, arXiv:1805.02867, 2018

- **Says:** softmax can be computed in one pass by keeping a running maximum
  and a running sum, rescaled whenever the maximum changes.
- **Limitation:** it is the normaliser only, not a full attention algorithm.
- **Used for:** the running-max / running-sum update at the heart of the
  TRIA-10 tiled attention.

### [S12] Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022, arXiv:2205.14135; Dao, *FlashAttention-2*, ICLR 2024, arXiv:2307.08691

- **Says:** tile Q, K and V into blocks that fit in on-chip SRAM and combine
  them with online softmax. The full N×N score matrix never touches main GPU
  memory. This is exact and much faster. Version 2 improves the work
  partitioning across the GPU.
- **Limitation:** the speed comes from a fused CUDA kernel. The same algorithm
  written as PyTorch loops gets the memory saving but not the speed, because
  every block becomes separate kernel launches and memory round-trips.
- **Used for:** the TRIA-10 algorithm, and the honest explanation of why our
  educational version is slower than PyTorch's fused `scaled_dot_product_attention`.

### [S13] Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023, arXiv:2211.17192; Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv:2302.01318, 2023

- **Says:** a small draft model proposes γ tokens and the large target model
  checks them all in one forward pass. The acceptance rule guarantees the same
  output distribution as the target alone. Under greedy decoding this reduces
  to "accept while argmaxes match".
- **Limitation:** speed-up depends on the acceptance rate and on how much
  cheaper the draft is. With a 3× size ratio and both models on one small GPU,
  the gain can vanish or turn into a slowdown.
- **Used for:** the TRIA-11 algorithm. Exact token equality with greedy target
  output is the correctness test, and acceptance rate and speed-up are the
  reported metrics.

## Comparison of alternatives

| Problem | Options considered | Chosen | Why |
| --- | --- | --- | --- |
| Matching method | Zero-shot prompting [S1, S2] · supervised fine-tuning [S7, S8] | Fine-tune, with zero-shot as the measured baseline | The brief requires fine-tuning and a before/after comparison. A 1.5B model is weak zero-shot. |
| Adaptation | Full fine-tune · LoRA fp16 · QLoRA | QLoRA [S8] | Full fine-tuning of 1.5B needs about 25 GB (weights, gradients and Adam states). LoRA in fp16 needs the 2.88 GB base plus activations. QLoRA needs about 3.3 GB in total (ADR 0005). |
| KV cache layout | Contiguous per-layer tensors · paged blocks [S10] | Contiguous | One request at a time, so paging buys nothing and costs complexity. |
| Attention speed-up | Fused kernel (FlashAttention, PyTorch SDPA) · own tiled implementation [S11, S12] | Own implementation for Task 4, benchmarked against SDPA | The brief needs our own blocked algorithm. Fused kernels are the speed reference. |
| Faster decoding | Speculative sampling [S13] · Medusa-style extra heads · quantised target | Greedy speculative decoding | Needs no extra training, the same-family draft exists, and it is exactly verifiable. |
| Model family | Qwen2.5 · Llama 3.2 | Qwen2.5 [S6] | Apache-2.0, ungated, a 0.5B same-tokenizer draft, and the smallest KV cache (ADR 0004). |
