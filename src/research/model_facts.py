"""Pull architecture, size and licence facts for the candidate models from Hugging Face.

Every number comes from the model's own config.json and Hub metadata; the memory
columns are arithmetic on those numbers (formulas in docs/research/models.md).
Run: uv run python -m src.research.model_facts
"""

import json

from huggingface_hub import HfApi, hf_hub_download

from src.config import settings

# Llama 3.2 repos are gated; unsloth publishes the same config.json ungated.
CANDIDATES = {
    "Qwen2.5-0.5B-Instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    "Qwen2.5-1.5B-Instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    "Qwen2.5-3B-Instruct": ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
    "Llama-3.2-1B-Instruct": ("meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"),
    "Llama-3.2-3B-Instruct": ("meta-llama/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-3B-Instruct"),
}
GB = 1024**3


def facts(official: str, mirror: str, api: HfApi) -> dict:
    cfg = json.loads(open(hf_hub_download(mirror, "config.json"), encoding="utf-8").read())
    info = api.model_info(mirror, files_metadata=False)
    official_info = api.model_info(official)
    params = info.safetensors.total
    hidden, layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim") or hidden // heads
    embed = cfg["vocab_size"] * hidden
    tied = cfg.get("tie_word_embeddings", False)
    # bitsandbytes 4-bit leaves embeddings / lm_head in 16-bit
    unquantised = embed if tied else 2 * embed
    return {
        "repo": official,
        "config_from": mirror,
        "licence": (official_info.card_data or {}).get("license"),
        "gated": bool(official_info.gated),
        "params": params,
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": cfg["vocab_size"],
        "tied_embeddings": tied,
        "context_window": cfg["max_position_embeddings"],
        "rope_theta": cfg.get("rope_theta"),
        "fp16_weights_gb": round(params * 2 / GB, 2),
        "nf4_weights_gb": round(((params - unquantised) * 0.5 + unquantised * 2) / GB, 2),
        "kv_cache_mb_per_1k_tokens_fp16": round(2 * layers * kv_heads * head_dim * 2 * 1024 / 1024**2, 1),
    }


def main() -> None:
    api = HfApi()
    rows = {name: facts(*repos, api) for name, repos in CANDIDATES.items()}
    out = settings.results_dir / "model_candidates.json"
    out.write_text(json.dumps(rows, indent=2))
    for name, r in rows.items():
        print(
            f"{name:24} params={r['params'] / 1e9:.2f}B L={r['layers']} kv={r['kv_heads']}/{r['heads']} "
            f"ctx={r['context_window']} lic={r['licence']} fp16={r['fp16_weights_gb']}GB "
            f"nf4={r['nf4_weights_gb']}GB kv/1k={r['kv_cache_mb_per_1k_tokens_fp16']}MB"
        )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
