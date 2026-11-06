"""Prove the draft and target models share one tokenizer (required for speculative decoding).

Greedy speculative decoding compares draft and target token *ids*, so both
tokenizers must map every string to the same ids and back. We compare the full
vocabularies, special tokens, the chat template, and the ids produced on
clinical sample text.
Run: uv run python -m src.speculative.check_tokenizers
"""

import json

from transformers import AutoTokenizer

from src.config import settings

SAMPLES = [
    "A 58-year-old woman with type 2 diabetes (HbA1c 8.2%) and stage III NSCLC.",
    "Exclusion: prior malignancy within 5 years, except basal cell carcinoma.",
    '{"label": "not enough information", "evidence_ids": [0, 3]}',
    "eGFR < 30 mL/min/1.73m², ECOG ≥ 2, pregnant or breast-feeding.",
]


def main() -> None:
    draft = AutoTokenizer.from_pretrained(settings.draft_model_id)
    target = AutoTokenizer.from_pretrained(settings.target_model_id)
    chat = [{"role": "user", "content": SAMPLES[0]}]
    checks = {
        "same_vocab": draft.get_vocab() == target.get_vocab(),
        "same_len": len(draft) == len(target),
        "same_special_tokens": draft.all_special_ids == target.all_special_ids,
        "same_eos": draft.eos_token_id == target.eos_token_id,
        "same_chat_template": draft.chat_template == target.chat_template,
        "same_ids_on_samples": all(draft.encode(s) == target.encode(s) for s in SAMPLES),
        "round_trip_ok": all(target.decode(draft.encode(s)) == s for s in SAMPLES),
        "same_chat_ids": draft.apply_chat_template(chat, add_generation_prompt=True)
        == target.apply_chat_template(chat, add_generation_prompt=True),
    }
    report = {
        "draft": settings.draft_model_id,
        "target": settings.target_model_id,
        "tokenizer_len": len(target),
        "eos_token": target.eos_token,
        "checks": checks,
        "compatible": all(checks.values()),
        "sample_ids": {s: target.encode(s) for s in SAMPLES},
    }
    out = settings.results_dir / "tokenizer_compat.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"tokenizer size {len(target)}, eos {target.eos_token!r} -> compatible={report['compatible']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
