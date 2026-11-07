"""Token-length statistics and the truncation report (brief Task 1: document truncation).

Lengths are measured with the target model's own tokenizer and chat template:
  prompt   = system + user turn + assistant header (what inference sees)
  training = prompt + gold JSON answer + end token (what fine-tuning sees)
If a training example exceeds MAX_LEN, the note would be cut from its last
sentence backwards (criterion and answer are never cut); the report lists every
gold evidence sentence that such a cut would remove.
Run: uv run python -m src.data.lengths   (needs `task data` first)
"""

import json
import statistics

from transformers import AutoTokenizer

from src.config import settings
from src.data.prompt import messages, target

MAX_LEN = 1024  # longest training example is 636 tokens; see docs/domain/trialmatch/prompt.md


def read_split(name: str) -> list[dict]:
    path = settings.data_dir / "processed" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def lengths(tok, rec: dict) -> tuple[int, int]:
    text = tok.apply_chat_template(messages(rec), add_generation_prompt=True, tokenize=False)
    prompt = tok(text, add_special_tokens=False)["input_ids"]
    answer = tok(target(rec) + tok.eos_token, add_special_tokens=False)["input_ids"]
    return len(prompt), len(prompt) + len(answer)


def kept_sentences(tok, rec: dict, max_len: int) -> int:
    """How many leading note sentences fit when the full training example must fit max_len."""
    for n in range(len(rec["sentences"]), 0, -1):
        if lengths(tok, {**rec, "sentences": rec["sentences"][:n]})[1] <= max_len:
            return n
    return 0


def summary(values: list[int]) -> dict:
    s = sorted(values)
    return {"min": s[0], "median": int(statistics.median(s)), "p95": s[int(0.95 * (len(s) - 1))], "max": s[-1]}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(settings.target_model_id)
    report = {"tokenizer": settings.target_model_id, "max_len": MAX_LEN, "splits": {}, "truncated": []}
    for name in ("train", "val", "test"):
        rows = read_split(name)
        pairs = [lengths(tok, r) for r in rows]
        answers = [t - p for p, t in pairs]
        report["splits"][name] = {
            "rows": len(rows),
            "prompt_tokens": summary([p for p, _ in pairs]),
            "training_tokens": summary([t for _, t in pairs]),
            "answer_tokens": summary(answers),
            "note_sentences": summary([len(r["sentences"]) for r in rows]),
            "over_max_len": sum(t > MAX_LEN for _, t in pairs),
        }
        for r, (_, t) in zip(rows, pairs):
            if t > MAX_LEN:
                keep = kept_sentences(tok, r, MAX_LEN)
                lost = [i for i in r["evidence_ids"] if i >= keep]
                report["truncated"].append(
                    {"split": name, "annotation_id": r["annotation_id"], "tokens": t,
                     "sentences": len(r["sentences"]), "kept": keep, "gold_evidence_lost": lost}
                )
    report["gold_evidence_lost_total"] = sum(len(x["gold_evidence_lost"]) for x in report["truncated"])
    out = settings.results_dir / "truncation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, s in report["splits"].items():
        print(f"{name:5} rows={s['rows']} prompt={s['prompt_tokens']} train={s['training_tokens']} "
              f"answer={s['answer_tokens']} over_{MAX_LEN}={s['over_max_len']}")
    print(f"truncated rows={len(report['truncated'])} gold evidence sentences lost={report['gold_evidence_lost_total']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
