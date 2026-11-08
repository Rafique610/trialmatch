"""Screening logic: split an unseen note, run each criterion, parse, explain, aggregate (ADR 0001, 0002)."""

import re

from src.data.lengths import MAX_LEN
from src.data.load import parse_sentences
from src.data.prompt import parse_output

MAX_PROMPT_TOKENS = 3000  # memory cap on the 6 GB GPU (reference attention materialises L x L scores)
DISCLAIMER = "Preliminary rule-based outcome · educational screening prototype, not a clinical enrollment decision."

_NUMBERED_FIRST = re.compile(r"^\s*0\.\s")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_ABBREV = re.compile(r"(?:^|(?<=[\s(]))(Dr|Mr|Mrs|Ms|Prof|St|vs|e\.g|i\.e|approx|etc|No|Fig|pt|yo|y\.o)\.$", re.IGNORECASE)


class InputError(ValueError):
    pass


def split_note(note: str) -> list[str]:
    """A note already numbered "0. ...", "1. ..." (the dataset's format) is parsed exactly like training data.
    Anything else is split on line breaks and sentence-ending punctuation followed by a capital or digit."""
    if _NUMBERED_FIRST.match(note):
        try:
            return parse_sentences(note)
        except ValueError:
            pass
    out = []
    for line in note.splitlines():
        sentences: list[str] = []
        for part in (s.strip() for s in _SENTENCE_END.split(line)):
            if part and sentences and _ABBREV.search(sentences[-1]):
                sentences[-1] += " " + part  # "Dr." / "e.g." / "approx." did not end the sentence
            elif part:
                sentences.append(part)
        out += sentences  # a line break always ends a sentence
    return out


def explanation(label: str | None, ctype: str, ids: list[int], sentences: list[str], parse_error: str | None) -> str:
    """Built from the cited sentences, never generated (ADR 0001)."""
    if label is None:
        return f"The model's answer could not be read ({parse_error}), so this criterion needs manual review."
    if not ids:
        if label == "not enough information":
            return "No sentence in the note addresses this criterion, so it cannot be decided from the note."
        return f"Labelled '{label}' without citing a sentence; check the note before relying on it."
    cited = " ".join(f"[{i}] {sentences[i]}" for i in ids)
    return f"Labelled '{label}' ({ctype}) based on: {cited}"


def screen_one(engine, sentences: list[str], ctype: str, ctext: str, mode: str, gamma: int) -> dict:
    ids = engine.prompt_ids(sentences, ctype, ctext)
    if ids.shape[0] > MAX_PROMPT_TOKENS:
        raise InputError(f"note + criterion is {ids.shape[0]} tokens; the limit on this GPU is {MAX_PROMPT_TOKENS}")
    run = engine.decode(ids, mode, gamma)
    p = parse_output(run["text"], ctype, len(sentences))
    warnings = []
    if ids.shape[0] > MAX_LEN:
        warnings.append(f"Prompt is {ids.shape[0]} tokens, longer than any training example ({MAX_LEN}); treat the answer with extra care.")
    if p["dropped_ids"]:
        warnings.append(f"The model cited sentence ids that do not exist and were dropped: {p['dropped_ids']}.")
    return {"type": ctype, "text": ctext, "label": p["label"], "evidence_ids": p["evidence_ids"],
            "evidence": [{"id": i, "text": sentences[i]} for i in p["evidence_ids"]],
            "explanation": explanation(p["label"], ctype, p["evidence_ids"], sentences, p["parse_error"]),
            "missing_information": p["label"] in ("not enough information", None), "parse_error": p["parse_error"],
            "raw_output": run["text"], "warnings": warnings,
            "timing": {k: run[k] for k in ("prompt_tokens", "new_tokens", "prefill_ms", "decode_ms", "tok_s", "peak_gpu_gb")
                       } | {k: run[k] for k in ("gamma", "rounds", "acceptance_rate") if k in run}}


def aggregate(results: list[dict]) -> dict:
    """ADR 0002, checked in order. A parse error counts as needs review, never as a pass."""
    failed = [i for i, r in enumerate(results) if (r["type"], r["label"]) in (("inclusion", "not included"), ("exclusion", "excluded"))]
    unknown = [i for i, r in enumerate(results) if r["label"] in ("not enough information", None)]
    if failed:
        outcome, drivers, reason = "likely ineligible", failed, "at least one inclusion criterion is not met or one exclusion criterion applies"
    elif unknown:
        outcome, drivers, reason = "needs review", unknown, "no criterion fails, but at least one cannot be decided from the note"
    else:
        outcome, drivers, reason = "likely eligible", list(range(len(results))), "every inclusion criterion is met or not applicable and no exclusion applies"
    rule = ("1) any inclusion 'not included' or exclusion 'excluded' → likely ineligible; "
            "2) else any 'not enough information' (or unreadable answer) → needs review; 3) else likely eligible")
    return {"outcome": outcome, "reason": reason, "driving_criteria": drivers, "rule": rule, "disclaimer": DISCLAIMER}


def screen(engine, note: str, criteria: list[dict], mode: str, gamma: int) -> dict:
    sentences = split_note(note)
    if not sentences:
        raise InputError("the note has no sentences after splitting")
    results = [screen_one(engine, sentences, c["type"], c["text"], mode, gamma) for c in criteria]
    return {"sentences": [{"id": i, "text": s} for i, s in enumerate(sentences)], "results": results,
            "overall": aggregate(results), "mode": mode, "gamma": gamma if mode == "speculative" else None,
            "provenance": engine.provenance}
