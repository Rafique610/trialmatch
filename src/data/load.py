"""Load the TrialGPT criterion annotations and keep only expert-written fields.

GPT-4 predictions, GPT-4 explanations, the expert grade of those explanations,
and the original authors' `training` flag are dropped here, so nothing later
can see them. The notes arrive already numbered ("0. ...\n1. ..."); we parse
those numbers rather than re-split the text, because expert evidence ids point
at them.
"""

import json
import re

from datasets import load_dataset

from src.config import settings

# Allow-list: everything not named here is dropped.
KEEP = ["annotation_id", "patient_id", "trial_id", "trial_title", "criterion_type", "criterion_text"]
DROPPED = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility", "training"]
LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
_NUMBERED = re.compile(r"^(\d+)\.\s+(.*)$")


def parse_sentences(note: str) -> list[str]:
    """Split a numbered note into sentences; index i holds sentence "i. ...".

    A line without a number continues the previous sentence. Numbers must run 0, 1, 2, ...
    """
    sentences: list[str] = []
    for line in note.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUMBERED.match(line)
        if m and int(m.group(1)) == len(sentences):
            sentences.append(m.group(2))
        elif sentences:
            sentences[-1] += " " + line
        else:
            raise ValueError(f"note does not start with '0. ': {line[:60]!r}")
    return sentences


def clean(row: dict) -> dict:
    sentences = parse_sentences(row["note"])
    evidence = sorted(set(json.loads(row["expert_sentences"])))
    if any(i < 0 or i >= len(sentences) for i in evidence):
        raise ValueError(f"annotation {row['annotation_id']}: evidence {evidence} outside 0..{len(sentences) - 1}")
    label = row["expert_eligibility"]
    if label not in LABELS[row["criterion_type"]]:
        raise ValueError(f"annotation {row['annotation_id']}: label {label!r} not valid for {row['criterion_type']}")
    return {**{k: row[k] for k in KEEP}, "sentences": sentences, "label": label, "evidence_ids": evidence}


def load_records() -> list[dict]:
    ds = load_dataset(settings.dataset_id, revision=settings.dataset_revision, split="train")
    return [clean(row) for row in ds]
