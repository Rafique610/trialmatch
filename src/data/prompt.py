"""The one prompt/answer contract shared by baseline, fine-tuning, evaluation and the app.

Input: numbered patient sentences + criterion type + criterion text (brief, Task 1).
Output: JSON {"label": ..., "evidence_ids": [...]} (ADR 0001). The explanation is
built later from the cited sentences, never generated.
"""

import json
import re

from src.data.load import LABELS

SYSTEM = (
    "You are a clinical trial screening assistant. You judge one eligibility criterion "
    "against a patient note, using only what the note says."
)

MEANING = {
    "inclusion": {
        "included": "the note shows the patient meets the criterion",
        "not included": "the note shows the patient does not meet the criterion",
        "not enough information": "the note does not say enough to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
    "exclusion": {
        "excluded": "the note shows the patient has this exclusion condition, so they would be excluded",
        "not excluded": "the note shows the patient does not have this exclusion condition",
        "not enough information": "the note does not say enough to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
}


def numbered(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


def user_message(sentences: list[str], criterion_type: str, criterion_text: str) -> str:
    labels = "\n".join(f'- "{k}": {v}' for k, v in MEANING[criterion_type].items())
    return (
        f"Patient note (numbered sentences):\n{numbered(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        f"Choose exactly one label:\n{labels}\n\n"
        'Answer with JSON only: {"label": "<label>", "evidence_ids": [<numbers of the sentences '
        "that support the label>]}. Use [] when no sentence is relevant. Never treat missing "
        "information as proof either way."
    )


def messages(record: dict) -> list[dict]:
    """Chat messages for one record (no answer): what the model sees at inference."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(record["sentences"], record["criterion_type"], record["criterion_text"])},
    ]


def target(record: dict) -> str:
    """The gold answer string the model is trained to produce."""
    return json.dumps({"label": record["label"], "evidence_ids": record["evidence_ids"]})


_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_output(text: str, criterion_type: str, n_sentences: int) -> dict:
    """Read the model's answer. Anything unusable is flagged, never guessed.

    Returns {"label", "evidence_ids", "parse_error", "dropped_ids"}; label is None on a parse error.
    """
    for candidate in _JSON_OBJECT.findall(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if label not in LABELS[criterion_type]:
            return {"label": None, "evidence_ids": [], "parse_error": f"label {label!r} not allowed", "dropped_ids": []}
        raw = obj.get("evidence_ids", [])
        raw = raw if isinstance(raw, list) else []
        ids = sorted({i for i in raw if isinstance(i, int) and 0 <= i < n_sentences})
        dropped = [i for i in raw if i not in ids]
        return {"label": label, "evidence_ids": ids, "parse_error": None, "dropped_ids": dropped}
    return {"label": None, "evidence_ids": [], "parse_error": "no JSON object", "dropped_ids": []}
