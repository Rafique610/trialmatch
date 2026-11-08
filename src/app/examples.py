"""Real validation examples for the UI's "load example" (never test data).
One example = one patient note + the criteria experts annotated for one trial, with their gold labels."""

from collections import defaultdict
from functools import lru_cache

from src.config import settings
from src.data.prompt import numbered
from src.eval.run import read_jsonl


@lru_cache(maxsize=1)
def examples(limit: int = 6, max_criteria: int = 8) -> list[dict]:
    groups = defaultdict(list)
    for r in read_jsonl(settings.data_dir / "processed" / "val.jsonl"):
        groups[(r["patient_id"], r["trial_id"])].append(r)
    # most varied label sets first, so the examples show every kind of answer; one per patient
    ranked = sorted(groups.values(), key=lambda rs: (-len({r["label"] for r in rs[:max_criteria]}), rs[0]["patient_id"], rs[0]["trial_id"]))
    out, seen = [], set()
    for rs in ranked:
        if rs[0]["patient_id"] in seen:
            continue
        seen.add(rs[0]["patient_id"])
        rs = rs[:max_criteria]
        out.append({"id": f"{rs[0]['patient_id']}/{rs[0]['trial_id']}", "patient_id": rs[0]["patient_id"], "trial_id": rs[0]["trial_id"],
                    "trial_title": rs[0].get("trial_title", ""), "note": numbered(rs[0]["sentences"]),
                    "criteria": [{"type": r["criterion_type"], "text": r["criterion_text"], "expert_label": r["label"],
                                  "expert_evidence_ids": r["evidence_ids"]} for r in rs]})
        if len(out) == limit:
            break
    return out
