"""Patient-grouped 37/8/8 split exactly as the brief prescribes, plus a published manifest.

Sort unique patient ids, shuffle with numpy default_rng(42), first 37 -> train,
next 8 -> val, last 8 -> test. Every annotation follows its patient.
Run: uv run python -m src.data.split   (or: task data)
"""

import hashlib
import json
from collections import Counter

import numpy as np

from src.config import settings
from src.data.load import load_records

SPLITS = ("train", "val", "test")


def split_patients(patient_ids: list[str]) -> dict[str, list[str]]:
    ids = np.array(sorted(set(patient_ids)))
    np.random.default_rng(settings.split_seed).shuffle(ids)
    n_train, n_val, n_test = settings.split_sizes
    if len(ids) != n_train + n_val + n_test:
        raise ValueError(f"expected {n_train + n_val + n_test} patients, found {len(ids)}")
    cuts = (ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :])
    return {name: part.tolist() for name, part in zip(SPLITS, cuts)}


def build(records: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    patients = split_patients([r["patient_id"] for r in records])
    owner = {p: name for name, ps in patients.items() for p in ps}
    parts = {name: [r for r in records if owner[r["patient_id"]] == name] for name in SPLITS}
    manifest = {
        "dataset": settings.dataset_id,
        "dataset_revision": settings.dataset_revision,
        "rule": "sorted unique patient_id, numpy default_rng(42).shuffle, first 37 train / next 8 val / last 8 test",
        "seed": settings.split_seed,
        "total": {"patients": len(owner), "annotations": len(records)},
        "splits": {
            name: {
                "patient_ids": patients[name],
                "patients": len(patients[name]),
                "annotations": len(rows),
                "per_patient": dict(Counter(r["patient_id"] for r in rows)),
                "labels": {
                    ctype: dict(Counter(r["label"] for r in rows if r["criterion_type"] == ctype))
                    for ctype in ("inclusion", "exclusion")
                },
                "empty_evidence": sum(not r["evidence_ids"] for r in rows),
                "annotation_ids": sorted(r["annotation_id"] for r in rows),
            }
            for name, rows in parts.items()
        },
    }
    return parts, manifest


def to_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    parts, manifest = build(load_records())
    out_dir = settings.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest["files_sha256"] = {}
    for name, rows in parts.items():
        text = to_jsonl(rows)
        (out_dir / f"{name}.jsonl").write_text(text, encoding="utf-8")
        manifest["files_sha256"][f"{name}.jsonl"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = settings.results_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for name in ("train", "val", "test"):
        s = manifest["splits"][name]
        print(f"{name:5} patients={s['patients']:2} annotations={s['annotations']:4} empty_evidence={s['empty_evidence']:3} labels={s['labels']}")
    print(f"manifest sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]} -> {path}")


if __name__ == "__main__":
    main()
