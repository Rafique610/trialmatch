"""Leakage and integrity checks for the 37/8/8 patient-grouped split (rubric B1, B2)."""

import json

import pytest

from src.config import settings
from src.data.load import DROPPED, LABELS, load_records, parse_sentences
from src.data.split import build


@pytest.fixture(scope="module")
def records():
    return load_records()


@pytest.fixture(scope="module")
def built(records):
    return build(records)


def test_sizes_match_brief(built):
    _, manifest = built
    assert [manifest["splits"][s]["patients"] for s in ("train", "val", "test")] == [37, 8, 8]


def test_no_patient_in_two_splits(built):
    parts, _ = built
    seen = [{r["patient_id"] for r in rows} for rows in parts.values()]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])


def test_every_annotation_assigned_once(records, built):
    parts, _ = built
    ids = [r["annotation_id"] for rows in parts.values() for r in rows]
    assert sorted(ids) == sorted(r["annotation_id"] for r in records)
    assert len(ids) == len(set(ids)) == 1015


def test_no_gpt4_or_flag_fields(records):
    for r in records:
        assert not set(DROPPED) & set(r)
        assert not any("gpt4" in k for k in r)


def test_labels_and_evidence_valid(records):
    for r in records:
        assert r["label"] in LABELS[r["criterion_type"]]
        assert all(0 <= i < len(r["sentences"]) for i in r["evidence_ids"])


def test_split_is_deterministic(records, built):
    _, again = build(records)
    assert again == built[1]


def test_published_manifest_is_current(built):
    path = settings.results_dir / "split_manifest.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    for name in ("train", "val", "test"):
        assert on_disk["splits"][name]["patient_ids"] == built[1]["splits"][name]["patient_ids"]


def test_parse_sentences_handles_wrapped_lines():
    assert parse_sentences("0. First.\n1. Second\ncontinues.\n2. Third.") == ["First.", "Second continues.", "Third."]
    with pytest.raises(ValueError):
        parse_sentences("no number here")
