"""App backend (TRIA-12): validation errors, the outcome rule, one full screen call, report.
CPU tests use a stand-in engine with scripted answers; the GPU test uses the real fine-tuned model."""

import json

import pytest
import torch
from fastapi.testclient import TestClient

from src.app.main import create_app
from src.app.screening import aggregate, split_note
from src.data.prompt import user_message

ANSWERS = {  # criterion text -> scripted model answer
    "Adults aged 18 to 75": '{"label": "included", "evidence_ids": [0]}',
    "History of stroke": '{"label": "not excluded", "evidence_ids": [2]}',
    "Prior chemotherapy": '{"label": "excluded", "evidence_ids": [1]}',
    "Pregnant": '{"label": "not enough information", "evidence_ids": []}',
    "Garbled": "I think maybe yes",
}


class FakeEngine:
    provenance = {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": "Aduhere/trialmatch-qwen2.5-1.5b-qlora", "adapter_sha256": "abc",
                  "draft_model": "Qwen/Qwen2.5-0.5B-Instruct", "precision": "fp16", "decoding": "greedy", "gpu": "test"}
    draft_loaded = False

    def prompt_ids(self, sentences, ctype, ctext):
        self.last = ctext
        return torch.zeros(len(user_message(sentences, ctype, ctext)) // 4)  # ~4 characters per token

    def decode(self, ids, mode, gamma=2):
        text = ANSWERS[self.last]
        return {"text": text, "tokens": list(range(12)), "prompt_tokens": int(ids.shape[0]), "new_tokens": 12,
                "prefill_ms": 10.0, "decode_ms": 20.0, "tok_s": 400.0, "peak_gpu_gb": 3.5}


NOTE = "A 58-year-old man presents with chest pain. He received chemotherapy in 2019. He has no history of stroke."


@pytest.fixture
def client():
    return TestClient(create_app(engine=FakeEngine()), raise_server_exceptions=False)


def body(*crits, note=NOTE, **kw):
    return {"note": note, "criteria": [{"type": t, "text": x} for t, x in crits], **kw}


@pytest.mark.parametrize("payload,where", [
    (body(("inclusion", "Adults aged 18 to 75"), note="   "), "note"),
    (body(), "criteria"),
    (body(("other", "Adults aged 18 to 75")), "type"),
    (body(("inclusion", "")), "text"),
    (body(("inclusion", "Adults aged 18 to 75"), mode="beam"), "mode"),
    (body(("inclusion", "Adults aged 18 to 75"), mode="speculative", gamma=0), "gamma"),
    (body(("inclusion", "Adults aged 18 to 75"), note="x" * 20_001), "note"),
    (body(*[("inclusion", "Adults aged 18 to 75")] * 31), "criteria"),
])
def test_validation_errors_are_422(client, payload, where):
    r = client.post("/screen", json=payload)
    assert r.status_code == 422 and where in json.dumps(r.json()["detail"])


def test_full_screen_call(client):
    r = client.post("/screen", json=body(("inclusion", "Adults aged 18 to 75"), ("exclusion", "History of stroke"), ("exclusion", "Prior chemotherapy")))
    assert r.status_code == 200
    d = r.json()
    assert [s["id"] for s in d["sentences"]] == [0, 1, 2]
    assert [x["label"] for x in d["results"]] == ["included", "not excluded", "excluded"]
    assert d["results"][2]["evidence"] == [{"id": 1, "text": "He received chemotherapy in 2019."}]
    assert "He received chemotherapy in 2019." in d["results"][2]["explanation"]
    assert d["overall"]["outcome"] == "likely ineligible" and d["overall"]["driving_criteria"] == [2]
    assert d["provenance"]["adapter"] == "Aduhere/trialmatch-qwen2.5-1.5b-qlora"


@pytest.mark.parametrize("crits,outcome,drivers", [
    ([("inclusion", "Adults aged 18 to 75"), ("exclusion", "History of stroke")], "likely eligible", [0, 1]),
    ([("inclusion", "Adults aged 18 to 75"), ("exclusion", "Pregnant")], "needs review", [1]),
    ([("exclusion", "Pregnant"), ("exclusion", "Prior chemotherapy")], "likely ineligible", [1]),
    ([("inclusion", "Garbled"), ("inclusion", "Adults aged 18 to 75")], "needs review", [0]),
])
def test_outcome_rule(client, crits, outcome, drivers):
    d = client.post("/screen", json=body(*crits)).json()
    assert d["overall"]["outcome"] == outcome and d["overall"]["driving_criteria"] == drivers


def test_unreadable_answer_is_flagged_not_guessed(client):
    res = client.post("/screen", json=body(("inclusion", "Garbled"))).json()["results"][0]
    assert res["label"] is None and res["parse_error"] == "no JSON object" and res["missing_information"]
    assert "manual review" in res["explanation"]


def test_missing_information_flag(client):
    res = client.post("/screen", json=body(("exclusion", "Pregnant"))).json()["results"][0]
    assert res["missing_information"] and res["evidence"] == [] and "cannot be decided" in res["explanation"]


def test_too_long_is_a_clear_422(client):
    long_note = " ".join(["The patient reports mild fatigue."] * 450)  # ~15k characters, under the schema limit
    r = client.post("/screen", json=body(("inclusion", "Adults aged 18 to 75"), note=long_note))
    assert r.status_code == 422 and "limit on this GPU" in r.json()["detail"]


def test_not_loaded_is_503_and_health_says_loading():
    c = TestClient(create_app(load_on_startup=False), raise_server_exceptions=False)
    assert c.get("/health").json()["status"] == "loading"
    r = c.post("/screen", json=body(("inclusion", "Adults aged 18 to 75")))
    assert r.status_code == 503 and "loading" in r.json()["detail"]


def test_report_has_disclaimer_and_provenance(client):
    result = client.post("/screen", json=body(("exclusion", "Prior chemotherapy"))).json()
    for fmt, kind in (("markdown", "text/markdown"), ("html", "text/html")):
        r = client.post("/report", json={"result": result, "format": fmt})
        assert r.status_code == 200 and r.headers["content-type"].startswith(kind) and "attachment" in r.headers["content-disposition"]
        assert "not a clinical enrollment decision" in r.text and "Aduhere/trialmatch-qwen2.5-1.5b-qlora" in r.text and "likely ineligible" in r.text


def test_split_note_plain_and_numbered():
    assert split_note("He is 40. She is 30.\nBP 120/80 today.") == ["He is 40.", "She is 30.", "BP 120/80 today."]
    assert split_note("0. First sentence.\n1. Second\ncontinues here.") == ["First sentence.", "Second continues here."]
    assert split_note("Dr. Smith saw him. Temp 37.5 C was normal.") == ["Dr. Smith saw him.", "Temp 37.5 C was normal."]
    assert split_note("Labs, e.g. CBC, were normal. Pt. has HTN.") == ["Labs, e.g. CBC, were normal.", "Pt. has HTN."]


def test_aggregate_empty_evidence_never_passes():
    out = aggregate([{"type": "inclusion", "label": None}])
    assert out["outcome"] == "needs review"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU and the fine-tuned model")
def test_real_screen_all_modes():
    from src.app.engine import Engine

    c = TestClient(create_app(engine=Engine()))
    payload = body(("inclusion", "Adults aged 18 to 75 years"), ("exclusion", "History of stroke"))
    labels = {}
    for mode in ("no_cache", "kv_cache", "speculative"):
        d = c.post("/screen", json=payload | {"mode": mode}).json()
        assert d["overall"]["outcome"] in ("likely eligible", "likely ineligible", "needs review")
        labels[mode] = [(x["label"], x["evidence_ids"]) for x in d["results"]]
    assert labels["no_cache"] == labels["kv_cache"] == labels["speculative"]  # same greedy answer from every decoder
    b = c.post("/benchmark", json={"note": NOTE, "criterion": {"type": "exclusion", "text": "History of stroke"}, "gamma": 2}).json()
    assert all(r["same_tokens_as_kv_cache"] for r in b["runs"])


def test_examples_are_real_val_records(client):
    ex = client.get("/examples").json()
    assert 1 <= len(ex) <= 6 and len({e["patient_id"] for e in ex}) == len(ex)
    e = ex[0]
    assert e["note"].startswith("0. ") and 1 <= len(e["criteria"]) <= 8
    c = e["criteria"][0]
    assert c["type"] in ("inclusion", "exclusion") and c["expert_label"]
    assert split_note(e["note"])[0] == e["note"].splitlines()[0][3:]  # numbered note parses back exactly


def test_aggregate_endpoint_matches_rule(client):
    r = client.post("/aggregate", json={"results": [{"type": "inclusion", "label": "included"}, {"type": "exclusion", "label": "excluded"}]})
    assert r.status_code == 200 and r.json()["outcome"] == "likely ineligible" and r.json()["driving_criteria"] == [1]
    assert client.post("/aggregate", json={"results": []}).status_code == 422
    assert client.post("/aggregate", json={"results": [{"type": "inclusion", "label": "maybe"}]}).status_code == 422
