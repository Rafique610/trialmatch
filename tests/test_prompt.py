import json

from src.config import settings
from src.data.lengths import MAX_LEN
from src.data.prompt import messages, parse_output, target

REC = {
    "sentences": ["A 60-year-old man.", "He has type 2 diabetes.", "No cancer history."],
    "criterion_type": "inclusion",
    "criterion_text": "Type 2 diabetes",
    "label": "included",
    "evidence_ids": [1],
}


def test_prompt_has_numbered_note_and_only_allowed_labels():
    user = messages(REC)[1]["content"]
    assert "0. A 60-year-old man.\n1. He has type 2 diabetes." in user
    assert '"included"' in user and '"excluded"' not in user


def test_target_round_trips():
    assert parse_output(target(REC), "inclusion", 3) == {
        "label": "included", "evidence_ids": [1], "parse_error": None, "dropped_ids": []
    }


def test_extra_text_and_bad_ids_tolerated_but_reported():
    out = parse_output('Sure! {"label": "Not Included", "evidence_ids": [2, 9, "x"]} done', "inclusion", 3)
    assert out["label"] == "not included" and out["evidence_ids"] == [2] and out["dropped_ids"] == [9, "x"]


def test_unusable_answers_are_errors_not_guesses():
    assert parse_output("I think yes", "inclusion", 3)["parse_error"] == "no JSON object"
    wrong_set = parse_output('{"label": "excluded", "evidence_ids": []}', "inclusion", 3)
    assert wrong_set["label"] is None and wrong_set["parse_error"]


def test_published_truncation_report_loses_no_evidence():
    report = json.loads((settings.results_dir / "truncation_report.json").read_text(encoding="utf-8"))
    assert report["max_len"] == MAX_LEN
    assert report["gold_evidence_lost_total"] == 0
