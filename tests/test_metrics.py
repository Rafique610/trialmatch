"""Metrics checked against numbers worked out by hand, plus a gold-as-prediction run."""

import json

import pytest

from src.config import settings
from src.data.load import LABELS
from src.eval.metrics import evidence_metrics, label_metrics
from src.eval.run import evaluate, read_jsonl


def test_label_metrics_by_hand():
    gold = ["included", "included", "not included", "not enough information"]
    pred = ["included", "not enough information", "not included", None]
    m = label_metrics(gold, pred, LABELS["inclusion"])
    assert m["accuracy"] == 0.5 and m["parse_errors"] == 1
    assert m["per_class"]["included"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert m["per_class"]["not enough information"]["f1"] == 0.0
    assert m["per_class"]["not applicable"]["f1"] is None  # absent from gold: undefined, not 0
    assert m["macro_f1"] == 0.5556  # (0.6667 + 1 + 0) / 3 present labels
    assert m["macro_f1_all_labels"] == 0.4167  # absent label counted as 0
    assert m["confusion"]["labels"][-1] == "parse_error"
    nei = m["confusion"]["labels"].index("not enough information")
    assert m["confusion"]["matrix"][nei][-1] == 1  # gold NEI predicted as parse_error


def test_evidence_metrics_by_hand():
    m = evidence_metrics([[1, 2], [], [], [5]], [[2, 3], [], [4], []])
    assert m["micro"] == {"precision": 0.3333, "recall": 0.3333, "f1": 0.3333, "tp": 1, "fp": 2, "fn": 2}
    assert m["macro"] == {"precision": 0.25, "recall": 0.25, "f1": 0.1667}
    assert m["macro_empty_as_correct"] == {"precision": 0.5, "recall": 0.5, "f1": 0.375}
    assert m["cases"] == {"both_empty": 1, "gold_empty_pred_nonempty": 1, "gold_nonempty_pred_empty": 1, "both_nonempty": 1}


def _run_dir(tmp_path, preds):
    (tmp_path / "meta.json").write_text(json.dumps({"split": "val", "model": "gold-as-prediction"}))
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    return tmp_path


def test_gold_as_prediction_scores_perfect(tmp_path):
    gold = read_jsonl(settings.data_dir / "processed" / "val.jsonl")
    preds = [{"annotation_id": g["annotation_id"], "label": g["label"], "evidence_ids": g["evidence_ids"]} for g in gold]
    m = evaluate(_run_dir(tmp_path, preds))
    for subset in ("overall", "inclusion", "exclusion"):
        assert m[subset]["labels"]["accuracy"] == 1.0 and m[subset]["labels"]["macro_f1"] == 1.0
        assert m[subset]["evidence"]["micro"]["f1"] == 1.0 and m[subset]["evidence"]["macro"]["f1"] == 1.0
    assert (tmp_path / "confusion_overall.png").exists() and (tmp_path / "per_record.jsonl").exists()


def test_missing_predictions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="no prediction"):
        evaluate(_run_dir(tmp_path, []))
