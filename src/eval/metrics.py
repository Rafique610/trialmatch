"""Criterion label and evidence-sentence metrics (brief Task 2; rubric C1-C3).

Conventions (full text in docs/domain/trialmatch/eval.md):
- A parse error is its own predicted class, "parse_error". It is always wrong,
  and it is never remapped to a real label.
- Macro-F1 averages over the labels that occur in the gold data of that subset,
  so a class absent from the split (e.g. "excluded" on test) is reported as
  undefined rather than as 0. A variant averaged over every label of the subset
  (absent ones scoring 0) is reported alongside it.
- Evidence: see evidence_metrics.
"""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data.load import LABELS

PARSE_ERROR = "parse_error"
ALL_LABELS = sorted({label for labels in LABELS.values() for label in labels})


def label_metrics(gold: list[str], pred: list[str | None], labels: list[str]) -> dict:
    pred = [p if p is not None else PARSE_ERROR for p in pred]
    present = [label for label in labels if label in gold]
    p, r, f, support = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    per_class = {
        label: (
            {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(support[i])}
            if label in present
            else {"precision": None, "recall": None, "f1": None, "support": 0, "predicted": pred.count(label)}
        )
        for i, label in enumerate(labels)
    }
    cm_labels = labels + [PARSE_ERROR]
    return {
        "n": len(gold),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(sum(per_class[label]["f1"] for label in present) / len(present), 4) if present else None,
        "macro_f1_all_labels": round(float(f.mean()), 4),
        "parse_errors": pred.count(PARSE_ERROR),
        "per_class": per_class,
        "confusion": {"labels": cm_labels, "matrix": confusion_matrix(gold, pred, labels=cm_labels).tolist()},
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = 0.0 if (tp + fp) or (tp + fn) else None  # one side empty, the other not -> F1 0
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 4) if kept else None


def evidence_metrics(gold_sets: list[list[int]], pred_sets: list[list[int]]) -> dict:
    """Sentence-id retrieval scores.

    micro: TP/FP/FN pooled over all records (empty sets simply add nothing).
    macro: per-record P/R/F1, averaged over the records where each is defined.
      - gold empty, pred empty: nothing defined, so the record is skipped and
        counted as a correct abstention ("both_empty")
      - gold empty, pred non-empty: precision 0, recall undefined, F1 0
      - gold non-empty, pred empty: recall 0, precision undefined, F1 0
    macro_empty_as_correct: alternative where a both-empty record scores 1/1/1.
    """
    tp = fp = fn = 0
    per = []
    counts = {"both_empty": 0, "gold_empty_pred_nonempty": 0, "gold_nonempty_pred_empty": 0, "both_nonempty": 0}
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        t = len(g & p)
        tp, fp, fn = tp + t, fp + len(p - g), fn + len(g - p)
        per.append(_prf(t, len(p - g), len(g - p)))
        key = ("both_empty" if not p else "gold_empty_pred_nonempty") if not g else ("gold_nonempty_pred_empty" if not p else "both_nonempty")
        counts[key] += 1
    micro = _prf(tp, fp, fn)
    alt = [(1.0, 1.0, 1.0) if x == (None, None, None) else x for x in per]
    return {
        "n": len(gold_sets),
        "micro": {"precision": _r(micro[0]), "recall": _r(micro[1]), "f1": _r(micro[2]), "tp": tp, "fp": fp, "fn": fn},
        "macro": {k: _mean([x[i] for x in per]) for i, k in enumerate(("precision", "recall", "f1"))},
        "macro_empty_as_correct": {k: _mean([x[i] for x in alt]) for i, k in enumerate(("precision", "recall", "f1"))},
        "cases": counts,
    }


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def score(rows: list[dict]) -> dict:
    """rows: gold record fields + pred_label (str|None) + pred_evidence (list[int])."""
    out = {}
    for subset, labels in (("overall", ALL_LABELS), ("inclusion", LABELS["inclusion"]), ("exclusion", LABELS["exclusion"])):
        part = rows if subset == "overall" else [r for r in rows if r["criterion_type"] == subset]
        out[subset] = {
            "labels": label_metrics([r["label"] for r in part], [r["pred_label"] for r in part], labels),
            "evidence": evidence_metrics([r["evidence_ids"] for r in part], [r["pred_evidence"] for r in part]),
        }
    out["evidence_when_gold_nonempty"] = evidence_metrics(
        [r["evidence_ids"] for r in rows if r["evidence_ids"]], [r["pred_evidence"] for r in rows if r["evidence_ids"]]
    )
    out["dropped_ids_total"] = sum(len(r.get("dropped_ids", [])) for r in rows)
    return out
