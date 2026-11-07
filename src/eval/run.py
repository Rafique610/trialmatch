"""Score one run folder: results/<run>/predictions.jsonl + meta.json -> metrics, per-record rows, plots.

predictions.jsonl rows: {"annotation_id", "raw_output", "label", "evidence_ids", "parse_error", "dropped_ids"}
meta.json: {"split", "model", "adapter", "decoding", ...} written by the predictor (TRIA-6).
Run: uv run python -m src.eval.run results/<run>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import settings  # noqa: E402
from src.eval.metrics import score  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def join(gold: list[dict], preds: list[dict]) -> list[dict]:
    by_id = {p["annotation_id"]: p for p in preds}
    missing = [g["annotation_id"] for g in gold if g["annotation_id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} gold rows have no prediction, e.g. {missing[:5]}")
    return [
        {**g, "pred_label": by_id[g["annotation_id"]]["label"], "pred_evidence": by_id[g["annotation_id"]]["evidence_ids"],
         "dropped_ids": by_id[g["annotation_id"]].get("dropped_ids", []), "raw_output": by_id[g["annotation_id"]].get("raw_output")}
        for g in gold
    ]


def plot_confusion(cm: dict, title: str, path: Path) -> None:
    labels, m = cm["labels"], cm["matrix"]
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expert (gold)")
    top = max(max(row) for row in m) or 1
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > top / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    gold = read_jsonl(settings.data_dir / "processed" / f"{meta['split']}.jsonl")
    rows = join(gold, read_jsonl(run_dir / "predictions.jsonl"))
    metrics = {"meta": meta, **score(rows)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_record = [
        {k: r[k] for k in ("annotation_id", "patient_id", "criterion_type", "label", "pred_label", "evidence_ids", "pred_evidence", "raw_output")}
        | {"correct": r["label"] == r["pred_label"]}
        for r in rows
    ]
    (run_dir / "per_record.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_record), encoding="utf-8")
    for subset in ("overall", "inclusion", "exclusion"):
        plot_confusion(metrics[subset]["labels"]["confusion"], f"{run_dir.name} · {subset}", run_dir / f"confusion_{subset}.png")
    return metrics


def main() -> None:
    run_dir = Path(sys.argv[1])
    m = evaluate(run_dir)
    for subset in ("overall", "inclusion", "exclusion"):
        lab, ev = m[subset]["labels"], m[subset]["evidence"]
        print(f"{subset:9} n={lab['n']:3} acc={lab['accuracy']} macroF1={lab['macro_f1']} parse_err={lab['parse_errors']} "
              f"evidence micro F1={ev['micro']['f1']} macro F1={ev['macro']['f1']}")
    print(f"saved {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
