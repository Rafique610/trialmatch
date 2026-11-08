"""Before/after table from two scored run folders.
Run: uv run python -m src.eval.compare results/baseline_test results/finetuned_test > results/test_comparison.md
"""

import json
import sys
from pathlib import Path


def fmt(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def table(before: Path, after: Path) -> str:
    a, b = (json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in (before, after))
    out = [f"# {before.name} vs {after.name}", "", f"Rows: {a['overall']['labels']['n']} (same rows, same decoding).", "",
           "| Part | Metric | Before | After |", "| --- | --- | --- | --- |"]
    for part in ("overall", "inclusion", "exclusion"):
        la, lb, ea, eb = a[part]["labels"], b[part]["labels"], a[part]["evidence"], b[part]["evidence"]
        rows = [("accuracy", la["accuracy"], lb["accuracy"]), ("macro-F1", la["macro_f1"], lb["macro_f1"]),
                ("parse errors", la["parse_errors"], lb["parse_errors"]),
                ("evidence micro-F1", ea["micro"]["f1"], eb["micro"]["f1"]), ("evidence macro-F1", ea["macro"]["f1"], eb["macro"]["f1"])]
        out += [f"| {part} | {name} | {x if isinstance(x, int) else fmt(x)} | {y if isinstance(y, int) else fmt(y)} |" for name, x, y in rows]
    for part in ("inclusion", "exclusion"):
        out += ["", f"## Per-class F1 · {part}", "", "| Label | Support | Before | After |", "| --- | --- | --- | --- |"]
        for label, cb in b[part]["labels"]["per_class"].items():
            out.append(f"| {label} | {cb['support']} | {fmt(a[part]['labels']['per_class'][label]['f1'])} | {fmt(cb['f1'])} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(table(Path(sys.argv[1]), Path(sys.argv[2])), end="")
