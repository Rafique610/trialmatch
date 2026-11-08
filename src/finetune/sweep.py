"""Run every training config in order (one subprocess each, so GPU memory is freed between runs),
then write a comparison table of every run and epoch.
A failed run is recorded and the queue moves on.
Run: uv run python -m src.finetune.sweep configs/A_r16.json configs/B_r8.json ...
     uv run python -m src.finetune.sweep --report      (table only)
"""

import json
import subprocess
import sys
import time
from datetime import datetime

from src.config import settings

matplotlib_ok = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib_ok = False


def keep_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs (released automatically when it exits)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def run_all(configs: list[str]) -> None:
    keep_awake()
    status = settings.results_dir / "finetune_queue.log"
    for cfg in configs:
        t0 = time.time()
        with status.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M} start {cfg}\n")
        code = subprocess.call([sys.executable, "-m", "src.finetune.train", cfg])
        with status.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M} end {cfg} exit={code} minutes={(time.time() - t0) / 60:.0f}\n")


def report() -> None:
    rows, best = [], None
    for path in sorted(settings.results_dir.glob("train_*/summary.json")):
        if "_smoke" in path.parent.name:
            continue
        s = json.loads(path.read_text(encoding="utf-8"))
        for e in s["epochs"]:
            rows.append((s["config"]["name"], s["config"].get("note", ""), e))
            if best is None or (e["val_macro_f1"] or 0) > (best[2]["val_macro_f1"] or 0):
                best = rows[-1]
    lines = [
        "# Fine-tuning runs (validation, scored in-training with the 4-bit model)",
        "",
        "| Run | Epoch | Val loss | Acc | Macro-F1 | Incl F1 | Excl F1 | Evidence micro-F1 | Parse err | Train min | Peak GB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, _, e in rows:
        lines.append(f"| {name} | {e['epoch']} | {e['val_loss']} | {e['val_accuracy']} | {e['val_macro_f1']} | {e['val_incl_macro_f1']} | "
                     f"{e['val_excl_macro_f1']} | {e['val_evidence_micro_f1']} | {e['val_parse_errors']} | {e['train_seconds'] / 60:.1f} | {e['peak_gpu_gb']} |")
    if best:
        lines += ["", f"Best by val macro-F1: **{best[0]} epoch {best[2]['epoch']}** (macro-F1 {best[2]['val_macro_f1']})."]
    (settings.results_dir / "finetune_runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if matplotlib_ok and rows:
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
        for name in dict.fromkeys(r[0] for r in rows):
            es = [r[2] for r in rows if r[0] == name]
            a.plot([e["epoch"] for e in es], [e["val_loss"] for e in es], marker="o", label=name)
            b.plot([e["epoch"] for e in es], [e["val_macro_f1"] for e in es], marker="o", label=name)
        a.set(title="Validation loss", xlabel="epoch")
        b.set(title="Validation macro-F1", xlabel="epoch")
        b.legend()
        fig.tight_layout()
        fig.savefig(settings.results_dir / "finetune_val_curves.png", dpi=150)


if __name__ == "__main__":
    if sys.argv[1:] != ["--report"]:
        run_all(sys.argv[1:])
    report()
