"""Export every number the paper uses from results/ into LaTeX (TRIA-14). Nothing in the paper is typed by hand.

Writes report/tables/*.tex (tables), report/tables/numbers.tex (\\newcommand macros for numbers quoted in the text)
and report/figures/train_curves.pdf, and copies the result figures the paper includes.
Run: uv run python -m report.make_tables
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R, OUT, FIG = ROOT / "results", ROOT / "report" / "tables", ROOT / "report" / "figures"
RUNS = ["A_r16", "B_r8", "C_r16_weighted", "D_r16_lr5e-5"]
RUN_NAME = {"A_r16": "A: r16", "B_r8": "B: r8", "C_r16_weighted": "C: r16 + weights", "D_r16_lr5e-5": "D: r16, lr 5e-5"}


def load(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def table(name, cols, header, rows, caption, label, star=False, note=None):
    env = "table*" if star else "table"
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    tab = f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    if not star:  # single column: never wider than the column
        tab = f"\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{%\n{tab}}}"  # shrink only
    tex = (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n"
           f"\\setlength{{\\tabcolsep}}{{4pt}}\n{tab}\n"
           + (f"\\\\[2pt]\\parbox{{{'0.95' if star else '0.98'}\\linewidth}}{{\\scriptsize {note}}}\n" if note else "") + f"\\end{{{env}}}\n")
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    num = {}

    # ── data split
    man = load("split_manifest.json")
    rows = []
    for s in ("train", "val", "test"):
        sp = man["splits"][s]
        inc, exc = sp["labels"]["inclusion"], sp["labels"]["exclusion"]
        rows.append([s, sp["patients"], sp["annotations"], inc.get("included", 0), inc.get("not included", 0), inc.get("not enough information", 0),
                     inc.get("not applicable", 0), exc.get("excluded", 0), exc.get("not excluded", 0), exc.get("not enough information", 0),
                     exc.get("not applicable", 0)])
    table("split", "lrr|rrrr|rrrr", r" & & & \multicolumn{4}{c|}{Inclusion} & \multicolumn{4}{c}{Exclusion} \\ Split & Pat. & Rows & inc & not inc & NEI & NA & exc & not exc & NEI & NA",
          rows, "Patient-grouped split (seed 42) and label counts per split.", "tab:split", star=True,
          note=f"Source: results/split\\_manifest.json (dataset revision {man['dataset_revision'][:12]}). NEI = not enough information, NA = not applicable.")
    num.update(TotalPatients=man["total"]["patients"], TotalRows=man["total"]["annotations"], TestRows=man["splits"]["test"]["annotations"],
               TestExcluded=man["splits"]["test"]["labels"]["exclusion"].get("excluded", 0), TrainRows=man["splits"]["train"]["annotations"])

    # ── test before/after
    b, f = load("baseline_test/metrics.json"), load("finetuned_test/metrics.json")
    rows = []
    for part in ("overall", "inclusion", "exclusion"):
        lb, lf, eb, ef = b[part]["labels"], f[part]["labels"], b[part]["evidence"], f[part]["evidence"]
        rows.append([part, lb["n"], f3(lb["accuracy"]), f3(lf["accuracy"]), f3(lb["macro_f1"]), f3(lf["macro_f1"]),
                     f3(eb["micro"]["f1"]), f3(ef["micro"]["f1"]), f3(eb["macro"]["f1"]), f3(ef["macro"]["f1"]), f"{lb['parse_errors']} / {lf['parse_errors']}"])
    table("test", "lr|rr|rr|rr|rr|r", r"Subset & $n$ & \multicolumn{2}{c|}{Accuracy} & \multicolumn{2}{c|}{Macro-F1} & \multicolumn{2}{c|}{Evid.\ micro-F1} & \multicolumn{2}{c|}{Evid.\ macro-F1} & Parse err.",
          rows, "Sealed test split, untrained (left of each pair) vs fine-tuned (right). Same 206 criteria, same prompt and decoding.", "tab:test", star=True,
          note="Source: results/baseline\\_test/metrics.json, results/finetuned\\_test/metrics.json. Macro-F1 averages labels present in the gold set; evidence F1 compares cited sentence ids.")
    num.update(TestAccBase=f3(b["overall"]["labels"]["accuracy"]), TestAccFT=f3(f["overall"]["labels"]["accuracy"]),
               TestFoneBase=f3(b["overall"]["labels"]["macro_f1"]), TestFoneFT=f3(f["overall"]["labels"]["macro_f1"]),
               TestEvBase=f3(b["overall"]["evidence"]["micro"]["f1"]), TestEvFT=f3(f["overall"]["evidence"]["micro"]["f1"]),
               TestEvNonemptyFT=f3(f["evidence_when_gold_nonempty"]["micro"]["f1"]))

    # ── per-class F1 on test
    rows = []
    for part in ("inclusion", "exclusion"):
        for lab, cf in f[part]["labels"]["per_class"].items():
            cb = b[part]["labels"]["per_class"][lab]
            rows.append([part[:3] + ".", lab, cf["support"], f3(cb["f1"]), f3(cf["f1"]), f3(cf["precision"]), f3(cf["recall"])])
    table("perclass", "llr|rr|rr", r"Type & Label & Supp. & F1 base & F1 FT & P FT & R FT", rows,
          "Per-class results on the test split (FT = fine-tuned).", "tab:perclass",
          note="Source: per\\_class in results/*\\_test/metrics.json. ``--'': label absent from the test gold set.")

    # ── confusion cells (largest off-diagonal) for the error analysis
    cm = f["overall"]["labels"]["confusion"]
    labs, mat = cm["labels"], cm["matrix"]
    cells = sorted(((mat[i][j], labs[i], labs[j]) for i in range(len(labs)) for j in range(len(labs)) if i != j and mat[i][j]), reverse=True)[:5]
    table("confusion_top", "lll", r"Count & Gold & Predicted", [[c, g, p] for c, g, p in cells],
          "Largest error cells of the fine-tuned model on test.", "tab:conf",
          note="Source: confusion matrix in results/finetuned\\_test/metrics.json.")
    num.update(TestWrong=sum(mat[i][j] for i in range(len(labs)) for j in range(len(labs)) if i != j), TopCell=cells[0][0])

    # ── fine-tuning sweep: best epoch per run by val macro-F1
    rows, best = [], None
    sums = {r: load(f"train_{r}/summary.json") for r in RUNS}
    for r, s in sums.items():
        e = max(s["epochs"], key=lambda e: e["val_macro_f1"])
        rows.append([RUN_NAME[r], s["config"]["rank"], f"{s['config']['lr']:g}", "yes" if s["config"]["class_weighting"] else "no", e["epoch"],
                     f"{e['val_loss']:.3f}", f3(e["val_accuracy"]), f3(e["val_macro_f1"]), f3(e["val_excl_macro_f1"]), f3(e["val_evidence_micro_f1"])])
        if best is None or e["val_macro_f1"] > best[1]["val_macro_f1"]:
            best = (r, e)
    table("sweep", "lrrcr|rrrrr", r"Run & $r$ & LR & Wts & Ep. & Val loss & Acc & Macro-F1 & Excl F1 & Evid.\ F1", rows,
          "QLoRA sweep, best epoch of each run by validation macro-F1 (4-bit, in-training scoring).", "tab:sweep", star=True,
          note="Source: results/train\\_*/summary.json. Wts = inverse-square-root label weighting.")
    num.update(BestRun=RUN_NAME[best[0]], BestEpoch=best[1]["epoch"], BestValFone=f3(best[1]["val_macro_f1"]),
               TrainableParams=f"{sums['C_r16_weighted']['trainable_params'] / 1e6:.1f}")
    fv = load("finetuned_val/metrics.json")
    num.update(ValAccFT=f3(fv["overall"]["labels"]["accuracy"]), ValFoneFT=f3(fv["overall"]["labels"]["macro_f1"]))

    # ── training curves figure: train loss (run C) + val macro-F1 per epoch (all runs)
    log = [json.loads(l) for l in (R / "train_C_r16_weighted" / "log.jsonl").read_text(encoding="utf-8").splitlines() if l]
    fig, (a, c) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    a.plot([x["step"] for x in log], [x["loss"] for x in log], lw=0.8, color="#1d4fa0", label="train loss (run C)")
    ep = sums["C_r16_weighted"]["epochs"]
    steps_per_epoch = len(log) / len(ep)
    a.plot([e["epoch"] * steps_per_epoch for e in ep], [e["val_loss"] for e in ep], "o-", color="#a8201a", ms=3, label="val loss (run C)")
    a.set(xlabel="optimiser step", ylabel="loss", yscale="log")
    a.legend(fontsize=7, frameon=False)
    for r in RUNS:
        es = sums[r]["epochs"]
        c.plot([e["epoch"] for e in es], [e["val_macro_f1"] for e in es], "o-", ms=3, label=RUN_NAME[r])
    c.set(xlabel="epoch", ylabel="val macro-F1")
    c.legend(fontsize=7, frameon=False)
    for ax in (a, c):
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    fig.tight_layout()
    fig.savefig(FIG / "train_curves.pdf")

    # ── decoding sensitivity (repetition penalty) on validation
    rows = []
    for name, pen, folder in (("untrained", "1.1", "baseline_val"), ("untrained", "1.0", "baseline_val_greedy"),
                              ("fine-tuned", "1.1", "finetuned_val"), ("fine-tuned", "1.0", "finetuned_val_greedy")):
        m = load(f"{folder}/metrics.json")["overall"]
        rows.append([name, pen, f3(m["labels"]["accuracy"]), f3(m["labels"]["macro_f1"]), f3(m["evidence"]["micro"]["f1"])])
    table("penalty", "llrrr", r"Model & Rep.\ pen. & Acc & Macro-F1 & Evid.\ F1", rows,
          "Validation: generate's default repetition penalty (1.1) vs pure greedy (1.0).", "tab:penalty",
          note="Source: results/\\{baseline,finetuned\\}\\_val\\{,\\_greedy\\}/metrics.json. No label changed in any of the 206 rows.")

    # ── KV cache
    kv = load("bench/kvcache.json")
    rows = []
    for r in kv["benchmark"]:
        n_out = r["output_len"] - 1
        rows.append([r["input_len"], f"{r['cached']['prefill_ms']['mean']:.0f}", f"{r['no_cache']['decode_ms']['mean'] / n_out:.1f}",
                     f"{r['cached']['decode_ms']['mean'] / n_out:.1f}", f"{r['decode_speedup']:.1f}$\\times$",
                     f"{r['no_cache']['peak_gb']:.2f} / {r['cached']['peak_gb']:.2f}", f"{r['cached']['cache_bytes_per_step'][-1] / 2**20:.1f}"])
    eq = all(e["cached_equals_own_no_cache"] and e["cached_equals_hf_no_cache"] for e in kv["equivalence"])
    combos = ", ".join(str(e["input_len"]) + "/" + str(e["output_len"]) for e in kv["equivalence"])
    table("kvcache", "rrrrrrr", r"Input & Prefill ms & ms/tok no cache & ms/tok cache & Speed-up & Peak GB & Cache MB", rows,
          "KV-cache benchmark, 64 new tokens, mean of 5 runs after warm-up.", "tab:kv",
          note=f"Source: results/bench/kvcache.json. Token equality with full recomputation at {combos} (input/output): "
               f"{'all identical' if eq else 'MISMATCH'}.")
    last = kv["benchmark"][-1]
    num.update(KvMaxSpeedup=f"{last['decode_speedup']:.1f}", KvMaxInput=last["input_len"], KvCachedMs=f"{last['cached']['decode_ms']['mean'] / (last['output_len'] - 1):.0f}",
               KvNoCacheMs=f"{last['no_cache']['decode_ms']['mean'] / (last['output_len'] - 1):.0f}", KvFirstSpeedup=f"{kv['benchmark'][0]['decode_speedup']:.1f}")

    # ── attention
    at, ae = load("bench/attention.json"), load("bench/attention_errors.json")
    def m(x):
        return "--" if "ms" not in x else f"{x['ms']:.1f} / {x['extra_mb']:.0f}"
    rows = [[r["seq"], m(r["reference"]), m(r["tiled_128"]), m(r["sdpa_math"]), m(r["sdpa_mem_efficient"])] for r in at["timing"]]
    table("attention", "rrrrr", r"Seq & Reference & Tiled ($B{=}128$) & SDPA math & SDPA mem-eff.", rows,
          "Attention time (ms) / extra memory (MB), causal, 12 heads, $d{=}128$, fp16.", "tab:attn",
          note="Source: results/bench/attention.json (1 warm-up, 5 repeats). The flash SDPA backend is absent from the Windows PyTorch build.")
    s32, s16 = ae["summary"]["float32"], ae["summary"]["float16"]
    long = at["timing"][-1]
    num.update(AttnErrThirtyTwo=f"{s32['tiled_vs_ref_max']:.1e}", AttnErrSixteen=f"{s16['tiled_vs_ref_max']:.1e}", SdpaErrSixteen=f"{s16['sdpa_vs_ref_max']:.1e}",
               AttnConfigs=s32["configs"] + s16["configs"], AttnLongSeq=long["seq"], AttnRefMB=f"{long['reference']['extra_mb']:.0f}",
               AttnTiledMB=f"{long['tiled_128']['extra_mb']:.0f}", AttnMemRatio=f"{long['reference']['extra_mb'] / long['tiled_128']['extra_mb']:.0f}",
               InModelDiff=f"{at['in_model']['prefill_logits_max_abs_diff']:.2f}")

    # ── speculative
    sp = load("bench/speculative.json")
    w = next(x for x in sp["workloads"] if x["workload"].startswith("task, fine-tuned"))
    rows = [[g["gamma"], f3(g["acceptance_rate"]), f"{g['tokens_per_round']:.2f}", f"{g['draft_ms_per_round']:.0f}", f"{g['verify_ms_per_round']:.0f}",
             f"{g['tok_s']:.1f}", f"{g['speedup']:.2f}$\\times$", g["outputs_identical_to_greedy"]] for g in w["gammas"]]
    table("speculative", "rrrrrrrr", r"$\gamma$ & Accept & Tok/pass & Draft ms & Verify ms & tok/s & Speed-up & Identical", rows,
          f"Speculative decoding, fine-tuned target, {w['prompts']} screening prompts (plain greedy: {w['baseline']['tok_s']:.1f} tok/s).", "tab:spec",
          note=f"Source: results/bench/speculative.json (1 warm-up, {sp['repeats']} repeats). Speed-up = plain cached greedy time / speculative time.")
    num.update(SpecBaseStep=f"{w['baseline']['target_decode_ms_per_token']:.0f}", SpecDraftMs=f"{w['gammas'][0]['draft_ms_per_round']:.0f}",
               SpecBestSpeedup=f"{max(g['speedup'] for g in w['gammas']):.2f}", SpecWorstSpeedup=f"{min(g['speedup'] for g in w['gammas']):.2f}",
               SpecAccMin=f"{min(g['acceptance_rate'] for g in w['gammas']):.2f}", SpecAccMax=f"{max(g['acceptance_rate'] for g in w['gammas']):.2f}",
               SpecHFcheck=f"{sp['hf_assisted_reference']['ours_speculative_equals_hf_assisted']}/{sp['hf_assisted_reference']['prompts']}")

    # ── numbers used in the text
    def macro(k):
        return k.replace("_", "")
    (OUT / "numbers.tex").write_text("% generated by report/make_tables.py from results/ — do not edit\n"
                                     + "".join(f"\\newcommand{{\\{macro(k)}}}{{{v}}}\n" for k, v in num.items()), encoding="utf-8")

    # ── confusion matrices drawn for the page: short labels, readable counts
    short = {"included": "inc", "not included": "not inc", "excluded": "exc", "not excluded": "not exc", "not enough information": "NEI",
             "not applicable": "NA", "parse_error": "parse err"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, part in zip(axes, ("inclusion", "exclusion")):
        c = f[part]["labels"]["confusion"]
        rows_i = [i for i, lab in enumerate(c["labels"]) if lab != "parse_error"]
        mat = [c["matrix"][i] for i in rows_i]
        top = max(map(max, mat))
        ax.imshow(mat, cmap="Blues")
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v > top / 2 else "#14233f")
        ax.set_xticks(range(len(c["labels"])), [short[lab] for lab in c["labels"]], fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(len(rows_i)), [short[c["labels"][i]] for i in rows_i], fontsize=7)
        ax.set_title(f"{part} (n={f[part]['labels']['n']})", fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("gold", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "confusion.pdf")

    # ── figures copied from results/ (app screenshots cropped to the meaningful first viewport)
    for src, dst in (("bench/kvcache.png", "kvcache.png"), ("bench/attention.png", "attention.png"), ("bench/speculative.png", "speculative.png")):
        shutil.copyfile(R / src, FIG / dst)
    from PIL import Image

    Image.open(R / "ui/3-screen-results.png").crop((0, 0, 1440, 900)).save(FIG / "app_screen.png")
    Image.open(R / "ui/4-bench-decoders.png").crop((0, 0, 1440, 900)).save(FIG / "app_bench.png")
    print(f"tables: {sorted(p.name for p in OUT.iterdir())}\nnumbers: {len(num)}")


if __name__ == "__main__":
    main()
