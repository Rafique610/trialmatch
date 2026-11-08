"""Downloadable screening report (Markdown or HTML) with provenance. Built only from the screen result it is given."""

import html
from datetime import datetime, timezone


def markdown(result: dict) -> str:
    o, p = result["overall"], result["provenance"]
    lines = [
        "# TrialMatch screening report", "",
        f"> **{o['disclaimer']}**", "",
        f"**Preliminary outcome: {o['outcome']}** — {o['reason']}.", "",
        f"Rule: {o['rule']}.", "",
        "## Criteria", "",
        "| # | Type | Criterion | Label | Evidence |", "| --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(result["results"]):
        mark = " **(drives outcome)**" if i in o["driving_criteria"] and o["outcome"] != "likely eligible" else ""
        ev = ", ".join(str(e["id"]) for e in r["evidence"]) or "—"
        lines.append(f"| {i + 1} | {r['type']} | {r['text'].replace('|', '/')} | {r['label'] or 'unreadable answer'}{mark} | {ev} |")
    lines += ["", "## Explanations", ""]
    for i, r in enumerate(result["results"]):
        lines.append(f"{i + 1}. {r['explanation']}" + "".join(f" *Warning: {w}*" for w in r["warnings"]))
    lines += ["", "## Patient note (numbered as the model saw it)", ""]
    lines += [f"{s['id']}. {s['text']}" for s in result["sentences"]]
    lines += ["", "## Provenance", "",
              f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
              f"- Base model: {p['base_model']}", f"- Adapter: {p['adapter']} (sha256 {p['adapter_sha256'] or 'n/a'})",
              f"- Decoding: {result['mode']}" + (f", γ = {result['gamma']}" if result.get("gamma") else "") + f", {p['decoding']}, {p['precision']}",
              f"- GPU: {p['gpu']}", "- Code: https://github.com/Rafique610/trialmatch", ""]
    return "\n".join(lines)


def to_html(result: dict) -> str:
    """Same content as the Markdown, as a standalone page (no external assets)."""
    o, p = result["overall"], result["provenance"]
    esc = html.escape
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{esc(r['type'])}</td><td>{esc(r['text'])}</td><td><b>{esc(r['label'] or 'unreadable answer')}</b></td>"
        f"<td>{esc(r['explanation'])}{''.join('<br><i>Warning: ' + esc(w) + '</i>' for w in r['warnings'])}</td></tr>"
        for i, r in enumerate(result["results"]))
    note = "".join(f"<li value='{s['id']}'>{esc(s['text'])}</li>" for s in result["sentences"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>TrialMatch screening report</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d2433}}
.warn{{background:#fff4e5;border-left:4px solid #e08a00;padding:.6rem 1rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #d0d5dd;padding:.4rem;vertical-align:top;text-align:left}}ol{{padding-left:2rem}}</style></head><body>
<h1>TrialMatch screening report</h1><p class="warn"><b>{esc(o['disclaimer'])}</b></p>
<h2>Preliminary outcome: {esc(o['outcome'])}</h2><p>{esc(o['reason'])}.<br><small>Rule: {esc(o['rule'])}.</small></p>
<h2>Criteria</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Explanation</th></tr>{rows}</table>
<h2>Patient note</h2><ol start="0">{note}</ol>
<h2>Provenance</h2><ul><li>Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}</li>
<li>Base model: {esc(p['base_model'])}</li><li>Adapter: {esc(p['adapter'])} (sha256 {esc(p['adapter_sha256'] or 'n/a')})</li>
<li>Decoding: {esc(result['mode'])}{', γ = ' + str(result['gamma']) if result.get('gamma') else ''}, {esc(p['decoding'])}, {esc(p['precision'])}</li>
<li>GPU: {esc(p['gpu'])}</li></ul></body></html>"""
