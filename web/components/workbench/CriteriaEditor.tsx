"use client";

import { useState } from "react";
import { ClipboardPaste, Plus, Trash2 } from "lucide-react";
import type { Criterion, CriterionType } from "@/lib/api";
import { LIMITS, parseBlock } from "@/lib/criteria";
import { useWorkbench } from "@/lib/workbench";

export function CriteriaEditor({ codes, invalid }: { codes: string[]; invalid: Set<string> }) {
  const wb = useWorkbench();
  const [paste, setPaste] = useState<string | null>(null);
  const rows = wb.criteria;

  // any edit makes the row a new criterion: an expert annotation no longer applies to it
  const update = (i: number, patch: Partial<Criterion>) =>
    wb.setCriteria(rows.map((c, k) => (k === i ? { type: c.type, text: c.text, ...patch } : c)));
  const add = (type: CriterionType) => wb.setCriteria([...rows, { type, text: "" }]);
  const parsed = paste ? parseBlock(paste) : [];

  return (
    <div className="editor">
      <ol className="editor-rows">
        {rows.map((c, i) => (
          <li key={i} className="editor-row">
            <span className={`code mono ${c.type}`}>{codes[i]}</span>
            <div className="segmented small" role="group" aria-label={`Row ${i + 1} type`}>
              {(["inclusion", "exclusion"] as const).map((t) => (
                <button key={t} aria-pressed={c.type === t} onClick={() => update(i, { type: t })}>{t === "inclusion" ? "Inclusion" : "Exclusion"}</button>
              ))}
            </div>
            <label className="visually-hidden" htmlFor={`criterion-${i}`}>Criterion {codes[i]}</label>
            <input
              id={`criterion-${i}`}
              className="input"
              value={c.text}
              maxLength={LIMITS.criterionMax}
              aria-invalid={invalid.has(`criterion-${i}`) || undefined}
              placeholder={c.type === "inclusion" ? "e.g. Adults aged 18 to 75 years" : "e.g. History of stroke"}
              onChange={(e) => update(i, { text: e.target.value })}
            />
            <button className="btn quiet" onClick={() => wb.setCriteria(rows.filter((_, k) => k !== i))} aria-label={`Remove ${codes[i]}`} disabled={rows.length === 1}>
              <Trash2 size={15} aria-hidden="true" />
            </button>
          </li>
        ))}
      </ol>

      <div className="editor-foot">
        <button className="btn" onClick={() => add("inclusion")} disabled={rows.length >= LIMITS.criteria}><Plus size={14} aria-hidden="true" /> Inclusion</button>
        <button className="btn" onClick={() => add("exclusion")} disabled={rows.length >= LIMITS.criteria}><Plus size={14} aria-hidden="true" /> Exclusion</button>
        <button className="btn quiet" onClick={() => setPaste(paste == null ? "" : null)} aria-expanded={paste != null}>
          <ClipboardPaste size={15} aria-hidden="true" /> Paste a protocol block
        </button>
        <span className="meta">{rows.length} / 30 criteria</span>
      </div>

      {paste != null && (
        <div className="paste">
          <label htmlFor="paste" className="field-label">Paste criteria: one per line, under “Inclusion criteria” / “Exclusion criteria” headings</label>
          <textarea id="paste" className="textarea" rows={7} value={paste} onChange={(e) => setPaste(e.target.value)}
            placeholder={"Inclusion criteria:\n- Age 40 to 75 years\n- HbA1c between 7.0% and 10.0%\nExclusion criteria:\n- History of stroke"} />
          <div className="editor-foot">
            <button className="btn primary" disabled={!parsed.length} onClick={() => {
              const keep = rows.filter((r) => r.text.trim());
              wb.setCriteria([...keep, ...parsed].slice(0, LIMITS.criteria)); setPaste(null);
            }}>
              Add {parsed.length || ""} {parsed.length === 1 ? "criterion" : "criteria"}
            </button>
            <span className="meta">
              {parsed.length ? `${parsed.filter((p) => p.type === "inclusion").length} inclusion · ${parsed.filter((p) => p.type === "exclusion").length} exclusion` : "Nothing recognised yet"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
