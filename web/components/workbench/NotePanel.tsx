"use client";

import { useEffect, useMemo, useRef } from "react";
import { FileText, PencilLine } from "lucide-react";
import type { Example } from "@/lib/api";
import { LIMITS, rowCodes } from "@/lib/criteria";
import { useWorkbench } from "@/lib/workbench";

export function NotePanel({ examples, onHoverSentence }: { examples: Example[]; onHoverSentence(id: number | null): void }) {
  const wb = useWorkbench();
  const list = useRef<HTMLOListElement>(null);
  const cited = useMemo(() => new Set(wb.selected != null ? wb.results[wb.selected]?.evidence_ids ?? [] : []), [wb.selected, wb.results]);
  const screened = wb.sentences != null;
  const codes = useMemo(() => rowCodes(wb.criteria), [wb.criteria]);
  // sentence id -> rows whose answer cites it (the note as a map back to the form)
  const citedBy = useMemo(() => {
    const m = new Map<number, number[]>();
    wb.results.forEach((r, i) => r?.evidence_ids.forEach((id) => m.set(id, [...(m.get(id) ?? []), i])));
    return m;
  }, [wb.results]);

  // bring the first cited sentence into view when a row is selected
  useEffect(() => {
    if (!cited.size || !list.current) return;
    const first = Math.min(...cited);
    list.current.querySelector<HTMLElement>(`[data-sentence="${first}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [cited]);

  const noteError = wb.note.length > LIMITS.noteChars || (wb.attempted && !wb.note.trim());

  return (
    <section className="sheet note-panel" aria-labelledby="note-title">
      <div className="sheet-head">
        <FileText size={16} aria-hidden="true" />
        <h2 id="note-title">Patient note</h2>
        <span className="meta">
          {screened ? `${wb.sentences!.length} sentences · ${citedBy.size} cited` : wb.note ? `${wb.note.length.toLocaleString()} / 20,000 characters` : ""}
        </span>
        <span className="spacer" />
        {screened ? (
          <button className="btn quiet" onClick={wb.clearResults} disabled={wb.running}>
            <PencilLine size={15} aria-hidden="true" /> Edit note
          </button>
        ) : (
          <label className="example-picker">
            <span className="visually-hidden">Load a validation example</span>
            <select
              className="select"
              value=""
              disabled={!examples.length || wb.running}
              onChange={(e) => { const ex = examples.find((x) => x.id === e.target.value); if (ex) wb.loadExample(ex); }}
            >
              <option value="">{examples.length ? "Load an example…" : "Examples load when the backend is up"}</option>
              {examples.map((x) => (
                <option key={x.id} value={x.id}>{`${x.patient_id} · ${x.trial_title.slice(0, 48)}${x.trial_title.length > 48 ? "…" : ""} (${x.criteria.length})`}</option>
              ))}
            </select>
          </label>
        )}
      </div>

      {screened ? (
        <ol className="sentences" ref={list} aria-label="Note as the model saw it">
          {wb.sentences!.map((s) => (
            <li
              key={s.id}
              data-sentence={s.id}
              className={cited.has(s.id) ? "cited" : undefined}
              onPointerEnter={(e) => e.pointerType === "mouse" && onHoverSentence(s.id)}
              onPointerLeave={() => onHoverSentence(null)}
            >
              <span className="sid mono" aria-hidden="true">{s.id}</span>
              <span className="stext">
                <mark className={cited.has(s.id) ? "on" : undefined}>{s.text}</mark>
                {citedBy.get(s.id)?.map((i) => (
                  <button key={i} className={`cite${wb.selected === i ? " on" : ""}`} onClick={() => wb.select(i)} title={`Cited by ${codes[i]}`}>
                    {codes[i]}
                  </button>
                ))}
              </span>
              {cited.has(s.id) && <span className="visually-hidden"> (cited by the selected criterion)</span>}
            </li>
          ))}
        </ol>
      ) : (
        <div className="note-edit">
          <label htmlFor="note" className="visually-hidden">Patient note</label>
          <textarea
            id="note"
            className="textarea"
            rows={18}
            value={wb.note}
            aria-invalid={noteError || undefined}
            aria-describedby="note-hint"
            placeholder="Paste a patient note…"
            onChange={(e) => wb.setNote(e.target.value)}
          />
          <p id="note-hint" className={`hint${noteError ? " bad" : ""}`}>
            {noteError ? (wb.note.trim() ? "Too long: the limit is 20,000 characters." : "Paste a patient note or load an example.") : ""}
          </p>
        </div>
      )}
    </section>
  );
}
