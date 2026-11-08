"use client";

import { useMemo, useRef, useState } from "react";
import { ChevronDown, CircleAlert, ClipboardCheck, Loader2, Minus, Play, Plus, Square, X } from "lucide-react";
import { MODE_NAMES, type Example, type Mode } from "@/lib/api";
import { rowCodes, validate } from "@/lib/criteria";
import { useWorkbench } from "@/lib/workbench";
import { CriteriaEditor } from "./CriteriaEditor";
import { ResultRow } from "./ResultRow";
import { Outcome } from "./Outcome";
import { ResultsSummary } from "./ResultsSummary";

const MODES: Mode[] = ["no_cache", "kv_cache", "speculative"];
const MODE_HINT: Record<Mode, string> = {
  no_cache: "re-reads everything for each word",
  kv_cache: "reads once, remembers (default)",
  speculative: "small model drafts, big model checks",
};

export function EligibilityForm({ hoverSentence, examples }: { hoverSentence: number | null; examples: Example[] }) {
  const wb = useWorkbench();
  const [problems, setProblems] = useState<{ field: string; message: string }[]>([]);
  const [advanced, setAdvanced] = useState(false);
  const summary = useRef<HTMLDivElement>(null);
  const codes = useMemo(() => rowCodes(wb.criteria), [wb.criteria]);
  const showingResults = wb.running || wb.results.some(Boolean);
  const done = wb.results.filter(Boolean).length;

  const start = () => {
    wb.markAttempted();
    const errs = validate(wb.note, wb.criteria);
    setProblems(errs);
    if (errs.length) { requestAnimationFrame(() => summary.current?.focus()); return; }
    wb.run();
  };

  const sections = (["inclusion", "exclusion"] as const).map((type) => ({
    type, rows: wb.criteria.map((c, i) => ({ c, i })).filter(({ c }) => c.type === type),
  })).filter((s) => s.rows.length);

  return (
    <section className="sheet form" aria-labelledby="form-title">
      <div className="sheet-head form-head">
        <ClipboardCheck size={16} aria-hidden="true" />
        <h1 id="form-title">Trial criteria</h1>
        <span className="meta" title={wb.example?.trial_title}>{wb.example ? wb.example.trial_title : `${wb.criteria.length} ${wb.criteria.length === 1 ? "criterion" : "criteria"}`}</span>
      </div>
      <div className="toolbar">
        <button className="btn quiet disclosure" aria-expanded={advanced} aria-controls="method-panel" onClick={() => setAdvanced(!advanced)}>
          <ChevronDown size={15} aria-hidden="true" className={advanced ? "flip" : undefined} />
          Advanced · method: <b>{MODE_NAMES[wb.mode]}{wb.mode === "speculative" ? ` (γ ${wb.gamma})` : ""}</b>
        </button>
        <span className="spacer" />
        {wb.running ? (
          <button className="btn" onClick={wb.stop}>
            <Square size={14} aria-hidden="true" /> Stop · {done}/{wb.criteria.length}
          </button>
        ) : (
          <button className="btn primary" onClick={start}>
            <Play size={14} aria-hidden="true" /> Check eligibility
          </button>
        )}
      </div>
      {advanced && (
        <div className="method-panel" id="method-panel">
          <p className="method-intro">Same answer, different speed — compare on <a href="/benchmarks">Benchmarks</a>.</p>
          <div className="segmented" role="group" aria-label="Generation method">
            {MODES.map((m) => (
              <button key={m} aria-pressed={wb.mode === m} onClick={() => wb.setMode(m)} disabled={wb.running}>{MODE_NAMES[m]}</button>
            ))}
          </div>
          {wb.mode === "speculative" && (
            <div className="stepper" role="group" aria-label="Words drafted per round (γ)">
              <button onClick={() => wb.setGamma(Math.max(1, wb.gamma - 1))} disabled={wb.running || wb.gamma <= 1} aria-label="Fewer drafted words"><Minus size={14} aria-hidden="true" /></button>
              <output aria-live="polite" title="γ: words the small model drafts per round">γ {wb.gamma}</output>
              <button onClick={() => wb.setGamma(Math.min(8, wb.gamma + 1))} disabled={wb.running || wb.gamma >= 8} aria-label="More drafted words"><Plus size={14} aria-hidden="true" /></button>
            </div>
          )}
          <p className="method-hint">{MODE_HINT[wb.mode]}{wb.mode === "speculative" ? " · γ = words drafted per round" : ""}</p>
        </div>
      )}

      {problems.length > 0 && !showingResults && (
        <div className="alert" role="alert" tabIndex={-1} ref={summary} aria-labelledby="problems-title">
          <CircleAlert size={18} aria-hidden="true" />
          <div>
            <strong id="problems-title">Fix {problems.length === 1 ? "this" : `these ${problems.length}`} first</strong>
            <ul>{problems.map((p) => <li key={p.field + p.message}><a href={p.field === "note" ? "#note" : `#${p.field}`}>{p.message}</a></li>)}</ul>
          </div>
          <button className="btn quiet" onClick={() => setProblems([])} aria-label="Dismiss"><X size={15} /></button>
        </div>
      )}

      {wb.error && (
        <div className={`alert${wb.error.status === 503 ? " warn" : ""}`} role="alert">
          <CircleAlert size={18} aria-hidden="true" />
          <div>
            <strong>{wb.error.message}</strong>
            <span>{wb.error.recovery.split("`").map((part, k) => (k % 2 ? <code key={k}>{part}</code> : part))}</span>
            {done > 0 && <span> {done} of {wb.criteria.length} criteria were screened before it stopped.</span>}
          </div>
          <button className="btn quiet" onClick={wb.dismissError} aria-label="Dismiss"><X size={15} /></button>
        </div>
      )}

      {showingResults ? (
        <>
          <ResultsSummary />
          {sections.map((s) => (
            <div className="crf-section" key={s.type} role="table" aria-label={`${s.type} criteria`}>
              <div className={`crf-cols ${s.type}`} role="row">
                <span role="columnheader">{s.type === "inclusion" ? "Inclusion criteria" : "Exclusion criteria"}</span>
                {(s.type === "inclusion" ? ["Included", "Not included"] : ["Excluded", "Not excluded"]).concat(["Not enough info", "Not applicable"]).map((h) => (
                  <span role="columnheader" key={h} className="colhead">{h}</span>
                ))}
              </div>
              {s.rows.map(({ c, i }) => (
                <ResultRow
                  key={i}
                  code={codes[i]}
                  criterion={c}
                  result={wb.results[i] ?? null}
                  pending={wb.running && wb.results[i] == null}
                  active={wb.current === i}
                  selected={wb.selected === i}
                  linked={hoverSentence != null && !!wb.results[i]?.evidence_ids.includes(hoverSentence)}
                  onSelect={() => wb.select(wb.selected === i ? null : i)}
                />
              ))}
            </div>
          ))}
          {wb.running && (
            <p className="run-note" role="status">
              <Loader2 size={14} className="spin" aria-hidden="true" /> Checking {codes[wb.current ?? 0]} ({done + 1} of {wb.criteria.length}) — the model answers one criterion at a time.
            </p>
          )}
          <Outcome codes={codes} />
        </>
      ) : (
        <>
          {!wb.note.trim() && wb.criteria.every((c) => !c.text.trim()) && (
            <div className="start">
              <p className="start-title">New here? Load an example — it includes an expert’s answers to compare with.</p>
              {examples.length > 0 && (
                <button className="btn" onClick={() => wb.loadExample(examples[0])}>Load an example</button>
              )}
            </div>
          )}
          <CriteriaEditor codes={codes} invalid={new Set(problems.map((p) => p.field))} />
        </>
      )}
    </section>
  );
}
