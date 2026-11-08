"use client";

import { Check, CircleHelp, Loader2, TriangleAlert, UserCheck } from "lucide-react";
import { LABELS, type Criterion, type CriterionResult, type Label } from "@/lib/api";

const TONE: Record<Label, string> = {
  included: "yes", "not excluded": "yes", "not included": "no", excluded: "no", "not enough information": "unk", "not applicable": "na",
};

interface Props {
  code: string; criterion: Criterion; result: CriterionResult | null; pending: boolean; active: boolean;
  selected: boolean; linked: boolean; onSelect(): void;
}

/** One ruled CRF row: the criterion, all four answer boxes (the model's ticked), and, when selected, its evidence and working. */
export function ResultRow({ code, criterion, result, pending, active, selected, linked, onSelect }: Props) {
  const options = LABELS[criterion.type];
  const expert = criterion.expert_label;
  const agrees = result && expert ? result.label === expert : null;
  const state = active ? "active" : pending ? "pending" : result ? "done" : "skipped";

  return (
    <div className={`crf-row ${state}${selected ? " selected" : ""}${linked ? " linked" : ""}`} role="row">
      <button className="crf-main" role="cell" onClick={onSelect} disabled={!result} aria-expanded={result ? selected : undefined}
        aria-label={`${code}: ${criterion.text}${result ? ` — model answer: ${result.label ?? "unreadable"}` : ""}`}>
        <span className={`code mono ${criterion.type}`}>{code}</span>
        <span className="ctext">{criterion.text}</span>
        {result && expert && !agrees && (
          <span className="differs">Expert: {expert}</span>
        )}
        {result && (
          <span className="chips" aria-hidden="true">
            {result.evidence_ids.length ? result.evidence_ids.map((id) => <span key={id} className="chip mono">{id}</span>) : <span className="chip none">no evidence</span>}
          </span>
        )}
      </button>
      {options.map((opt) => {
        const ticked = result?.label === opt;
        const isExpert = expert === opt;
        return (
          <span key={opt} role="cell" className={`box-cell${isExpert ? " expert" : ""}`} title={isExpert ? "Expert annotation" : undefined}
            data-label={opt === "not enough information" ? "Not enough info" : opt[0].toUpperCase() + opt.slice(1)}>
            <span className={`box ${ticked ? `on ${TONE[opt]}` : ""}`} aria-label={ticked ? `${opt} (model)` : undefined}>
              {ticked ? <Check size={13} strokeWidth={3} aria-hidden="true" /> : active ? <span className="scan" aria-hidden="true" /> : null}
            </span>
          </span>
        );
      })}

      {active && <span className="row-status" role="cell"><Loader2 size={13} className="spin" aria-hidden="true" /> screening…</span>}
      {!result && !pending && !active && <span className="row-status" role="cell">not screened</span>}
      {result && result.label === null && (
        <span className="row-status bad" role="cell"><TriangleAlert size={13} aria-hidden="true" /> answer unreadable — needs review</span>
      )}

      {result && selected && (
        <div className="crf-detail" role="cell">
          {result.evidence.length ? (
            <ul className="quotes">
              {result.evidence.map((e) => <li key={e.id}><span className="qid mono">Sentence {e.id}</span> “{e.text}”</li>)}
            </ul>
          ) : (
            <p className="explain">{result.explanation}</p>
          )}
          <dl className="facts">
            {result.missing_information && (
              <div><dt><CircleHelp size={14} aria-hidden="true" /> Missing information</dt><dd>Not in the note → needs review</dd></div>
            )}
            {expert && (
              <div>
                <dt><UserCheck size={14} aria-hidden="true" /> Expert annotation</dt>
                <dd>
                  {expert}
                  {criterion.expert_evidence_ids?.length ? ` · sentences ${criterion.expert_evidence_ids.join(", ")}` : " · no sentences cited"}
                  <span className={`agree ${agrees ? "yes" : "no"}`}>{agrees ? "model agrees" : "model differs"}</span>
                </dd>
              </div>
            )}
            {result.warnings.map((w) => <div key={w}><dt><TriangleAlert size={14} aria-hidden="true" /> Warning</dt><dd>{w}</dd></div>)}
            <div>
              <dt>Model output</dt>
              <dd className="mono raw">{result.raw_output || "—"}</dd>
            </div>
            <div>
              <dt>Measured</dt>
              <dd className="mono">
                {result.timing.prompt_tokens} prompt tok · {result.timing.new_tokens} new · prefill {result.timing.prefill_ms} ms · decode {result.timing.decode_ms} ms
                {" · "}{result.timing.tok_s} tok/s · peak {result.timing.peak_gpu_gb} GB
                {result.timing.acceptance_rate != null && ` · acceptance ${(result.timing.acceptance_rate * 100).toFixed(0)}% (γ ${result.timing.gamma})`}
              </dd>
            </div>
          </dl>
        </div>
      )}
    </div>
  );
}
