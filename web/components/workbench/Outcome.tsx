"use client";

import { useState } from "react";
import { CircleCheck, CircleHelp, CircleX, Download, FileCode2, PencilLine } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { useWorkbench } from "@/lib/workbench";

const LOOK = {
  "likely eligible": { tone: "yes", Icon: CircleCheck, step: 3 },
  "likely ineligible": { tone: "no", Icon: CircleX, step: 1 },
  "needs review": { tone: "unk", Icon: CircleHelp, step: 2 },
} as const;

const RULE = [
  "Any criterion failed → likely ineligible",
  "Else any “can’t tell” → needs review",
  "Else → likely eligible",
];

/** Closes the form: the rule-based outcome, the rule itself with the step that fired, its drivers, and the report. */
export function Outcome({ codes }: { codes: string[] }) {
  const wb = useWorkbench();
  const [busy, setBusy] = useState<"html" | "markdown" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const o = wb.overall;

  const download = async (format: "html" | "markdown") => {
    const result = wb.screenResult();
    if (!result) return;
    setBusy(format); setError(null);
    try {
      const blob = await api.report(result, format);
      const url = URL.createObjectURL(blob);
      const a = Object.assign(document.createElement("a"), { href: url, download: `trialmatch-report.${format === "html" ? "html" : "md"}` });
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} ${e.recovery}` : "The report could not be created.");
    } finally {
      setBusy(null);
    }
  };

  if (!o) {
    return wb.running ? null : (
      <div className="outcome pending-outcome">
        <p>The outcome is decided only when every criterion has an answer.</p>
        <button className="btn" onClick={wb.clearResults}><PencilLine size={14} aria-hidden="true" /> Edit criteria</button>
      </div>
    );
  }

  const { tone, Icon, step } = LOOK[o.outcome];
  return (
    <div className={`outcome ${tone}`} id="outcome" aria-live="polite">
      <div className="outcome-head">
        <h2 className="verdict"><Icon size={22} aria-hidden="true" /> <span className="verdict-lead">Outcome:</span> {o.outcome}</h2>
        {o.outcome !== "likely eligible" && (
          <p className="drivers">
            Driven by{" "}
            {o.driving_criteria.map((i, k) => (
              <span key={i}>{k > 0 && ", "}<button className="link" onClick={() => wb.select(i)}>{codes[i]}</button></span>
            ))}
          </p>
        )}
      </div>
      <ol className="rule" aria-label="Aggregation rule">
        {RULE.map((r, k) => <li key={r} className={k + 1 === step ? "fired" : undefined}>{r}</li>)}
      </ol>
      <p className="disclaimer">A fixed rule over the answers above — not a prediction, not a clinical decision.</p>
      <div className="outcome-actions">
        <button className="btn primary" onClick={() => download("html")} disabled={!!busy}><Download size={14} aria-hidden="true" /> {busy === "html" ? "Preparing…" : "Download report"}</button>
        <button className="btn" onClick={() => download("markdown")} disabled={!!busy}><FileCode2 size={14} aria-hidden="true" /> {busy === "markdown" ? "Preparing…" : "Markdown"}</button>
        <span className="spacer" />
        <button className="btn quiet" onClick={wb.clearResults}><PencilLine size={14} aria-hidden="true" /> Edit criteria</button>
      </div>
      {error && <p className="hint bad" role="alert">{error}</p>}
    </div>
  );
}
