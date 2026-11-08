"use client";

import { Check } from "lucide-react";
import { useWorkbench } from "@/lib/workbench";

/** Where the evaluator is in the three-step task. The sequence carries meaning, so the steps are numbered. */
const TONE = { "likely eligible": "yes", "likely ineligible": "no", "needs review": "unk" } as const;

export function StepStrip() {
  const wb = useWorkbench();
  const hasNote = !!wb.note.trim();
  const hasCriteria = wb.criteria.some((c) => c.text.trim().length >= 3);
  const hasResults = wb.running || wb.results.some(Boolean);
  const steps = [
    { title: "Patient note", hint: "", done: hasNote },
    { title: "Trial criteria", hint: "", done: hasCriteria },
    { title: "Results", hint: wb.overall ? wb.overall.outcome : hasResults ? "checking…" : "", done: !!wb.overall },
  ];
  const current = steps.findIndex((s) => !s.done);

  return (
    <ol className="steps" aria-label="Progress">
      {steps.map((s, i) => {
        const state = s.done ? "done" : i === current ? "current" : "todo";
        return (
          <li key={s.title} className={state} aria-current={state === "current" ? "step" : undefined}>
            <span className="step-num" aria-hidden="true">{s.done ? <Check size={14} strokeWidth={3} /> : i + 1}</span>
            <span className="step-text"><b>{s.title}</b>{s.hint && <span className={`step-tag ${wb.overall ? TONE[wb.overall.outcome] : ""}`}>{s.hint}</span>}</span>
          </li>
        );
      })}
    </ol>
  );
}
