"use client";

import { MousePointerClick } from "lucide-react";
import type { Label } from "@/lib/api";
import { useWorkbench } from "@/lib/workbench";

const GROUP: Record<Label, "met" | "failed" | "unknown" | "na"> = {
  included: "met", "not excluded": "met", "not included": "failed", excluded: "failed", "not enough information": "unknown", "not applicable": "na",
};

const TONE = { "likely eligible": "yes", "likely ineligible": "no", "needs review": "unk" } as const;

/** The results in one line before the detail: counts by answer kind, and how to read the form. */
export function ResultsSummary() {
  const wb = useWorkbench();
  const got = wb.results.filter(Boolean);
  if (!got.length) return null;
  const n = { met: 0, failed: 0, unknown: 0, na: 0, unreadable: 0 };
  for (const r of got) r!.label ? n[GROUP[r!.label]]++ : n.unreadable++;
  const hasExpert = wb.criteria.some((c) => c.expert_label);

  return (
    <div className="summary" aria-live="polite">
      <p className="summary-counts">
        <span className="count yes"><b>{n.met}</b> met</span>
        <span className="count no"><b>{n.failed}</b> not met</span>
        <span className="count unk"><b>{n.unknown}</b> can’t tell</span>
        {n.na > 0 && <span className="count na"><b>{n.na}</b> don’t apply</span>}
        {n.unreadable > 0 && <span className="count no"><b>{n.unreadable}</b> unreadable</span>}
        <span className="of">{wb.running ? `checked ${got.length} of ${wb.criteria.length}` : `of ${wb.criteria.length} criteria`}</span>
        {wb.overall && (
          <a className={`overall-pill ${TONE[wb.overall.outcome]}`} href="#outcome">
            Overall: <b>{wb.overall.outcome}</b> · why ↓
          </a>
        )}
      </p>
      <p className="summary-how">
        <MousePointerClick size={14} aria-hidden="true" />
        Click a row for its evidence · filled box = model
        {hasExpert && <> · <span className="ring inline" aria-hidden="true" /> = expert</>}
      </p>
    </div>
  );
}
