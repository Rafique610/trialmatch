"use client";

import { useEffect, useState } from "react";
import { api, type Example } from "@/lib/api";
import { NotePanel } from "./NotePanel";
import { EligibilityForm } from "./EligibilityForm";
import { StepStrip } from "./StepStrip";

/** Note on the left, the eligibility form on the right; they share which sentence / row is in focus. */
export function Workbench({ initialExamples }: { initialExamples: Example[] }) {
  const [examples, setExamples] = useState(initialExamples);
  const [hoverSentence, setHoverSentence] = useState<number | null>(null);

  useEffect(() => {
    if (examples.length) return;
    let alive = true;
    const retry = setInterval(() => api.examples().then((e) => { if (alive && e.length) { setExamples(e); clearInterval(retry); } }, () => {}), 5000);
    return () => { alive = false; clearInterval(retry); };
  }, [examples.length]);

  return (
    <>
      <StepStrip />
      <div className="workbench">
        <NotePanel examples={examples} onHoverSentence={setHoverSentence} />
        <EligibilityForm hoverSentence={hoverSentence} examples={examples} />
      </div>
    </>
  );
}
