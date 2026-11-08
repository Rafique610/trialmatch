import { Workbench } from "@/components/workbench/Workbench";
import type { Example } from "@/lib/api";

const API_URL = process.env.TRIALMATCH_API ?? "http://127.0.0.1:8000"; // same backend the /api rewrite targets

/** Validation examples are fetched on the server; the page still works (and retries) if the backend is not up yet. */
async function loadExamples(): Promise<Example[]> {
  try {
    const res = await fetch(`${API_URL}/examples`, { cache: "no-store", signal: AbortSignal.timeout(3000) });
    return res.ok ? res.json() : [];
  } catch {
    return [];
  }
}

export default async function ScreenPage() {
  return <Workbench initialExamples={await loadExamples()} />;
}
