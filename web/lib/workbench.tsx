"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, ApiError, type Criterion, type CriterionResult, type Example, type Mode, type Overall, type Provenance, type ScreenResponse, type Sentence } from "./api";

/** Shared screening state for the workbench and the benchmark page (kept in this tab's sessionStorage). */
interface Draft { note: string; criteria: Criterion[]; mode: Mode; gamma: number; example: Pick<Example, "id" | "trial_title"> | null }

interface State extends Draft {
  results: (CriterionResult | null)[];
  sentences: Sentence[] | null;
  overall: Overall | null;
  provenance: Provenance | null;
  running: boolean;
  current: number | null;
  error: ApiError | null;
  selected: number | null;
  attempted: boolean;
}

interface Actions {
  setNote(note: string): void;
  setCriteria(c: Criterion[]): void;
  setMode(m: Mode): void;
  setGamma(g: number): void;
  loadExample(e: Example): void;
  select(i: number | null): void;
  markAttempted(): void;
  run(): Promise<void>;
  stop(): void;
  clearResults(): void;
  dismissError(): void;
  screenResult(): ScreenResponse | null;
}

const KEY = "trialmatch.draft.v1";
const EMPTY: Draft = { note: "", criteria: [{ type: "inclusion", text: "" }], mode: "kv_cache", gamma: 2, example: null };
const Ctx = createContext<(State & Actions) | null>(null);

export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [results, setResults] = useState<(CriterionResult | null)[]>([]);
  const [sentences, setSentences] = useState<Sentence[] | null>(null);
  const [overall, setOverall] = useState<Overall | null>(null);
  const [provenance, setProvenance] = useState<Provenance | null>(null);
  const [running, setRunning] = useState(false);
  const [current, setCurrent] = useState<number | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [attempted, setAttempted] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(KEY);
      if (saved) setDraft({ ...EMPTY, ...JSON.parse(saved) });
    } catch { /* storage unavailable: start empty */ }
  }, []);
  useEffect(() => {
    try { sessionStorage.setItem(KEY, JSON.stringify(draft)); } catch { /* not persisted; still works */ }
  }, [draft]);

  const clearResults = useCallback(() => {
    setResults([]); setSentences(null); setOverall(null); setSelected(null); setCurrent(null);
  }, []);

  const edit = useCallback((patch: Partial<Draft>) => {
    abort.current?.abort();
    setDraft((d) => ({ ...d, ...patch }));
    clearResults();
  }, [clearResults]);

  const run = useCallback(async () => {
    const { note, criteria, mode, gamma } = draft;
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    clearResults(); setError(null); setRunning(true);
    const got: (CriterionResult | null)[] = criteria.map(() => null);
    setResults([...got]);
    try {
      for (let i = 0; i < criteria.length; i++) {
        setCurrent(i);
        const r = await api.screenOne(note, criteria[i], mode, gamma, ctl.signal);
        got[i] = r.results[0];
        setResults([...got]); setSentences(r.sentences); setProvenance(r.provenance);
        if (i === 0) setSelected(0);
      }
      setCurrent(null);
      setOverall(await api.aggregate(got.map((r) => ({ type: r!.type, label: r!.label }))));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(e instanceof ApiError ? e : new ApiError(String(e), 500, "Try again."));
      setCurrent(null);
    } finally {
      setRunning(false);
    }
  }, [draft, clearResults]);

  const value = useMemo<State & Actions>(() => ({
    ...draft, results, sentences, overall, provenance, running, current, error, selected, attempted,
    markAttempted: () => setAttempted(true),
    setNote: (note) => edit({ note, example: null }),
    setCriteria: (criteria) => edit({ criteria }),
    setMode: (mode) => setDraft((d) => ({ ...d, mode })),
    setGamma: (gamma) => setDraft((d) => ({ ...d, gamma })),
    loadExample: (e) => { edit({ note: e.note, criteria: e.criteria, example: { id: e.id, trial_title: e.trial_title } }); setError(null); },
    select: setSelected,
    run, stop: () => abort.current?.abort(), clearResults, dismissError: () => setError(null),
    screenResult: () => (overall && sentences && provenance && results.every(Boolean)
      ? { sentences, results: results as CriterionResult[], overall, mode: draft.mode, gamma: draft.mode === "speculative" ? draft.gamma : null, provenance }
      : null),
  }), [draft, results, sentences, overall, provenance, running, current, error, selected, attempted, edit, run, clearResults]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useWorkbench() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useWorkbench outside WorkbenchProvider");
  return v;
}
