"use client";

import Link from "next/link";
import { useState } from "react";
import { Check, CircleAlert, Loader2, Minus, Play, Plus, X } from "lucide-react";
import { api, ApiError, MODE_NAMES, type DecoderBench as Result } from "@/lib/api";
import { rowCodes } from "@/lib/criteria";
import { useWorkbench } from "@/lib/workbench";
import { Bars, fmt } from "./Bars";

const METHODS = [
  { name: "No cache", what: "Re-reads everything for each new word." },
  { name: "KV cache", what: "Reads once, remembers what it computed." },
  { name: "Speculative", what: "0.5B model drafts words, 1.5B model checks them in one go." },
];

export function DecoderBench() {
  const wb = useWorkbench();
  const [pick, setPick] = useState(0);
  const [gamma, setGamma] = useState(wb.gamma);
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const usable = wb.criteria.map((c, i) => ({ c, i })).filter(({ c }) => c.text.trim().length >= 3);
  const codes = rowCodes(wb.criteria);
  const criterion = wb.criteria[pick] && wb.criteria[pick].text.trim().length >= 3 ? wb.criteria[pick] : usable[0]?.c;

  const intro = (
    <div className="explain-box">
      <p className="explain-lead"><b>Same answer, three ways to write it</b> — only the speed should differ.</p>
      <dl className="methods">
        {METHODS.map((m) => <div key={m.name}><dt>{m.name}</dt><dd>{m.what}</dd></div>)}
      </dl>
    </div>
  );

  if (!wb.note.trim() || !criterion) {
    return (
      <div className="bench">
        {intro}
        <div className="empty">
          <p><strong>Check a patient on the Screen tab first</strong> — this compares the methods on that input.</p>
          <Link className="btn primary" href="/">Go to Screen</Link>
        </div>
      </div>
    );
  }

  const run = async () => {
    setBusy(true); setError(null);
    try { setRes(await api.benchmark(wb.note, criterion, gamma)); }
    catch (e) { setError(e instanceof ApiError ? e : new ApiError(String(e), 500, "Try again.")); }
    finally { setBusy(false); }
  };

  const total = (r: Result["runs"][number]) => r.prefill_ms + r.decode_ms;
  const byMode = res ? Object.fromEntries(res.runs.map((r) => [r.mode, r])) : {};
  const fastest = res ? [...res.runs].sort((a, b) => total(a) - total(b))[0] : null;
  const same = res?.runs.every((r) => r.same_tokens_as_kv_cache);
  const spec = byMode.speculative;

  return (
    <div className="bench">
      {intro}
      <div className="bench-controls">
        <label className="field">
          <span className="field-label">Criterion to answer</span>
          <select className="select" value={wb.criteria.indexOf(criterion)} onChange={(e) => setPick(Number(e.target.value))} disabled={busy}>
            {usable.map(({ c, i }) => <option key={i} value={i}>{`${codes[i]} · ${c.text.slice(0, 70)}`}</option>)}
          </select>
        </label>
        <div className="field">
          <span className="field-label">Words drafted per round (γ, speculative only)</span>
          <div className="stepper" role="group" aria-label="Words drafted per round">
            <button onClick={() => setGamma(Math.max(1, gamma - 1))} disabled={busy || gamma <= 1} aria-label="Fewer"><Minus size={14} aria-hidden="true" /></button>
            <output>{gamma}</output>
            <button onClick={() => setGamma(Math.min(8, gamma + 1))} disabled={busy || gamma >= 8} aria-label="More"><Plus size={14} aria-hidden="true" /></button>
          </div>
        </div>
        <button className="btn primary" onClick={run} disabled={busy}>
          {busy ? <Loader2 size={14} className="spin" aria-hidden="true" /> : <Play size={14} aria-hidden="true" />}
          {busy ? "Running all three…" : "Compare the three methods"}
        </button>
      </div>
      <p className="bench-note">Measured live on this GPU after one warm-up.</p>

      {error && (
        <div className="alert" role="alert">
          <CircleAlert size={18} aria-hidden="true" />
          <div><strong>{error.message}</strong> <span>{error.recovery}</span></div>
          <button className="btn quiet" onClick={() => setError(null)} aria-label="Dismiss"><X size={15} /></button>
        </div>
      )}

      {res && fastest && (
        <>
          <div className="findings" aria-live="polite">
            <p className="findings-title">Result</p>
            <ul>
              <li className={same ? "good" : "bad"}>{same ? <>Identical answers from all three: <code>{res.runs[1]?.answer}</code></> : <>Answers differ — see table</>}</li>
              <li>Fastest: <b>{MODE_NAMES[fastest.mode]}</b> {fmt(total(fastest))} ms · no cache {fmt(total(byMode.no_cache))} ms{res.prompt_tokens < 1000 ? " (short prompt, small gain)" : ""}</li>
              {spec && (
                <li>
                  Speculative: <b>{((spec.acceptance_rate ?? 0) * 100).toFixed(0)}%</b> of drafts accepted, {(total(byMode.kv_cache) / total(spec)).toFixed(2)}× KV-cache speed
                  {total(spec) < total(byMode.kv_cache) ? "" : " (the 0.5B draft is too slow on this GPU to pay off)"}
                </li>
              )}
            </ul>
          </div>

          <div className="bench-grid">
            <Bars caption={`Time per answer (${res.prompt_tokens}-token prompt)`} unit="ms"
              rows={res.runs.map((r) => ({ label: MODE_NAMES[r.mode], value: total(r), tone: r.mode }))} />
          </div>

          <div className="table-wrap">
            <table className="data">
              <caption className="visually-hidden">Measured details per method</caption>
              <thead>
                <tr><th>Method</th><th>Same answer?</th><th className="num">Reading the prompt (ms)</th><th className="num">Writing the answer (ms)</th>
                  <th className="num">Total (ms)</th><th className="num">Tokens written</th><th className="num">Tokens / second</th>
                  <th className="num">GPU memory (GB)</th><th className="num">Drafts accepted</th><th className="num">Speed vs no cache</th></tr>
              </thead>
              <tbody>
                {res.runs.map((r) => (
                  <tr key={r.mode}>
                    <th scope="row">{MODE_NAMES[r.mode]}{r.mode === "speculative" ? ` (γ ${res.gamma})` : ""}</th>
                    <td>{r.same_tokens_as_kv_cache
                      ? <span className="ok"><Check size={14} aria-hidden="true" /> identical</span>
                      : <span className="bad"><X size={14} aria-hidden="true" /> differs</span>}</td>
                    <td className="num mono">{fmt(r.prefill_ms)}</td><td className="num mono">{fmt(r.decode_ms)}</td><td className="num mono">{fmt(total(r))}</td>
                    <td className="num mono">{r.new_tokens}</td><td className="num mono">{fmt(r.tok_s)}</td><td className="num mono">{r.peak_gpu_gb.toFixed(2)}</td>
                    <td className="num mono">{r.acceptance_rate != null ? `${(r.acceptance_rate * 100).toFixed(0)}%` : "—"}</td>
                    <td className="num mono">{r.speedup_vs_no_cache.toFixed(2)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
