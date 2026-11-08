"use client";

import { useState } from "react";
import { CircleAlert, Loader2, Play, X } from "lucide-react";
import { api, ApiError, type AttentionBench as Result, type Measure } from "@/lib/api";
import { Bars, fmt } from "./Bars";

const SEQS = [256, 512, 1024, 2048, 4096];
const BLOCKS = [32, 64, 128, 256];
const METHODS = [
  { key: "reference", name: "Standard (full table)", tone: "ref" },
  { key: "tiled", name: "Tiled (ours)", tone: "tiled" },
  { key: "torch_sdpa", name: "PyTorch built-in", tone: "sdpa" },
] as const;

const cell = (m: Measure) => (m.error ? <span className="bad">{m.error}</span> : <>{fmt(m.ms!)} ms · {fmt(m.extra_mb!)} MB</>);

export function AttentionBench() {
  const [seqs, setSeqs] = useState<number[]>([256, 1024, 2048]);
  const [block, setBlock] = useState(128);
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const run = async () => {
    setBusy(true); setError(null);
    try { setRes(await api.attention(seqs, block)); }
    catch (e) { setError(e instanceof ApiError ? e : new ApiError(String(e), 500, "Try again.")); }
    finally { setBusy(false); }
  };
  const longest = res?.rows.at(-1);
  const ref = longest?.reference, tiled = longest?.tiled, sdpa = longest?.torch_sdpa;

  return (
    <div className="bench">
      <div className="explain-box">
        <p className="explain-lead">
          <b>Attention</b> compares every word with every earlier word. <b>Standard</b> builds the full length × length table;
          <b> tiled</b> works through it in blocks — same result, far less memory.
        </p>
      </div>
      <div className="bench-controls">
        <fieldset className="field">
          <legend className="field-label">Text lengths to test (tokens)</legend>
          <div className="checks">
            {SEQS.map((s) => (
              <label key={s} className="check">
                <input type="checkbox" checked={seqs.includes(s)} disabled={busy}
                  onChange={(e) => setSeqs(e.target.checked ? [...seqs, s].sort((a, b) => a - b) : seqs.filter((x) => x !== s))} />
                {s.toLocaleString()}
              </label>
            ))}
          </div>
        </fieldset>
        <label className="field">
          <span className="field-label">Block size for tiling</span>
          <select className="select" value={block} onChange={(e) => setBlock(Number(e.target.value))} disabled={busy}>
            {BLOCKS.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <button className="btn primary" onClick={run} disabled={busy || !seqs.length}>
          {busy ? <Loader2 size={14} className="spin" aria-hidden="true" /> : <Play size={14} aria-hidden="true" />}
          {busy ? "Measuring…" : "Compare attention methods"}
        </button>
      </div>
      <p className="bench-note">One layer’s attention (12 heads), measured live, 3 runs each.</p>

      {error && (
        <div className="alert" role="alert">
          <CircleAlert size={18} aria-hidden="true" />
          <div><strong>{error.message}</strong> <span>{error.recovery}</span></div>
          <button className="btn quiet" onClick={() => setError(null)} aria-label="Dismiss"><X size={15} /></button>
        </div>
      )}

      {res && longest && ref && tiled && sdpa && (
        <>
          <div className="findings" aria-live="polite">
            <p className="findings-title">Result</p>
            <ul>
              {ref.extra_mb != null && tiled.extra_mb != null && (
                <li className="good">Memory at {longest.seq.toLocaleString()} tokens: {fmt(ref.extra_mb)} MB → <b>{fmt(tiled.extra_mb)} MB</b> ({Math.round(ref.extra_mb / tiled.extra_mb)}× less)</li>
              )}
              <li className="good">Same result (max difference {longest.max_abs_error_tiled_vs_reference.toExponential(1)})</li>
              {ref.ms != null && tiled.ms != null && sdpa.ms != null && (
                <li>Slower: {fmt(tiled.ms)} ms vs {fmt(ref.ms)} ms — a Python loop; PyTorch’s fused kernel does the same in {fmt(sdpa.ms)} ms</li>
              )}
            </ul>
          </div>
          <div className="bench-grid">
            <Bars caption={`Memory at ${longest.seq.toLocaleString()} tokens`} unit="MB"
              rows={METHODS.map((m) => ({ label: m.name, value: longest[m.key].extra_mb ?? null, tone: m.tone, note: longest[m.key].error }))} />
            <Bars caption={`Time at ${longest.seq.toLocaleString()} tokens`} unit="ms"
              rows={METHODS.map((m) => ({ label: m.name, value: longest[m.key].ms ?? null, tone: m.tone, note: longest[m.key].error }))} />
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th className="num">Text length</th>{METHODS.map((m) => <th key={m.key} className="num">{m.name}</th>)}<th className="num">Largest difference (tiled vs standard)</th></tr></thead>
              <tbody>
                {res.rows.map((r) => (
                  <tr key={r.seq}>
                    <th scope="row" className="num mono">{r.seq.toLocaleString()}</th>
                    {METHODS.map((m) => <td key={m.key} className="num mono">{cell(r[m.key])}</td>)}
                    <td className="num mono">{r.max_abs_error_tiled_vs_reference.toExponential(1)}</td>
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
