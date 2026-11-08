/** Horizontal bars drawn from measured values (no chart library, no images). Values print beside each bar. */
export function Bars({ rows, unit, caption }: { rows: { label: string; value: number | null; tone?: string; note?: string }[]; unit: string; caption: string }) {
  const max = Math.max(...rows.map((r) => r.value ?? 0), 1e-9);
  return (
    <figure className="bars">
      <figcaption>{caption}</figcaption>
      {rows.map((r) => (
        <div className="bar-row" key={r.label}>
          <span className="bar-label">{r.label}</span>
          <span className="bar-track">
            {r.value != null && <span className={`bar ${r.tone ?? ""}`} style={{ width: `${Math.max(0.6, (100 * r.value) / max)}%` }} />}
          </span>
          <span className="bar-value mono">{r.value == null ? r.note ?? "n/a" : `${fmt(r.value)} ${unit}`}</span>
        </div>
      ))}
    </figure>
  );
}

export function fmt(v: number) {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}
