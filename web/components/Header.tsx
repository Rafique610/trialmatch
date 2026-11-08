"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { ClipboardList, Gauge } from "lucide-react";
import { api, type Health } from "@/lib/api";

type Status = { kind: "checking" } | { kind: "down" } | { kind: "loading" } | { kind: "ok"; health: Health };

export function Header() {
  const path = usePathname();
  const [status, setStatus] = useState<Status>({ kind: "checking" });

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      let next: Status;
      try {
        const h = await api.health();
        next = h.model_loaded ? { kind: "ok", health: h } : { kind: "loading" };
      } catch {
        next = { kind: "down" };
      }
      if (!alive) return;
      setStatus(next);
      timer = setTimeout(poll, next.kind === "ok" ? 30_000 : 4_000); // poll fast until the model is up
    };
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  const label = {
    checking: "Checking backend…",
    down: "Backend offline — start it with task app",
    loading: "Loading model…",
    ok: status.kind === "ok" ? `Model ready · ${status.health.provenance?.gpu.replace("NVIDIA GeForce ", "") ?? "GPU"}` : "",
  }[status.kind];

  return (
    <header className="header">
      <Link href="/" className="wordmark">
        <b>TrialMatch</b>
        <span>Qwen2.5-1.5B · QLoRA · criterion screening</span>
      </Link>
      <nav className="tabs" aria-label="Sections">
        <Link className="tab" href="/" aria-current={path === "/" ? "page" : undefined}>
          <ClipboardList size={16} aria-hidden="true" /> Screen
        </Link>
        <Link className="tab" href="/benchmarks" aria-current={path === "/benchmarks" ? "page" : undefined}>
          <Gauge size={16} aria-hidden="true" /> Benchmarks
        </Link>
      </nav>
      <div className="status" role="status" aria-live="polite">
        <span className={`dot ${status.kind === "ok" ? "ok" : status.kind === "down" ? "down" : "loading"}`} aria-hidden="true" />
        {label}
      </div>
    </header>
  );
}
