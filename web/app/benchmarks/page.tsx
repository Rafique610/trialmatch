"use client";

import { useState } from "react";
import { Cpu, Layers } from "lucide-react";
import { DecoderBench } from "@/components/bench/DecoderBench";
import { AttentionBench } from "@/components/bench/AttentionBench";

const TABS = [
  { id: "decoders", name: "Speed", Icon: Cpu, title: "Speed: three ways to write the same answer" },
  { id: "attention", name: "Memory", Icon: Layers, title: "Memory: two ways to compute attention" },
] as const;

export default function BenchmarksPage() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("decoders");
  const active = TABS.find((t) => t.id === tab)!;
  return (
    <section className="sheet bench-sheet" aria-labelledby="bench-title">
      <div className="sheet-head">
        <h1 id="bench-title">{active.title}</h1>
        <span className="spacer" />
        <div className="segmented" role="tablist" aria-label="Benchmark">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} aria-pressed={tab === t.id} onClick={() => setTab(t.id)}>
              <t.Icon size={14} aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: 6 }} />{t.name}
            </button>
          ))}
        </div>
      </div>
      {/* both stay mounted so a finished run survives switching tabs */}
      <div hidden={tab !== "decoders"}><DecoderBench /></div>
      <div hidden={tab !== "attention"}><AttentionBench /></div>
    </section>
  );
}
