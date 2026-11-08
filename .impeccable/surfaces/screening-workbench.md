# Surface brief · Screening workbench (web/app)

Scope: `/` screening workbench, plus `/benchmarks`. Mode: **Operate**. Code-led (app screens are always code-led).
Audience and job: the assignment evaluator pastes or loads a note and criteria, runs a screen, checks every label against the highlighted sentences, compares decoders live, and downloads the report.

## Direction contract

THESIS: The eligibility checklist *is* a clinical-trial case report form. Each criterion is a ruled CRF row with its full four-box answer column, so every possible answer stays visible. The model's answer is the ticked box. It refuses the category default of a SaaS card grid with coloured badges.

OWN-WORLD:
- Cool form-white ground with ink-navy type, hairline form-blue rules and numbered row codes (IN-1, EX-2).
- Square tick boxes and one highlighter-yellow mark for cited evidence.
- A single workhorse sans, Public Sans (the US government design-system face, at home on regulatory forms), with tabular figures and a system mono for raw output.
- No gradients, no glass, no shadows beyond a 1 px form edge.

STORY: The evaluator sees the note and the form side by side, runs the screen, and watches the boxes tick row by row. Clicking a row lights its sentences in the note like a highlighter on a printed chart. They believe the answer because the evidence is under their eye, then open Benchmarks to see the decoders measured on the same input.

FIRST VIEWPORT:
- Full-width banner stripe on top.
- Header: TrialMatch wordmark, a tab bar (Screen · Benchmarks), and model status on the right.
- Below, two columns: left 5/12, the "Patient note" panel, with numbered sentences once screened and a textarea before; right 7/12, the "Eligibility form" with criterion rows.
- The decoding controls and the primary "Screen patient" action sit in the form's header strip. The outcome block closes the form.

FORM: Case report form, position 1 of 3 on the grounded list (CRF; Drug Facts label; lab results report), chosen by the owner. Signature interaction: selecting a row highlights its cited sentences in the note and scrolls them into view; hovering a cited sentence highlights the rows that cite it. Motion: 150–200 ms tick-in as each row resolves, nothing decorative. seed: none (unavailable)

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
