# IEEE paper (TRIA-14)

`report/main.tex` (IEEEtran conference), `report/refs.bib`, `report/figures/`, `report/tables/`.
Build: `task paper`. It runs `report/make_tables.py`, then Tectonic 0.17 (from `.tools/`, git-ignored; package cache on D:),
and writes `report/main.pdf` (8 pages).

**No number without a source file.** Every table is generated from `results/` by `report/make_tables.py`, and every number
quoted in the prose is a macro from `report/tables/numbers.tex` (41 macros), generated the same way. Each table carries a
"Source:" note naming its file.

## Required sections (assignment §Technical Report)

| Required | Where |
| --- | --- |
| Title and author details | Title block (Muhammad Rafique, 23i-0747, CS-B) |
| Abstract and keywords | Abstract, Index Terms |
| Introduction and problem statement | §I |
| Related Work (cited literature) | §II, 16 references from `docs/research/literature.md` |
| Materials and Methods: dataset, split | §III-A, Table I |
| Selected model | §III-B |
| Fine-tuning | §III-C |
| Architecture diagram | Fig. 1 (TikZ) |
| Attention and decoding algorithms | Algorithms 1–3 (§III-D to III-F) |
| Application workflow | §III-G, Fig. 2 |
| Experimental setup and metrics | §IV |
| Results and discussion | §V |
| Training curves | Fig. 3 |
| Tables and graphs | Tables II–IX, Figs. 4–7 |
| Measured runtimes | Tables VII–IX, Figs. 5–7 |
| Error analysis | §V-C, Table V, Fig. 4 |
| Screenshots | Fig. 8 |
| Limitations | §VI |
| Conclusion | §VII |
| References | IEEEtran bibliography |
| AI-use appendix | Appendix A |

Every figure and table is referenced and interpreted in the text.

## Owner to check before submitting

- The author block, and whether to add the university name.
- The AI-use statement wording (Appendix A): it's the author's declaration.
- Fig. 8 screenshots were captured on battery (timings throttled). For cleaner numbers, rerun `scripts/ui_screens.py` plugged in, then `task paper`.
