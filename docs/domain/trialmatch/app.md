# App backend (TRIA-12)

FastAPI in `src/app/`: `main.py` (app + error mapping), `routers/screening.py` (thin endpoints),
`schemas/screen.py` (typed requests and responses with constraints), plus three feature-local
modules holding the logic:
- `engine.py`: model loading and decoding.
- `screening.py`: splitting, prompt, parsing, explanation and the outcome rule.
- `bench.py`: live benchmarks.
- `report.py`: the downloadable report.

Start: `task app` → http://localhost:8000/docs (every endpoint can be tried there).

## Startup

The fine-tuned model loads once, in a background thread, so the server answers immediately:
- **Target:** Qwen2.5-1.5B fp16 with the adapter **merged** (`checkpoints/final`, or
  `Aduhere/trialmatch-qwen2.5-1.5b-qlora` from the Hub if absent).
- **Draft:** the 0.5B, loaded lazily on the first speculative request.

`/health` reports `loading` / `ok`, and any request before the model is ready gets a 503.
- **Decoding:** always our own pure-greedy decoders (TRIA-9/11): no repetition penalty, at most 64 new tokens, stop at `<|im_end|>`.
- **Concurrency:** a lock serialises GPU work, because there's one GPU and one request decodes at a time.

## Endpoints

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /health` | – | model / draft loaded, provenance |
| `POST /screen` | `note`, `criteria[{type, text}]` (1–30), `mode` (`no_cache` / `kv_cache` / `speculative`), `gamma` (1–8) | numbered sentences; per criterion: label, evidence ids + text, explanation, missing-info flag, parse error, raw answer, warnings, timing; overall outcome; provenance |
| `POST /benchmark` | `note`, one `criterion`, `gamma` | the same prompt through all three decoders, measured now: prefill / decode ms, tok/s, peak GPU, acceptance, token-equality flag, speed vs no-cache |
| `POST /attention-bench` | `seq_lengths` ⊂ {256 … 4096}, `block` | reference vs tiled vs torch SDPA attention (12 heads, d = 128, fp16): ms, extra MB, tiled error |
| `POST /report` | a `/screen` result + `format` (`html` / `markdown`) | downloadable file with the outcome, criteria, explanations, numbered note, disclaimer and provenance (model, adapter + sha256, decoding mode, GPU, time) |

## Screening, step by step

1. **Split the note.**
   - A note already numbered `0. … 1. …` (the dataset's format) is parsed by the training parser,
     so evidence ids line up exactly.
   - Plain text is split at line breaks and at `. ! ?` followed by a capital letter or digit. A
     fragment ending in a common abbreviation (Dr., Mr., e.g., approx., pt., y.o., …) is rejoined.
   - Limitation: unusual abbreviations can still split a sentence. That only shifts sentence numbering;
     the model sees the same text.
2. **Prompt:** exactly the training prompt (`src/data/prompt.py`), one criterion at a time.
3. **Length:**
   - Prompts over 1,024 tokens (longer than any training example) get a warning.
   - Over 3,000 tokens → 422 with the token count. The reference attention materialises `L × L`
     scores, and that's the safe cap on 6 GB.
   - Nothing is ever silently truncated, because cutting a note could drop the evidence.
4. **Parse** with `parse_output`: an unreadable answer gives `label = null` plus the parse error,
   never a guess. Sentence ids that don't exist are dropped and reported in `warnings`.
5. **Explain** from the cited sentences (ADR 0001), never generated:
   "Labelled 'X' based on: [3] …". With no evidence, the explanation says so.
6. **Overall outcome** (ADR 0002, in order):
   - any inclusion `not included` or exclusion `excluded` → **likely ineligible**;
   - else any `not enough information` or unreadable answer → **needs review**;
   - else **likely eligible**.

   `driving_criteria` lists the criteria behind it. The response, the report and (TRIA-13) the UI
   all carry the rule and the disclaimer *"Preliminary rule-based outcome · educational screening
   prototype, not a clinical enrollment decision."*

## Errors

| Situation | Response |
| --- | --- |
| Empty / whitespace note, note > 20,000 chars, no criteria, > 30 criteria, criterion type not inclusion/exclusion, criterion < 3 chars, unknown mode, γ outside 1–8 | 422 from the schema, naming the field |
| Prompt > 3,000 tokens, note with no sentences | 422 with a plain message |
| Model still loading / failed to load | 503 |
| GPU out of memory | 503 "try a shorter note or fewer criteria" (cache emptied) |
| Anything else | 500 "Internal error; see the server log": no stack trace or path reaches the client |

## Checked live (8 Nov 2026, RTX 3050)

An unseen note (not from the dataset): a 62-year-old woman with diabetes, HbA1c 8.4%, TIA but no
stroke, eGFR 48, postmenopausal. It was screened against 6 criteria. The answers:

| Criterion | Answer | Evidence |
| --- | --- | --- |
| age 40–75 | included | [0] |
| HbA1c 7–10% | included | [1] |
| eGFR < 30 | not excluded | [4] |
| history of stroke | not excluded | [3], negation read correctly |
| pregnancy | not applicable | none |
| current insulin | not excluded | none, the "silence = absent" habit from the error analysis |

Overall: likely eligible. "Dr. Patel" stayed in one sentence. Each criterion took ≈ 90 ms prefill +
≈ 650 ms for 15 answer tokens.

`/benchmark` on one criterion gave the identical answer from all three decoders (speculative
acceptance 0.64, peak 3.8 GB with the draft). `/attention-bench` at 2,048 tokens: tiled 11 MB vs
reference 580 MB, error 4.9e-4.

## Not built

- **PDF export:** HTML is offered instead; the browser prints it to PDF.
- **Loading criteria by NCT id** from ClinicalTrials.gov: optional on the card. It would add a
  network dependency to the demo.

## Tests

`tests/test_app.py`: 20 CPU tests with a stand-in engine, covering:
- 8 validation 422s;
- one full screen call;
- all three outcome branches and the unreadable-answer branch;
- the missing-info flag;
- too-long → 422;
- loading → 503;
- the report in both formats;
- the splitter.

Plus 1 GPU test: the real model in all three modes gives the same labels, and `/benchmark` tokens match.

# App frontend (TRIA-13)

Next.js 16 + React 19 + TypeScript in `web/` (pnpm, package store in `.pnpm-store/` on D:).
- **Start:** `task app` starts the API (:8000) and the browser app (:3000) together. `task api` / `task web` start one each.
- **Backend calls:** the browser calls `/api/*`, which Next proxies to FastAPI, so there is one origin and no CORS.
- **Design process:**
  - ui-ux-pro-max produced the design system;
  - the owner chose the **case report form** direction over Drug Facts / lab report / SaaS;
  - impeccable gave the craft floor, the finish review and the fix pass.
  - Records: `PRODUCT.md`, `DESIGN.md` (Stitch-compatible format), the direction contract in `.impeccable/surfaces/screening-workbench.md`, and the sidecar `.impeccable/design.json`.
  - Google Stitch is wired in `.mcp.json` (key from the `STITCH_API_KEY` environment variable) for a later refinement pass.

## Screens

- **Screen** (`/`):
  - **Left:** the patient note (paste, or load one of 6 real validation examples via the backend's `GET /examples`). After screening it becomes the numbered sentences the model saw.
  - **Right:** the **eligibility form**: criteria rows (add, remove, type toggle, or paste a protocol block under Inclusion/Exclusion headings), the decoder (no cache / KV cache / speculative + γ), and **Screen patient**.
  - **Running:** criteria are screened **one request per criterion**, so rows tick in as they finish and the run can be stopped. The outcome then comes from `POST /aggregate`, so the ADR 0002 rule lives only in the backend.
  - **Each row:** the code (IN-n / EX-n), the criterion, evidence chips, and **all four answer boxes** with the model's ticked. On validation examples the expert's answer is circled.
  - **Selecting a row** sweeps a highlighter over its cited sentences and opens the explanation, missing-info flag, expert comparison, raw output and measured timing. **Hovering a sentence** lights the rows that cite it (mouse only).
  - **The outcome** closes the form: "Outcome: (verdict)", the reason, the drivers (links to rows), the three-step rule with the fired step bold, the disclaimer, and **Download report** (HTML / Markdown via `/report`).
- **Benchmarks** (`/benchmarks`):
  - **Decoders:** the current note plus a chosen criterion run live through all three decoders (`/benchmark`): a table of prefill / decode / total ms, tokens, tok/s, peak GPU, acceptance and speed vs no-cache, a token-equality check, and bars.
  - **Attention:** a live `/attention-bench` at chosen sequence lengths and tile size, comparing reference, tiled and PyTorch SDPA, with time, extra memory and max error. All numbers are measured on request.
- **Always:** the red-amber banner "Educational screening prototype — not a clinical enrollment decision" (one line on phones), and model status in the header (checking / loading / ready + GPU / offline).

## States and recovery (checked live)

| Situation | What the evaluator sees |
| --- | --- |
| No examples yet (backend starting) | Picker reads "Examples load when the backend is up"; retries every 5 s |
| Empty note / empty criterion | Error summary (focused) linking to each field; fields marked invalid |
| Backend down | Red status dot "Backend offline — start it with task app"; screening shows "The TrialMatch backend is not reachable. Start it with task app (or task api) and try again." |
| Model loading / GPU out of memory | Amber alert with the backend's message and "wait for the status light to turn green" |
| Unreadable model answer | Row says "answer unreadable — needs review"; the outcome counts it as needs review |
| Stopped mid-run | Unscreened rows read "not screened"; the outcome waits, with "Edit criteria" |

## Screenshots

`uv run --with playwright python -m scripts.ui_screens` (app running) drives the installed Microsoft Edge and writes `results/ui/` (no browser download):
- first run;
- validation error;
- results with a selected row;
- decoder benchmark;
- attention benchmark;
- mobile first run and mobile results.

The captures from 8 Nov were taken **on battery**, so their timings are throttled (decode about 5–10x slower). Recapture plugged in before the demo or the paper.

## Checks

- **Build and backend tests:** `pnpm build` (types + production build) is clean; pytest covers `/examples` and `/aggregate` (23 app tests).
- **Responsive:** desktop at 1440 px; mobile at 390 px with touch emulation. The page never overflows horizontally, and answer boxes move under the text with labels.
- **impeccable finish review** (in-thread fallback): disposition **fix** → four fixes:
  - mobile row reflow;
  - removed the kicker above the verdict;
  - drawn stepper icons;
  - touch-safe hover.

  All four scored resolved, and one regression (ring spacing) was fixed, so the disposition is **ship** for the scored fixes.
- **Not run:** impeccable's downloaded engine binary (concept dice, detector, comp tools), deliberately. The direction was chosen by the owner from three grounded options (seed: none, unavailable).

## Clarity pass (owner feedback: "confusing steps, benchmarks unclear")

impeccable `clarify` / `onboard`:
- **Screen page:**
  - a three-step strip (Patient note → Trial criteria → Results) with done, current and to-do states;
  - the form is renamed **Trial criteria**, and the primary action is **Check eligibility**;
  - the generation method moved into a collapsed **Advanced** row with one plain line per method ("all three give the same answer; they differ only in speed");
  - the first-run box now just says *try an example first*.
- **Results lead with a summary line:** counts (met / not met / can't tell / don't apply), an **Overall: <outcome> · why ↓** link to the rule block, and a plain legend (filled box = model, dashed circle = expert).
- **Benchmarks:**
  - tabs are named **Speed** and **Memory**, each opening with a plain-language explanation (what a token is, what each method does, what attention and tiling are);
  - after a run, a **What this run shows** box states the findings in sentences, computed from that run's numbers: identical answers, the fastest method and by how much, acceptance and why speculative did or didn't win, the memory ratio, error size and why tiled is slower;
  - column headers are in plain words (reading the prompt, writing the answer, GPU memory, drafts accepted).

## Google Stitch exploration (9 Nov 2026)

Stitch was connected through MCP: `.mcp.json` reads `STITCH_API_KEY` from the environment, and `.claude/settings.local.json` approves the server.
- **Setup:** project "TrialMatch · criterion screening", with our tokens loaded as its design system (Public Sans, form blue, cool white, 4 px corners, the DESIGN.md rules).
- **Generation:** it generated the Screen page from a description of the real content. Saved as `results/ui/stitch-exploration.png`. **Note:** it contains invented placeholder data.

**Rejected** (invented or misleading, against "real data only"):
- a latency readout, a named user, a patient ID and encounter details;
- "confidence" scores, and a made-up "confidence threshold" rule;
- an "Audit & Export" tab;
- decorative monospace everywhere.

**Adopted** (each uses data the app really has):
1. **Citation tags on every note sentence,** showing which rows cite it, clickable, plus a "7 sentences · 5 cited" count.
2. **Expert disagreement shown inline** on the row ("Expert: not included") without selecting it.
3. **A one-line progress bar** with connectors; the last step shows the outcome as a tag.
4. **Evidence shown as quoted sentences** in the row detail.
5. **Boxed sentence numbers.**
