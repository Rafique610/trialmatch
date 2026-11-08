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
