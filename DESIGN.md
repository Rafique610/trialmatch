---
name: TrialMatch
description: Criterion-level clinical-trial screening, rendered as a case report form with highlighter-marked evidence.
colors:
  ground: "#eef2f7"
  paper: "#ffffff"
  paper-2: "#f7f9fc"
  ink: "#14233f"
  ink-2: "#3f4d66"
  ink-3: "#5f6b82"
  rule: "#d5deea"
  rule-strong: "#a7b6cc"
  form-blue: "#1d4fa0"
  form-blue-tint: "#e7eef9"
  exclusion-violet: "#7a2f8f"
  highlighter: "#ffe46b"
  highlighter-soft: "#fff4bd"
  yes: "#17663f"
  yes-tint: "#e5f2ea"
  no: "#a8201a"
  no-tint: "#fbe9e7"
  unknown: "#8a4a00"
  unknown-tint: "#fbefdc"
  not-applicable: "#4b576c"
  not-applicable-tint: "#edf0f5"
  banner: "#a9381a"
typography:
  body:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.3
  section-title:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.005em"
  verdict:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 750
    lineHeight: 1.2
    letterSpacing: "-0.015em"
  data:
    fontFamily: "ui-monospace, Cascadia Mono, Consolas, monospace"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.5
rounded:
  form: "3px"
  box: "2px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "20px"
components:
  button-primary:
    backgroundColor: "{colors.form-blue}"
    textColor: "{colors.paper}"
    rounded: "{rounded.form}"
    padding: "0 14px"
    height: "34px"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.form}"
    padding: "0 14px"
    height: "34px"
  answer-box:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.box}"
    size: "18px"
  answer-box-yes:
    backgroundColor: "{colors.yes}"
    textColor: "{colors.paper}"
    rounded: "{rounded.box}"
    size: "18px"
  evidence-chip:
    backgroundColor: "{colors.highlighter-soft}"
    textColor: "{colors.ink}"
    rounded: "{rounded.box}"
    height: "20px"
  banner:
    backgroundColor: "{colors.banner}"
    textColor: "{colors.paper}"
    padding: "6px 16px"
---

# TrialMatch design system

## Overview

The screening workbench is a clinical-trial **case report form** (CRF): a ruled sheet on a cool desk, with numbered rows (IN-1, EX-2) and a four-box answer column on every criterion. The model's answer is the ticked box, and the other three answers stay visible, so uncertainty is never collapsed. Evidence is a **highlighter pen** on the printed note: selecting a row swipes yellow across the sentences it cites. It's an Operate surface: familiar controls and dense, scannable rows. The brand lives in precise details, not decoration. The source of truth is `web/app/globals.css` (tokens) and `web/app/form.css` (form components).

## Colors

Restrained. Neutrals carry the sheet, form blue marks action and selection, and one highlighter yellow marks evidence.
- **Ground `#eef2f7` vs paper `#ffffff`:** the desk under the form; sheets sit on it with a 1 px rule, never a shadow.
- **Ink `#14233f`:** all text. `ink-2` and `ink-3` are secondary and meta; every text colour passes 4.5:1 on paper.
- **Form blue `#1d4fa0`:** primary actions, the active tab, selected rows (tint `#e7eef9`), and inclusion row codes. Exclusion row codes use violet `#7a2f8f`, so the two sections read apart.
- **Highlighter `#ffe46b`:** only for cited evidence (the mark, the evidence chips, a hovered-sentence row link). Never decoration.
- **Answer states:**

  | State | Labels | Ink | Tint |
  | --- | --- | --- | --- |
  | yes | included / not excluded | `#17663f` | `#e5f2ea` |
  | no | not included / excluded | `#a8201a` | `#fbe9e7` |
  | unknown | not enough information | `#8a4a00` | `#fbefdc` |
  | not-applicable | not applicable | `#4b576c` | `#edf0f5` |

  A state is never carried by colour alone: the column header names it and the box carries a tick.
- **Banner `#a9381a`:** the persistent prototype warning, white text at 5.6:1.

## Typography

One workhorse family, **Public Sans**. It's the US government design-system face, at home on regulatory forms, and is self-hosted through `next/font`. Tabular figures are on globally.
- **Scale:** a tight one: 12 (labels) / 13 / 14 (body) / 15 (section titles) / 24 (the verdict).
- **Monospace:** only for data, such as row codes, sentence ids, raw model output and measurements, never as a "technical" costume.
- **Labels:** sentence case. No tracked uppercase, and no eyebrow labels above headings.

## Layout

- **Desktop:** a 5/12 note panel and a 7/12 form, 20 px gutter, page max 1480 px with 24 px padding.
- **Sticky chrome:** the banner is the only sticky bar (`--banner-h: 31px`). The note panel and the CRF column headers stick just below it.
- **Grid:**
  - the criterion column is `minmax(0, 1fr)`, plus four 76 px answer columns separated by hairlines;
  - details span the full row, indented to the text column.
- **Below 1080 px:** one column, `minmax(0, 1fr)` so long titles never widen the page.
- **Below 720 px:** the answer boxes drop under the criterion text, each labelled; the column headers hide; the banner shortens to one line.

## Elevation & Depth

Flat. Sheets are separated from the desk by a 1 px `rule` border, not shadows. Depth comes from the sticky banner and sticky column headers only. Focus uses a 2 px form-blue outline, plus a 3 px tint ring on inputs.

## Shapes

- **Square-cornered form:** 3 px radius on sheets, inputs and buttons; 2 px on the 18 px answer boxes and chips.
- **Rules:** hairline row rules `#edf1f6`, section rules `rule`, and a 2 px ink rule that closes the form above the outcome.
- **Circles:** the only one is the expert's dashed ring around the box the annotator chose, like a monitor circling a paper CRF.

## Components

- **CRF row:**
  - the code, the criterion, and the evidence chips (or "no evidence") under the text, plus the four answer boxes;
  - states: pending (muted), active (pulsing box fill + "screening…"), done, selected (blue tint, detail open), linked (highlighter-soft, when a hovered sentence is cited), skipped ("not screened");
  - the tick animates in over 160 ms.
- **Note sentence:** a mono id gutter plus text. When cited, the mark sweeps left to right over 380 ms (the one authored motion) and the id turns bold ink.
- **Outcome:** the heading reads "Outcome: <verdict>", tinted by state, with an icon. Below it: the reason, drivers as row-code links, the three-step rule with the fired step in bold, the disclaimer, and the report actions.
- **Buttons:** primary (form blue), secondary (paper with a strong rule), and quiet (no border). All are 34 px tall, with Lucide icons at 14–16 px.
- **Segmented control:** the decoder modes. The pressed segment gets a blue tint plus a 2 px inset underline.
- **Alerts:** a red tint for errors, an amber tint for "not ready". Each has the problem in bold, a recovery line, and links to the fields. Error summaries take focus.
- **Benchmark tables and bars:** tabular mono figures, right-aligned. Bars are drawn from measured values with the values printed beside them; there are no chart images.

## Do's and Don'ts

- **Do** keep all four answers visible on every row. "Not enough information" is a first-class answer.
- **Do** tie every highlight to cited sentence ids, and every number to a backend measurement.
- **Do** show the outcome rule and its drivers wherever the outcome appears.
- **Don't** add cards, gradients, glass, drop shadows, coloured side stripes, eyebrow labels or emoji/glyph icons.
- **Don't** use the highlighter yellow for anything but evidence, or state colours for decoration.
- **Don't** imply clinical validity: the banner and the "rule, not prediction" disclaimer stay.
