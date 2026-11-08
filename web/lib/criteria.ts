import type { Criterion, CriterionType } from "./api";

/** Form row codes: inclusion rows IN-1, IN-2 …; exclusion rows EX-1 … (numbered within their type). */
export function rowCodes(criteria: Criterion[]): string[] {
  const n = { inclusion: 0, exclusion: 0 };
  return criteria.map((c) => `${c.type === "inclusion" ? "IN" : "EX"}-${++n[c.type]}`);
}

/**
 * A pasted protocol block → criteria. Lines under a heading containing "inclusion" / "exclusion" take that type
 * (inclusion until told otherwise); bullets, numbering and blank lines are stripped.
 */
export function parseBlock(text: string): Criterion[] {
  let type: CriterionType = "inclusion";
  const out: Criterion[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const heading = line.replace(/[:\s]+$/, "").toLowerCase();
    if (/^(key\s+)?(inclusion|exclusion)(\s+criteria)?$/.test(heading)) {
      type = heading.includes("exclusion") ? "exclusion" : "inclusion";
      continue;
    }
    const textOnly = line.replace(/^([-*•·]|\d+[.)]|[a-z][.)])\s+/i, "").trim();
    if (textOnly.length >= 3) out.push({ type, text: textOnly });
  }
  return out;
}

export const LIMITS = { noteChars: 20_000, criteria: 30, criterionMin: 3, criterionMax: 2_000 };

/** Client-side mirror of the backend schema, so the evaluator sees problems before a round trip. */
export function validate(note: string, criteria: Criterion[]): { field: string; message: string }[] {
  const errors: { field: string; message: string }[] = [];
  if (!note.trim()) errors.push({ field: "note", message: "Paste a patient note or load an example." });
  else if (note.length > LIMITS.noteChars) errors.push({ field: "note", message: `The note is ${note.length.toLocaleString()} characters; the limit is 20,000.` });
  if (criteria.length === 0) errors.push({ field: "criteria", message: "Add at least one inclusion or exclusion criterion." });
  if (criteria.length > LIMITS.criteria) errors.push({ field: "criteria", message: `${criteria.length} criteria; the limit is 30 per screen.` });
  criteria.forEach((c, i) => {
    if (c.text.trim().length < LIMITS.criterionMin) errors.push({ field: `criterion-${i}`, message: `Row ${i + 1}: write the criterion (at least 3 characters).` });
  });
  return errors;
}
