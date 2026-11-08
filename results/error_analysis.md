# Error analysis · fine-tuned model on the test split

Model: Qwen2.5-1.5B-Instruct + adapter `Aduhere/trialmatch-qwen2.5-1.5b-qlora`, fp16, greedy with the model's default repetition penalty 1.1
(`results/finetuned_test/`, protocol in `results/test_protocol.md`). 206 test criteria,
60 wrong labels (accuracy 0.709), 0 parse errors. Sentence ids are the note's own numbering
(`[k]`). `G` = cited by the expert, `P` = cited by the model. Per-row data:
`results/finetuned_test/per_record.jsonl`, matched by `annotation_id`.

## Where the 60 label errors are

Confusion of the fine-tuned model (gold → predicted), largest cells first:

| Gold → predicted | Type | Count | Pattern |
| --- | --- | --- | --- |
| not enough information → not excluded | exclusion | 23 | treats "not mentioned" as "does not have it" |
| not excluded → not enough information | exclusion | 7 | the reverse: abstains where the expert assumed absence |
| not included → not enough information | inclusion | 6 | abstains when the note shows a *different* procedure/condition |
| included → not enough information | inclusion | 6 | misses implied findings ("tachycardic" for HR ≥ 90) |
| not applicable → not excluded / not enough information | exclusion | 6 + 1 | misses that a criterion cannot apply (sex, age) |
| others (≤ 2 each) | both | 11 | polarity flips, over-reach, NA vs NEI |

The two biggest cells point in opposite directions: 23 + 7 errors on the same question,
"the note says nothing about X". The gold labels themselves are not consistent here (case 9).

Evidence-only errors (label right, cited sentences differ from the expert): 45 rows. 25 are
`not enough information` rows, and in 17 of those the expert cited nothing while the model
cited background sentences.

## Cases

### 1 · Absent evidence read as proof
`annotation_id 193` · patient sigir-201411 · trial NCT01526382 · exclusion
**Criterion:** "Patients were moribund."
**Gold:** not enough information, evidence [0, 1, 2, 3, 4] · **Pred:** not excluded, evidence [0, 1, 2]
- [2] GP "On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic."
- [3] G "Her body temperature is normal and her blood pressure is 80/60."

**Cause: missing information treated as a negative.** The note never says whether she is
moribund. A BP of 80/60 with tachycardia (shock) makes it a real open question. The model
read "no mention" as "not moribund". This is the single largest error pattern (23 rows).

### 2 · Implied finding missed, wrong sentence cited
`186` · sigir-201411 · NCT01526382 · inclusion
**Criterion:** "Heart rate of at least 90/min;"
**Gold:** included [2] · **Pred:** not enough information [3]
- [2] G "…as well as tachypneic and tachycardic."
- [3] P "Her body temperature is normal and her blood pressure is 80/60."

**Cause: clinical inference not made.** "Tachycardic" means HR > 100, so it meets ≥ 90.
The model wanted a number and cited the vital-signs sentence that has no heart rate.

### 3 · Right evidence, wrong label
`187` · sigir-201411 · NCT01526382 · inclusion
**Criterion:** "A respiratory rate of at least 20/min or a PaCO2 of 32mmHg or lower or the use of mechanical ventilation;"
**Gold:** included [2] · **Pred:** not enough information [2]
- [2] GP "…as well as tachypneic and tachycardic."

**Cause: clinical inference not made (label-only error).** Same as case 2, but here the model
found the right sentence and still abstained. Evidence F1 counts this row as perfect, while
the label is wrong. This is why label and evidence are scored separately.

### 4 · Negation + near-synonym confusion
`176` · sigir-201410 · NCT02264964 · inclusion
**Criterion:** "No medical history of central vena catheterization."
**Gold:** included [0] · **Pred:** not included [0]
- [0] GP "A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning."

**Cause: negation and terminology.** The criterion is negated ("No history of…"), and the
note's procedure is *arterial* (femoral artery), not a central *venous* catheter. The model
matched "catheterization" and flipped to not included.

### 5 · Polarity flip despite supporting evidence
`383` · sigir-201421 · NCT00997100 · inclusion
**Criterion:** "Present with active SLE disease with at least one of the following symptoms: i) Arthritis - > 2 joints with pain and signs of inflammation … ii) Inflammatory-type skin rash iii) Oral ulcers"
**Gold:** included [0, 1] · **Pred:** not included [0, 1]
- [0] GP "A 21-year-old female is evaluated for progressive arthralgias and malaise."
- [1] GP "On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, … and swelling and tenderness of her wrists and ankles."

**Cause: label contradicts the model's own evidence.** Both cited sentences satisfy the
criterion: a malar rash, and swollen, tender wrists and ankles (> 2 joints). The explanation
is built from the cited sentences (ADR 0001), so a reader would see evidence that argues
against the label. This is one reason the app marks the outcome as preliminary.

### 6 · Temporal requirement not checked
`678` · sigir-20158 · NCT02562040 · inclusion
**Criterion:** "Diagnosis of mild sleep-disordered breathing (MSDB) defined as meeting all of the following criteria: Caregiver report of habitual snoring that occurs most of the night on at least three nights per week, and has been present for at least three months …"
**Gold:** not enough information [0] · **Pred:** included [0]
- [0] GP "A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings."

**Cause: temporal / frequency condition ignored.** The note confirms snoring but gives no
frequency (≥ 3 nights a week) or duration (≥ 3 months), and the criterion says "all of the
following". The model accepted it on the topic match.

### 7 · Unsupported claim: partial match taken as full
`762` · sigir-201513 · NCT00315042 · inclusion
**Criterion:** "A sore and scratchy throat and/or pain on swallowing (odynophagia) together with at least 2 of the following clinical signs:"
**Gold:** not enough information [0] · **Pred:** included [0]
- [0] GP "A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes."

**Cause: unsupported claim.** Dysphagia comes close to the first part. But the list of
"at least 2 clinical signs" is cut off in the criterion text itself, so it can't be checked.
The model claimed inclusion anyway.

### 8 · Wrong label set: applicability missed (two directions)
`751` · sigir-201513 · NCT01255670 · exclusion · **Criterion:** "breast-feeding"
**Gold:** not applicable [] · **Pred:** not enough information []
(note: [0] "A 5-year-old boy presents …")

`408` · sigir-201421 · NCT00997100 · exclusion · **Criterion:** "Female subject of child-bearing potential who is pregnant or lactating."
**Gold:** not enough information [] · **Pred:** not applicable [0]
- [0] P "A 21-year-old female is evaluated for progressive arthralgias and malaise."

**Cause: wrong label set (`not applicable` vs `not enough information`).** For a 5-year-old
boy, breast-feeding cannot apply, but the model abstained. For a 21-year-old woman, the
criterion does apply (she is of child-bearing potential), and her pregnancy status is unknown.
Here the model answered not applicable, citing the very sentence that shows the criterion applies.

### 9 · Contradictory annotation: the "not mentioned" convention
`193` (case 1, moribund, **gold NEI**) vs `429` · sigir-201422 · NCT00236912 · exclusion
**Criterion (429):** "Malnourished with low albumin" · **Gold:** not excluded [] · **Pred:** not enough information []
(also `425` "APACHE II (health) score > 25", `428` "Low platelet count (bleeds easily)": gold not excluded, pred NEI)

**Cause: contradictory annotation.** None of these notes mentions the condition. For 429,
425 and 428 the expert assumed "not mentioned → not excluded". For 193 (and 23 similar rows)
the expert labelled the same situation `not enough information`. The model learned both
conventions and applies them unpredictably. Some of the 30 errors in the two largest cells
are probably disagreements about this convention rather than reading mistakes. Following the
task rule, we do not treat absent evidence as proof, so the gold labels are reported as they
are, not "corrected".

### 10 · Contradictory annotation: pregnancy criterion for a man
`322` · sigir-201417 · NCT01745731 · exclusion
**Criterion:** "Pregnant women or women of childbearing age who do not have adequate contraception."
**Gold:** not excluded [0] · **Pred:** not applicable []
- [0] G "A 48-year-old white male with history of common variable immunodeficiency (CVID) …"

**Cause: contradictory annotation.** In `750` ("pregnancy", 5-year-old boy) the expert
used `not applicable`, the label the model gives here. For this man the expert chose
`not excluded`. The model's answer arguably matches the label definitions
(`src/data/prompt.py`) better than the gold does.

### 11 · Evidence-only error: over-citing when abstaining
`192` · sigir-201411 · NCT01526382 · inclusion
**Criterion:** "no clinical evidence of left atrial hypertension; and requirement for positive pressure ventilation."
**Gold:** not enough information [] · **Pred:** not enough information [0, 1, 2, 3]
- [0]–[3] P age and presentation, "She denies trauma.", examination, vital signs

**Cause: evidence-only error.** The label is right, but the model cites four background
sentences as "evidence" for having no information. An empty set is the right answer. It's
the most common evidence error (17 of the 45 evidence-only rows) and a main reason
evidence precision (0.517) trails recall (0.660).

## Parse errors

None. All 206 answers were valid JSON with a label from the right set, so the
`parse error` category from the card's list didn't occur on test.

## Takeaways

1. The largest share of errors comes from one question: what does silence in the note mean?
   Even the expert labels don't agree on it (cases 1, 9).
2. The 1.5B model misses simple clinical implications (cases 2, 3) and checks only part of
   multi-part, temporal or frequency criteria (cases 6, 7).
3. In cases 3, 5 and 8, label and evidence disagree with each other, so cited evidence alone
   does not guarantee a correct label. The app shows the cited sentences next to each label
   and marks the overall outcome as preliminary (ADR 0002).
