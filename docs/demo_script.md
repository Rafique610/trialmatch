# Demo video script (5–7 min, TRIA-15)

**Before recording:**
- Plug the laptop in (on battery the GPU is throttled 5–10×).
- Run `task app` and wait for the green "Model ready" light.
- Do one warm-up screen, so the first take isn't slow.
- Close other apps.

**Recording:** Windows: press `Win + Alt + R` (Xbox Game Bar) to start and stop. OBS also works. Record the browser window at 1440 px or wider. Upload the video unlisted to YouTube or Google Drive and paste the link into the README and the paper.

## Shot list (about 6 minutes)

| Time | Show | Say (short) |
| --- | --- | --- |
| 0:00–0:30 | The Screen page, banner visible | "TrialMatch checks one patient note against a trial's criteria, one criterion at a time. It's a Qwen2.5-1.5B model fine-tuned with QLoRA on expert annotations. This is an educational prototype, not a clinical tool." |
| 0:30–1:00 | Point at the three steps and the prototype banner | "Three steps: the note, the criteria, the results. Everything runs locally on a 6 GB laptop GPU." |
| 1:00–1:45 | **Unseen note:** paste the note below into the note box | "This note isn't from the dataset; I wrote it for the demo." |
| 1:45–2:15 | **Criteria:** click *Paste a protocol block*, paste the block below, then *Add* | "The protocol's inclusion and exclusion criteria, pasted as a block." |
| 2:15–3:15 | Press **Check eligibility**; rows tick in one by one; click the stroke row, then the insulin row | "Each row shows all four possible answers with the model's ticked. Clicking a row highlights the sentence it relied on. Here it reads 'TIA but no stroke' correctly. Insulin isn't mentioned, and it says *not excluded*: the 'not mentioned means absent' habit from our error analysis." |
| 3:15–3:40 | Point at the summary line and the outcome block | "The overall outcome is a fixed rule over these answers, shown with the step that fired. It's not a prediction." |
| 3:40–4:00 | **Export:** click *Download report*, open the HTML | "The report keeps the evidence, the rule and the provenance: model, adapter, decoding, time." |
| 4:00–5:00 | **Benchmarks → Speed:** *Compare the three methods* | "Our own decoders: no cache, KV cache and speculative. All three produce exactly the same answer. KV cache is fastest; speculative doesn't pay off on this GPU, because the 0.5B draft costs almost as much per token as the 1.5B model." |
| 5:00–5:30 | **Memory** tab: *Compare attention methods* | "Our tiled online-softmax attention gives the same result as the standard one, but needs far less memory, because it never builds the full length × length table." |
| 5:30–6:10 | **Invalid input:** back to Screen, *Edit note*, clear the note, press *Check eligibility* | "Bad input is caught with a clear message pointing to the field. Fix it and run again." Then load an example and show the dashed circle marking the expert's answer. |
| 6:10–6:30 | Close on the results | "Code, adapter, paper and logs are linked in the README. Thanks." |

## Unseen note (paste in step 1:00)

```
A 62-year-old woman with a 10-year history of type 2 diabetes is seen in clinic. Her HbA1c last month was 8.4%. She takes metformin 1000 mg twice daily. Dr. Patel notes she had a transient ischemic attack two years ago but no stroke. Her eGFR is 48 mL/min. She is postmenopausal.
```

## Protocol block (paste in step 1:45)

```
Inclusion criteria:
- Age 40 to 75 years
- HbA1c between 7.0% and 10.0%
Exclusion criteria:
- eGFR below 30 mL/min
- History of stroke
- Pregnant or breastfeeding
- Current insulin therapy
```

If the timing runs short, add an example (*Load an example*) to show the expert comparison: the dashed circle and the "Expert: …" line on rows where the model disagrees.
