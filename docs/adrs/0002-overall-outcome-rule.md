# ADR 0002: Overall patient outcome is a stated rule, not a prediction

Date: 5 November 2026 · Status: accepted (project owner)

## Context

The dataset labels one criterion at a time. It has no patient-level
"eligible for this trial" label. The app must still show an overall
preliminary outcome (rubric G2), and the brief requires it to be described as
an aggregation by explicit rules, not as a ground-truth outcome.

## Decision

Checked in this order:

1. **Likely ineligible**: any inclusion criterion is `not included`, or any
   exclusion criterion is `excluded`. One hard failure is enough.
2. **Needs review**: otherwise, any criterion is `not enough information`.
   Missing evidence is never treated as proof either way.
3. **Likely eligible**: otherwise. Every inclusion criterion is `included` or
   `not applicable`, and every exclusion criterion is `not excluded` or
   `not applicable`.

Unparseable model output for a criterion counts as `needs review`, never as a pass.

The app shows the criteria that drove the outcome (linking to their evidence)
and labels it "preliminary rule-based outcome · educational screening
prototype, not a clinical enrollment decision".

## Consequences

- One wrong criterion prediction can flip the outcome. This is stated in the
  report's Limitations.
- The outcome is never scored against data, because no patient-level labels exist.

## How to revert

Change the rule in the backend aggregation function (TRIA-12) and this ADR together.
