# Core rules (always loaded)

## Folder structure: one folder per feature

- Each feature gets its own folder holding everything it needs (code, helpers,
  its own types). Never split by technical layer (`models/`, `services/`,
  `utils/` at the top level).
- A new feature gets a new folder. Never mix two features' responsibilities.

## File size

- A `.py` file over ~300 lines does too much: split it or apply a fitting
  pattern. No god files.
- When a design pattern is used, tell the user which one and why. Don't
  over-engineer.

## Real data only

- Never show invented numbers in UI, API responses, reports or the paper.
  Every number comes from a real computation, file or measurement.
- If something can't be measured honestly, omit it or say so. Never fill
  with a plausible-looking placeholder.

## Ponytail ladder (active for all code)

Write the simplest thing that works. Before writing code, ask in order:

1. Does this need to exist at all? (YAGNI)
2. Is it already in this codebase? Reuse it.
3. Does the standard library do it?
4. Does a native platform feature cover it?
5. Does an already-installed dependency solve it? Never add a package without need.
6. Can it be one line?
7. Only then: write the minimum code that works.

- No unrequested abstractions: no factory for one product, no speculative interface.
- Bug fixes address the root cause: one shared guard over many caller patches.
- Every non-trivial piece of code gets one minimal runnable check.
