# Docs map

Start at `AGENTS.md` for routing. This file says where everything else is.

| Folder | What lives there |
| --- | --- |
| `docs/invariants/` | Rules that hold for any project on this stack. Load only the one your task needs. |
| `docs/domain/trialmatch/` | Facts about this project: models, dataset, decisions, phase status. |
| `docs/adrs/` | One file per decision: context, options, choice, consequences. |
| `docs/research/` | Literature notes and model comparison (TRIA-2). |

## Invariants

| Doc | Load when |
| --- | --- |
| `core.md` | Always: folders, file size, real data only, Ponytail |
| `processes.md` | Cards, commits, README, `.memory/` log |
| `tooling.md` | uv, Taskfile, pnpm, paths |
| `ml.md` | Data splits, training, evaluation, benchmarks |
| `backend.md` | FastAPI app (`src/app/`) |
| `frontend.md` | Next.js app (`web/`) |
| `security.md` | Settings, tokens, credentials |

## Plan

The plan is the Synverk board **TrialMatch** (key `TRIA`). Each card links a flow
diagram that explains what that card builds and why.
