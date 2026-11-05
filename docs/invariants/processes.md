# Processes

## Cards

- One Synverk card at a time. Before coding, tell the user which flow to read
  (name + link) and what the card builds.
- Test it yourself, fix obvious errors, then hand off in plain language with
  the exact terminal commands to verify it.
- Paste the check output as a card comment and move the card to review. Only
  a person approves; the agent never approves its own card.
- Keep a running list of things noticed along the way (gaps, config, missing
  checks) and flag them to the user instead of silently fixing or skipping.

## ADRs

- One file per decision in `docs/adrs/NNNN-short-name.md`: Context, Options,
  Decision, Consequences, How to revert. Rejected options stay in the file —
  the report needs them.

## README

- Update `README.md` on every card before handoff: new capabilities, paths,
  commands. Every commit carries an up-to-date README.

## Commits

- One commit per approved card (or explicitly batched group). Subject under 72
  characters, e.g. `chore(setup): scaffold uv project and docs tree`.
- Covers only that commit's diff. No essay body. No agent co-author line.
- The agent commits after approval and pushes to `origin` once a remote
  exists. Merging to `main` from another branch, force-pushing and tagging
  need a yes each time.

## `.memory/tasks.md` (git-ignored)

- One folder per day, e.g. `.memory/05-november-2026/tasks.md`, one file per day.
- Starts with a one-line day verdict and a bold summary row
  (X done / Y approved / Z changed), then a task-board table.
- Each row: time in 12-hour format with zone and UTC, e.g.
  `1:48 PM (UTC+05:00 · 08:48 UTC)` — always from the real clock, never guessed;
  card; status badge (✅ Approved / 🔄 In progress / ⚠️ Needs changes);
  what changed with file paths, detailed enough to reverse it.
- A separate section lists every rule change that day: what changed, why,
  how to revert.
