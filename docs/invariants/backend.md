# Backend (FastAPI)

- Inside the feature folder (`src/app/`): `routers/`, `schemas/`, plus
  `main.py`. Add `repositories/` only if there is storage to access.
- `main.py` only creates the app, adds middleware and includes routers.
- Routers are thin: wire an endpoint to a function. No business logic there.
- Every endpoint with a body or meaningful query params declares a Pydantic
  model in `schemas/`, with constraints (enums, lengths, ranges), so bad input
  gets a clear 422 and `/docs` stays typed and testable.
- After building an endpoint, tell the user how to try it in `/docs`.
- Errors never leak stack traces or file paths to the client.
- Storage, if needed: SQLite through `sqlite3` with `?` parameters only.
