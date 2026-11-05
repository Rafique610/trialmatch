# Security and settings

- Never read env vars directly in code. All settings go through the Pydantic
  `Settings` class in `src/config.py` (prefix `TM_`).
- Never hard-code credentials (tokens, keys, passwords) anywhere: code,
  configs, scripts, compose files. They live in `.env` / `.env.local`
  (git-ignored) or the tool's own login (`hf auth login`).
- Never print credentials in logs, errors or chat.
- Backend code: validate input with allow-lists, fail safely, keep errors
  generic to the client. Use backend-security-coder when writing security code.
