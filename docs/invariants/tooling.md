# Tooling

- Python: `uv` only. Run project code with `uv run ...`, never a bare
  `python`. Dependencies live in `pyproject.toml` + `uv.lock`; add them with
  `uv add` (dev tools: `uv add --dev`).
- GPU PyTorch comes from the PyTorch CUDA index pinned in `pyproject.toml`
  (`[tool.uv.sources]`). Don't install torch any other way.
- Node: `pnpm` with a committed lockfile.
- Every command a person runs regularly gets a `Taskfile.yml` entry
  (`task <name>`), so nobody memorises raw commands.
- Paths come from `src/config.py` (`settings.data_dir`, ...), never hard-coded
  strings scattered in code. Use `pathlib`.
- Docker, if ever added: two-stage Dockerfiles. On DNS/network failures retry
  at least 3 times before giving up.
