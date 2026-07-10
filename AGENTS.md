# AGENTS.md — instructions for AI contributors

## Build / test / lint

```bash
uv sync --extra dev
uv run pytest tests            # assert pytest's OWN exit code, never `| tail`
uv run ruff check . && uv run ruff format --check .
uv run codespell src tests README.md
```

## Hard rules

1. **Never run tests marked `cloud` or `licensed`.** `cloud` spends real
   tidy3d FlexCredits; `licensed` consumes a Lumerical license seat. The
   default `pytest tests` deselects nothing locally — CI filters with
   `-m "not physics and not gpu and not cloud and not licensed"`; do the same
   for anything expensive.
2. **`run()` is the only Solver method allowed to spend money/licenses/GPU.**
   `validate()`/`build()`/`estimate()` must stay offline and deterministic —
   the conformance suite (`tests/conformance/`) enforces this; don't weaken it.
3. **Validate through the exact artifact users execute** (example file, fresh
   interpreter), not through bespoke scripts that share your session's import
   state — see finding F10 (the tidy3d.web lazy-import crash) in `CHANGELOG.md`.
4. **Keep `ROADMAP.md` current** if you execute roadmap work: it is the living
   plan and the hand-off protocol between contributors.
5. Recorded artifacts under `tests/recorded/` must be sanitized (no license
   tokens, hostnames, API keys) — grep before committing.
6. All GitHub Actions must be pinned by full commit SHA.

## Orientation

- `src/gds_fdtd/` — package; solvers in `solvers/` (ABC + tidy3d, lumerical,
  beamz adapters), canonical results in `smatrix.py`, Tier-B pipeline in
  `grid.py` / `modes.py` / `extraction.py`, orchestration in `convergence.py` /
  `caching.py` / `validation.py`.
- `ROADMAP.md` — the living plan; read "Where we are" first. `HANDOFF.md` — the
  development arc + the live-validation runbook (tidy3d/Lumerical).
- `examples/` — every feature has a runnable example; each follows the
  standardized flow: geometry plot → validate/build/estimate → S-params plot →
  field plot.
