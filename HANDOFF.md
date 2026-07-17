# gds_fdtd — Development Handoff

**Audience:** the next engineer/agent picking up the project. This file holds
the compressed history and the working conventions; the live plan is
[`ROADMAP.md`](ROADMAP.md), per-engine verification evidence is
[`SOLVER_STATUS.md`](SOLVER_STATUS.md), and user-facing docs live at
<https://siepic.github.io/gds_fdtd/>.

**State:** `v0.6.0` is released (tagged 2026-07-15, signed GitHub release).
The three engines were validated **live** against each other during the 0.6
arc — tidy3d ↔ Lumerical within 0.0033 dB, beamz within 0.052 dB on the
identical job — and that agreement is locked into CI through recorded
artifacts. All-extras branch coverage is ≥90 (gated), `mypy --strict` passes
on the whole package (required check), and the examples are 13 executed
notebooks with committed outputs. The one loose end is the PyPI trusted
publisher (owner action; PyPI still serves 0.4.0 — see ROADMAP).

## 1. What gds_fdtd is (one screen)

Turns a photonic-chip **GDS layout + a technology YAML + a `SimulationSpec`**
into a 3D FDTD run on the engine of your choice, and returns one canonical
S-matrix:

```python
from gds_fdtd import get_solver, SimulationSpec, Technology

solver  = get_solver("tidy3d" | "lumerical" | "beamz")(component, tech, SimulationSpec())
smatrix = solver.run()          # the ONLY call that spends money / license / GPU
```

EDA-agnostic on the front (KLayout/SiEPIC, gdsfactory ≥9), solver-agnostic on
the back. Same `(component, technology, spec)` in, same `SMatrix` out, on all
engines. Architecture map: docs → *API reference*; frontends guide: docs →
*Frontends*.

## 2. The development arc (0.1 → 0.6)

| Version | What it was |
|---|---|
| **0.1–0.3** (2023–24) | Rough SiEPIC research scripts: `lum_tools.py` / `t3d_tools.py`, per-engine one-offs, PCell/CML helpers. |
| **0.4.0** (2025-08) | First "architecture": a `fdtd_solver` base + per-engine subclasses. (Its changelog over-promised — Touchstone/JSON/energy checks were only *real* in 0.5.) |
| **0.5.0** (2026-07-08, PR #23) | **The solver-agnostic rewrite**: `Solver` ABC + registry, canonical `SMatrix`, validated `SimulationSpec`, `Technology` v1/v2 + refractiveindex.info materials, Tier-B kernel pipeline, `JobSpec` + CLI, honest coverage (was a gamed ~100% over a 16% reality). 19 audited legacy bugs fixed. |
| **0.6.0** (2026-07-15) | Removed the entire pre-0.5 public surface (supported API: `get_solver` + `Technology` + `SimulationSpec` + `SMatrix`). Executed-notebook example curriculum + recorded cross-engine artifacts (crossing, S-bend, escalator, PBS, PSR). Docs overhaul. `GdsFdtdError` hierarchy, whole-package `mypy --strict`, property-based tests, real beamz e2e in CI, deprecation policy. |

Full details: `CHANGELOG.md`; migration table under the 0.6.0 entry.

## 3. Live-validation runbook (when refreshing engine evidence)

0. Baseline (offline, matches CI):
   `uv run pytest -m "not physics and not gpu and not cloud and not licensed"`.
1. Examples are the golden artifacts — execute the committed notebooks in a
   fresh interpreter (all free: beamz + local mode solver + recorded data).
2. tidy3d: `BUDGET_FC=0.5 uv run python scripts/cloud_smoke.py` (budget-gated,
   estimate-first, aborts over budget).
3. Lumerical: regenerate recorded artifacts on a licensed machine (or the
   `lumerical-nightly` self-hosted lane; `docs/self_hosted_runner.md`).
4. Update `SOLVER_STATUS.md` and the refreshed `tests/recorded/` /
   `examples/*/recorded/` artifacts **in the same PR**; scrub artifacts for
   secrets first (AGENTS.md rule #5).

## 4. Repo conventions & guardrails (do not violate)

- **Only `run()` spends** money/licenses/GPU. `validate`/`build`/`estimate`
  and constructors stay offline, pure, deterministic — `tests/conformance/`
  enforces this; don't weaken it.
- **Every change is a PR into `main`** — branch protection is ON (PR + green
  `pass` check + up-to-date branch + linear history; 0 required reviews, so a
  solo maintainer can self-merge, but the PR + CI path is mandatory).
- **Never commit secrets.** Recorded artifacts must be scrubbed.
- **Commit identity:** author as the project owner's personal identity; do
  not attribute commits to an AI co-author.
- **Validate through the artifact users run** — fresh interpreter, real files.
- **No coverage theatre** — every test asserts real behavior; ratchet
  coverage upward only.
- Docs build without the optional engines (`docs/conf.py` mocks them).

## 5. Gotchas / non-obvious

- `gds_fdtd.sparams` and `gds_fdtd.core` are gone (→ internal `_sparams`;
  geometry classes live in `gds_fdtd.geometry`). The `"sparams"` strings in
  `solvers/lumerical.py` are Lumerical sweep names / `.fsp` paths — don't
  touch them.
- `Technology.to_solver_dict()` returns the internal schema-v1 dict; loaders
  and adapters accept either a `Technology` model or that dict.
- Ports must face 0/90/180/270°. beamz v1 is single-mode TE and rejects
  y-facing ports; it auto-resolves indices from the unified tech.
- Materials carry hints for *every* engine; a missing optional engine must
  not break loading for the others.
- `filterwarnings` (pytest) turns any in-package `DeprecationWarning` into a
  test error — the only sanctioned source is the `Component.structures`
  nested-list shim, whose tests assert the warning explicitly.
- The rii database resolves via explicit `db_dir` → `GDS_FDTD_RII_DB` →
  `~/.cache/gds_fdtd/rii`; the examples ship a mini-DB under
  `examples/02_technology/rii_db` (the shared tech's substrate pins
  `source: rii`, so engine builds of that layer need the DB findable).

## 6. Where to go next

`ROADMAP.md` — remaining work (notebook-exec CI, codecov gates, link check,
merge queue), the feature menu, the carry-forward backlog with rationale, and
the owner-only actions (PyPI trusted publisher first among them).
