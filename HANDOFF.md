# gds_fdtd — Development Handoff (0.6.0)

**Audience:** the next engineer/agent, on a machine that **has** what this one did
not — tidy3d cloud credits (`TIDY3D_API_KEY`), a Lumerical license (`lumapi` on
path), and ideally a CUDA GPU. Your job is to run the **live validity checks** that
could only be recorded, not re-run, here, and to verify the examples end-to-end.

**State at handoff:** `main` carries the complete, breaking **0.6.0 legacy
cleanup** (PRs #27–#32). Everything below was verified **offline** on macOS with
the `dev`, `gdsfactory`, and `beamz` extras. Nothing that spends cloud credits,
a license, or a GPU was executed. **The 0.6 changes were pure refactors
(rename / move / remove) — no physics or numeric behavior changed** — but the
engine-facing paths deserve a live re-check before tagging `v0.6.0`.

---

## 1. What gds_fdtd is (one screen)

Turns a photonic-chip **GDS layout + a technology YAML + a `SimulationSpec`** into
a 3D FDTD run on the engine of your choice, and returns one canonical S-matrix:

```python
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

solver  = get_solver("tidy3d" | "lumerical" | "beamz")(component, tech, SimulationSpec())
smatrix = solver.run()          # the ONLY call that spends money / license / GPU
```

EDA-agnostic on the front (KLayout/SiEPIC, gdsfactory ≥9), solver-agnostic on the
back. Same `(component, technology, spec)` in, same `SMatrix` out, on all engines.

## 2. The development arc (0.1 → 0.6)

| Version | What it was |
|---|---|
| **0.1–0.3** (2023–24) | Rough SiEPIC research scripts: `lum_tools.py` / `t3d_tools.py`, per-engine one-offs, PCell/CML helpers. |
| **0.4.0** (2025-08) | First "architecture": a `fdtd_solver` base + `fdtd_solver_tidy3d` / `fdtd_solver_lumerical` + `fdtd_port`, `logging_config`, `sparams`. (Its changelog over-promised — Touchstone/JSON/energy checks were only *real* in 0.5.) |
| **0.5.0** (2026-07-08, PR #23, ~73 commits) | **The solver-agnostic rewrite** (modernization Phases 0–7). `Solver` ABC contract, engine registry + entry points, canonical `SMatrix`, validated `SimulationSpec`, `Technology` schema v1/v2 + refractiveindex.info materials, Tier-B kernel pipeline (`grid`/`modes`/`extraction`), convergence/caching/cross-solver validation, serializable `JobSpec` + `gds-fdtd` CLI + backends, error hierarchy, settings, JSON logging. 19 audited legacy bugs (B-series) + findings F1–F14 fixed by live validation. Honest coverage (was a gamed ~100% over a 16% reality). Three-engine agreement recorded. |
| post-0.5.0 | De-slop (PR #24); retired `MODERNIZATION_PLAN.md` → `ROADMAP.md`; Sigstore-signed releases + atheris fuzzing (WS5); `mypy --strict` expanded; **#27** flattened the tidy3d engine names (`fdtd_solver`/`fdtd_port` → `_TidyEngineBase`/`_TidyPort`). |
| **0.6.0** (this arc, PRs #28–#32; **on `main`, not yet tagged**) | **Staged legacy cleanup — removed the entire pre-0.5 public surface.** See below. |

### The 0.6.0 legacy cleanup (the 5 stages)
1. **#28** — deleted dead code: `core.technology` class + the tidy3d engine's unused results path (dropped its last dependency on the legacy `sparams`).
2. **#29** — removed `core.parse_yaml_tech`; every call site now uses `Technology.from_yaml(...)`.
3. **#30** — renamed `Technology.to_legacy_dict()` → `to_solver_dict()` (and `MaterialSpec.to_legacy()` → `to_solver_dict()`).
4. **#31** — removed the `gds_fdtd.core` PEP-562 shim module (the deprecated lowercase geometry names `port/structure/region/component/layout`).
5. **#32** — internalized the public `gds_fdtd.sparams` module → private `gds_fdtd._sparams`.

Earlier in the 0.6 line the deprecated `fdtd_solver_lumerical` / `fdtd_solver_tidy3d` classes and the `gds_fdtd.solver` module were also removed. **Net: the supported public API is now `get_solver(name)` + `Technology` + `SimulationSpec` + `SMatrix` (+ `geometry`, `lyprocessor`, `layout.gdsfactory`, `convergence`, `caching`, `validation`).** Full migration table in `CHANGELOG.md` under *[Unreleased] — targeting 0.6.0*.

## 3. Architecture map (where things live)

- **Layout ingestion:** `lyprocessor.py` (KLayout/SiEPIC pin+DevRec), `layout/gdsfactory.py` (gf ≥9). Ports auto-detected, never hand-placed. Primitives in `geometry.py` (`Port`/`Structure`/`Region`/`Component`, flat role-tagged structures).
- **Technology:** `technology.py` (pydantic; YAML schema v1 inline + v2 named-materials; `to_solver_dict()` is the internal dict the adapters consume), `materials/rii.py` (offline refractiveindex.info reader).
- **Contract:** `solvers/base.py` — `Solver` ABC (`validate`/`build`/`estimate` offline & free; **only `run()` spends**), `SolverCapabilities`, registry + `gds_fdtd.solvers` entry-point discovery.
- **Settings & results:** `spec.py` (`SimulationSpec`), `smatrix.py` (`SMatrix`; `.dat`/Touchstone/HDF5/npz I/O delegates to the internal `_sparams.py`).
- **Adapters:** `solvers/tidy3d.py` (+ internal `_tidy3d_base.py`, `_tidy3d_engine.py`), `solvers/lumerical.py` (offline `.lsf` generation + licensed run), `solvers/beamz.py` (JAX FDTD, free).
- **Tier-B (kernel engines):** `grid.py` (permittivity rasterizer), `modes.py` (mode solver protocol + tidy3d local backend), `extraction.py` (mode-overlap S-params).
- **Orchestration:** `convergence.py`, `caching.py` (`run_cached` job-hash cache), `validation.py` (`validate_across`), `execution/{jobspec,backends}.py`, `cli.py` (`gds-fdtd validate|build|estimate|run|convert|convert-tech|solvers`).
- **Errors/logging/config:** `errors.py` (`GdsFdtdError` tree → CLI exit codes 0/2/3/4), `logging_config.py` (JSON-lines), `settings.py` (`GDS_FDTD_*` env).

## 4. What is validated vs NOT (read this before trusting anything)

**Verified offline this session** (macOS, `dev`+`gdsfactory`+`beamz`):
- Full suite `-m "not physics and not gpu and not cloud and not licensed"`: **283 passed, coverage 82.72%** (config floor 75).
- `mypy --strict` on the 15 gated modules; `ruff` + `ruff format`; `codespell`; full `prek` hooks; `sphinx-build` docs.

**NOT run here (no credentials/hardware):**
- **tidy3d cloud** (`cloud`/`physics` tests, `scripts/cloud_smoke.py`; regenerating the recorded tidy3d artifacts behind examples 07/10). Note: the example gallery no longer *spends* credits — 05 uses tidy3d's free local mode solver, and 07/10 load recorded tidy3d output.
- **Lumerical** (`licensed` tests, the `lumerical-nightly` lane; regenerating the recorded Lumerical artifacts behind examples 07/10).
- **GPU** (`gpu` tests; fdtdz; beamz-GPU).
- **The three-engine agreement**: the artifacts in `tests/recorded/*.npz` (recorded **2026-07-07/08**, *before* the 0.6 flatten) are *replayed* offline every PR but were **not regenerated live** this arc. `SOLVER_STATUS.md` dates predate the cleanup — treat them as "last known good", not "verified post-0.6".

## 5. Your validation runbook (ordered)

Environment:
```bash
uv sync --extra dev --extra tidy3d --extra beamz --extra gdsfactory   # + prefab/siepic if testing 07/08
# Lumerical: install Lumerical, put lumapi on PYTHONPATH.  tidy3d: export TIDY3D_API_KEY / ~/.tidy3d/config
```

0. **Baseline (offline, must be green — matches CI):**
   `uv run pytest -m "not physics and not gpu and not cloud and not licensed"` → expect ~283 passed.
1. **Examples end-to-end — the golden rule.** Per `ROADMAP.md` principle #1 and finding F10: execute each committed notebook **as the exact committed artifact, in a fresh interpreter**, not via a shared session. The gallery is now numbered notebooks (paired jupytext `.py` + executed `.ipynb`): `00_quickstart`(beamz) `01_layout_to_component`(offline) `02_technology`(offline) `03_first_simulation`(beamz) `04_reading_results`(offline, recorded) `05_fields_and_modes`(tidy3d **local** mode solver, free) `06_convergence_and_caching`(beamz) `07_choosing_an_engine`(recorded 3-engine) `08_frontends`(beamz + offline) `09_cli_and_jobs`(offline) `10_cookbook`(beamz + recorded). Re-run them with `jupytext --to ipynb --execute` (or `jupyter execute` the `.ipynb`); each should reproduce its `SMatrix` and plots **for free** — none spends credits or a license. `tests/test_examples_importable.py` already asserts (every PR) that none imports a removed symbol.
2. **tidy3d cloud smoke:** `BUDGET_FC=0.5 uv run python scripts/cloud_smoke.py` (budget-gated; uploads → `estimate_cost` → deletes, aborts over budget, then runs & asserts physics). **Note:** this script was fixed this session — it previously imported the removed `parse_yaml_tech`; it now uses `Technology.from_yaml`. This is its first real run since that fix.
3. **Lumerical:** regenerate the recorded Lumerical artifacts (`tests/recorded/*.npz`, `examples/*/recorded/*.npz`, `examples/10_cookbook/recorded/si_sin_escalator*`) on a licensed machine; or wire the `lumerical-nightly` workflow to a self-hosted runner (`docs/self_hosted_runner.md`, set the `LUMERICAL_RUNNER` repo var).
4. **Three-engine agreement (regenerate live):** rebuild `tests/recorded/straight_mesh10_{tidy3d,lumerical,beamz}.npz` from the same job; confirm tidy3d↔Lumerical within ~**0.0033 dB** and beamz within ~**0.052 dB**; `test_three_engine_agreement` (in `tests/test_recorded_artifacts.py`) must still pass against the refreshed data.
5. **Refresh the record:** update `SOLVER_STATUS.md` (dates/versions/evidence) and the `tests/recorded/*` artifacts **in one PR**. **Sanitize first** — grep the artifacts for API keys, hostnames, license tokens (AGENTS.md rule #5).
6. **Run the gated markers you now can:** `-m "physics"`, `-m "cloud"`, `-m "licensed"`, `-m "gpu"` (each spends the corresponding resource — see the guardrails).

## 6. Repo conventions & guardrails (do not violate)

- **Only `run()` spends** money/licenses/GPU. `validate`/`build`/`estimate` and all constructors stay offline, pure, deterministic — the conformance suite (`tests/conformance/`) enforces this; don't weaken it.
- **Every change is a PR into `main`** — branch protection is ON (PR + green `pass` check + up-to-date branch + linear history; force-pushes blocked; admins enforced; **0 required reviews**, so a solo maintainer can self-merge, but the PR + CI path is mandatory). Strict up-to-date means you rebase/re-CI stacked PRs before each merge.
- **Never commit secrets.** Recorded artifacts under `tests/recorded/` must be scrubbed.
- **Commit identity:** author as the project owner's personal identity; do not attribute commits to an AI co-author.
- **Validate through the artifact users run** (finding F10) — fresh interpreter, real files.
- Docs build without the optional engines (`docs/conf.py` mocks tidy3d/lumapi/beamz/gdsfactory/prefab/SiEPIC/pya/jax).

## 7. Where we stand (metrics)

- **Version:** 0.6.0 line, **not yet tagged** (installed reports `0.5.1.dev…`; hatch-vcs derives the version from the git tag — tag `v0.6.0` to cut the release).
- **Coverage:** 82.72% (all-extras leg; base floor enforced at 75). WS2 target is >90% on all-extras.
- **CI:** full OS×Python matrix + all-extras leg + required `mypy --strict` core + `pass` gate + docs + atheris fuzz + Sigstore-signed release pipeline. All green on `main`.
- **OpenSSF Scorecard ~7.0**, branch protection now enabled; the next lever is the **PyPI trusted publisher** (owner action → unlocks the Packaging check).

## 8. Where to go next

- **Tag `v0.6.0`** once the live validation (§5) passes and `SOLVER_STATUS.md` is refreshed.
- **v1.0 lane:** freeze the public API + `__all__`; retire the last deprecation shim (`Component.structures` nested-list warning in `geometry.py`); write down the deprecation policy.
- **WS1:** turn the 18 examples into executed notebooks (jupytext) with committed outputs + a docs gallery.
- **Docs API reference:** `docs/modules.rst` autosummary is deliberately minimal. Expanding it to the full public surface (`smatrix`/`solvers`/`technology`/`spec`/`geometry`/…) first needs a module-docstring RST-hygiene pass — several modules (`convergence`, `extraction`, `validation`, `solvers/base`, `solvers/beamz`, `solvers/tidy3d`) use bare indented code examples (need `::` literal blocks) and `|S|²`/`|E|` (read as RST substitutions), which emit docutils warnings under autodoc. Do that pass, then expand + enable `sphinx-build -W`.
- **WS2:** push all-extras coverage to >90% — the low spots are the engine adapters (`solvers/beamz`, `solvers/tidy3d`, `solvers/lumerical`) and `cli`/`caching`/`modes`; add real (small) beamz end-to-end tests and property-based tests on `SMatrix`/`geometry`/`technology`.
- **WS4/WS5:** codecov project/patch PR gates; PyPI trusted publisher (owner); merge queue; CII-Best-Practices badge.
- **Feature menu:** fdtdz adapter (free GPU; the rasterize→modes→extraction pipeline is built and tested, blocked only on GPU hardware), parameter sweeps/optimization, `scikit-rf` interop, a small PDK of reference devices with known-good S-params.

Full detail + backlog rationale in **`ROADMAP.md`**; per-engine last-verified status in **`SOLVER_STATUS.md`**; AI-contributor rules in **`AGENTS.md`**.

## 9. Gotchas / non-obvious

- `gds_fdtd.sparams` and `gds_fdtd.core` are **gone** (→ internal `_sparams`, and geometry classes live in `gds_fdtd.geometry`). `import gds_fdtd.sparams`/`.core` raises `ModuleNotFoundError`. Use the `SMatrix` API for `.dat`/Touchstone/HDF5/npz.
- The `"sparams"` strings in `solvers/lumerical.py` are the **Lumerical INTERCONNECT sweep name / `.fsp` path**, not the Python module — don't touch them.
- `Technology.to_solver_dict()` returns the internal schema-v1 dict; the loader and adapters accept **either** a `Technology` model or that dict.
- Ports must face 0/90/180/270°. beamz v1 is single-mode TE and rejects y-facing ports (finding F14); it auto-resolves indices from the unified tech (finding F9).
- `filterwarnings = ["error::DeprecationWarning:gds_fdtd.*"]` in `pyproject.toml` turns any in-package `DeprecationWarning` into a test error — the only remaining source is the `Component` nested-list shim.
- Materials in a unified tech carry hints for *every* engine; a missing optional engine (e.g. tidy3d not installed) must not break loading for the others (finding F13) — keep it that way.
