# gds_fdtd roadmap

Living plan for the post-0.5.0 development arc. Any agent or contributor
should be able to read this, understand the current state, and pick up work
without losing context. Keep it current; move granular tracking to GitHub
Issues as items are picked up.

## Where we are — v0.6.0 (released 2026-07-15)

`v0.6.0` is tagged with a signed GitHub release (Sigstore bundles + SBOM).
The PyPI upload is the one step still blocked — see the owner actions below.

Solver-agnostic FDTD: one `Component` + one technology file + one
`SimulationSpec`, any engine (tidy3d / Lumerical / beamz) behind
`get_solver(name)(component, tech, spec)`. Three-engine agreement within
0.052 dB on identical jobs (tidy3d↔Lumerical within 0.0033 dB), recorded and
asserted every PR. All-extras branch coverage 90.2% (gate: 90; base floor 75).
`mypy --strict` passes on the whole package and is a required CI gate.
Docs on GitHub Pages (actions-based flow). OpenSSF Scorecard: branch
protection, signed releases, and CI fuzzing in place.

**0.6.0 highlights (all merged to `main`):**

- **Breaking legacy cleanup** (PRs #27–#32): the entire pre-0.5 public surface
  removed — `solver`/`solver_tidy3d`/`solver_lumerical`, `core`,
  `to_legacy_dict` (→ `to_solver_dict`), public `sparams` (→ internal
  `_sparams`). The supported API is `get_solver` + `Technology` +
  `SimulationSpec` + `SMatrix`, all exported at the top level.
- **Examples became an executed-notebook curriculum** (PR #34): 13 committed
  notebooks (`00_quickstart` … `10b_polarization`) with real outputs, recorded
  cross-engine artifacts (y-branch, sharp S-bend, crossing, escalator, PBS,
  PSR), the frontend × engine matrix, and the material-source selection system
  (`eda → rii → nk`).
- **Docs overhaul**: full API reference (27 module pages), a frontends guide
  (including "write your own"), the technology/materials page rebuilt around
  refractiveindex.info, real figures throughout, one unified project
  description.
- **Hardening**: user-input raises routed through the `GdsFdtdError`
  hierarchy; whole-package strict typing (caught three real bugs); property-
  based tests (hypothesis) + a real beamz end-to-end test; deprecation policy
  written (CONTRIBUTING.md).

See [`HANDOFF.md`](HANDOFF.md) for the development arc and the live-validation
runbook, and `CHANGELOG.md` for the full 0.6.0 entry.

## Guiding principles (non-negotiable)

1. **Validate through the exact artifact users run** — example files in a
   fresh interpreter/venv, not bespoke scripts sharing session state.
2. **No coverage theatre.** Every test asserts real behavior. A number that
   went up because a `pass` got executed is a regression, not progress.
3. **Only `run()` spends** money / licenses / GPU. Constructors and
   `validate`/`build`/`estimate` stay offline, pure, deterministic.
4. **Deferrals are documented, not silent.** If something can't be validated
   here (no GPU, no container runtime), it goes to the backlog with the
   verified facts a future executor needs.
5. **Every change is a PR into `main`.** No direct pushes.

## Completed workstreams (the 0.5.1→0.6.0 polish arc)

| Workstream | Outcome |
|---|---|
| **WS1 — Executed example notebooks** | DONE. 13 jupytext-paired notebooks, committed executed with real outputs; gallery renders in the docs via myst-nb; every simulation example shows its mode and its field; recorded artifacts carry PROVENANCE notes. |
| **WS2 — Real >90% coverage** | DONE. All-extras branch coverage 90.2%, `--cov-fail-under=90` in CI; hypothesis property tests for the numeric core; a real (slow-marked) beamz end-to-end test runs in the all-extras leg. |
| **WS3 — Robustness** | DONE. `GdsFdtdError` hierarchy on every user-input path (dual-inheriting the builtins); `mypy --strict` on all 34 source files as a required gate; top-level API exports; library-quiet logging; written deprecation policy. |
| **WS4/WS5 — Tooling & Scorecard (partial)** | Signed releases (Sigstore) and CI fuzzing (atheris) landed. Remaining items below. |

## Remaining work (post-0.6.0)

- **Notebook-execution CI job** (WS1 leftover): re-execute the offline
  notebooks (beamz + local + recorded, all free) on PRs and diff outputs.
  Today CI guards them via `tests/test_examples_importable.py` only.
- **Codecov project/patch gates** so PRs that drop coverage fail visibly.
- **Dependency freshness canary**: an allowed-failure "latest unpinned" job
  alongside the weekly lowest-floors job.
- **Docs link-check** job; **merge queue** (serializes the up-to-date-branch
  dance the dependabot trains currently do manually).
- **Scorecard**: Packaging resolves once the PyPI trusted publisher is
  registered; CII Best-Practices badge is an owner registration;
  Code-Review score is structurally capped for a solo maintainer.

## Feature ideas (menu — pick as inspiration strikes)

Not committed; a palette to choose from. Roughly ordered by impact.

- **fdtdz adapter** — finishes the free-GPU story; the whole
  rasterize→modes→extraction pipeline is already built and tested (blocked
  only on GPU hardware; see backlog).
- **Sweeps & optimization** — parameter sweeps over geometry/spec producing
  a tidy results table (pandas/xarray), and a hook for inverse-design loops.
- **S-parameter post-processing** — group delay, dispersion, insertion
  loss / crosstalk summaries, `scikit-rf` `Network` interop both directions.
- **Component library / PDK bridge** — a small set of parametric reference
  devices (crossing, DC, MMI, ring) with known-good S-params as fixtures.
- **Results caching backend** — content-addressed store beyond the local
  npz cache (e.g. an S3/GCS-backed cache for cluster sweeps).
- **Interactive report** — an HTML/notebook report per run (geometry,
  convergence, S-params, fields) as a single shareable artifact.
- **Richer technology** — anisotropic material helpers, gdsfactory
  `LayerStack` / KLayout `.lyt` import into technology v2.
- **Multi-frequency / broadband mode tracking**, bend-mode solving, PML
  convergence diagnostics.
- **beamz TM / multimode** — 10b showed beamz is TE-only; upstream beamz
  work plus adapter support would complete the free-engine polarization story.

## Carry-forward backlog (deferred with rationale)

- **fdtdz adapter (D9)** — needs an NVIDIA GPU + CUDA (fdtdz ships a
  CUDA-building sdist; won't even import without it). Verified kernel API and
  constraints recorded in git history. Natural to pair with a GPU CI lane.
- **femwell mode-solver backend (D8)** — second `ModeSolver` backend; the
  tidy3d local plugin already covers Tier-B needs at zero extra deps.
- **MEEP adapter** — deprioritized by owner; beamz fills the free-engine role.
- **Container images (ghcr) + conda-forge feedstock** — no container runtime
  on the dev machine to validate images; feedstock is owner-level.
- **tenacity retries on cloud calls** — modifying validated money-spending
  paths deserves its own live-revalidation session; make task submission
  idempotent (`task_name = gdsfdtd-{job_hash[:12]}`) first.
- **CITATION.cff** + Zenodo DOI for academic citation.
- **v1.0** — freeze the public API per the deprecation policy; the remaining
  `Component.structures` nested-list shim is the last deprecation to retire.

## Owner-only actions (need admin / external accounts)

- [x] **Branch protection on `main`** — PR + green `pass` + up-to-date +
      linear history required, force-pushes blocked, admins enforced.
- [x] **Pages source = GitHub Actions** — the artifact-based docs deploy is live.
- [ ] **PyPI trusted publisher** (project `gds_fdtd`, owner `SiEPIC`,
      workflow `release.yml`, env `pypi`) — in progress with Lukas; until it
      lands, tagged releases produce signed GitHub artifacts but the PyPI
      publish step cannot run (re-verified 2026-07-17: `invalid-publisher`,
      PyPI still serves 0.4.0). Once registered, re-run the failed publish job
      of the v0.6.0 Release run.
- [ ] **OpenSSF Best Practices badge** — register at bestpractices.dev.
- [ ] **`cloud-tests` environment** with a required reviewer (guards the
      budget-gated tidy3d smoke).
- [ ] **`LUMERICAL_RUNNER` repo variable** when a lab self-hosted runner
      exists (enables the nightly licensed lane).
- [ ] A **second reviewer/maintainer** would meaningfully raise the
      Code-Review score.

## Tracking

- This file = the durable plan.
- Granular work = GitHub Issues, one issue per remaining item above, linked
  from the PR that closes it.
- Every PR targets `main` and passes the full matrix before merge.
