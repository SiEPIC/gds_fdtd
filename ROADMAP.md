# gds_fdtd roadmap

Living plan for the post-0.5.0 development arc. Any agent or contributor
should be able to read this, understand the current state, and pick up work
without losing context. Keep it current; move granular tracking to GitHub
Issues as items are picked up.

## Where we are — v0.6.3 (released 2026-08-19)

**Maintenance review (2026-10-04):** the compatible Dependabot updates are
consolidated with a fresh lock and security fixes for Tornado (6.5.10),
urllib3 (2.8.0), and PyJWT (2.15.1).
The initial batch landed in #155; follow-up proposals #156–#158 bring
setup-uv to 10.2.0, CodeQL SARIF to 4.38.2, and hypothesis to 6.168.3.
The temporary BeamZ <0.5 hold from maintenance is superseded by the validated
0.5.3 migration below; Dependabot defers >=0.6 pending API validation.
Pydantic's >=2.13.5 floor remains deferred: gdsfactory 9.45.0 is the last supported version on Python 3.11,
and its kfactory dependency requires Pydantic <2.13. The fuzz extra is
restricted to Linux x86_64/Python >=3.12 to match available Atheris wheels without
reducing the package's supported Python/platform range. See the Unreleased
changelog for the complete maintenance scope.

**Unreleased BeamZ migration (2026-10-04; PR #154 remains draft):** the adapter now supports
`beamz>=0.5.3,<0.6` (locked to 0.5.3), with immutable ports/sources/monitors
and detached modal/field results. Canonical polygons replace the removed
0.4 geometry helper; preparation stays on the CPU. The 0.5.0/0.5.1/0.5.2
release matrix runs locally on the RTX 3090; reproducible scripts, full
S-matrix magnitudes, convergence diagnostics, and plots live in
[`benchmarks/`](benchmarks/README.md). The historical 0.4.3 recordings remain
unchanged. TE fundamental mode and x-facing ports remain the validated
adapter scope; y-facing ports, TM, and multimode need separate validation.

**Reviewer-requested convergence study completed (2026-10-04):** the fixed-setup
BeamZ 0.5.3 S-bend sweep covers meshes 10/14/20/25/30 on RTX 3090. Both final
through-path changes pass the predeclared 0.05 dB tolerance: 0.02655 dB
(20→25) and 0.03720 dB (25→30), across both directions and all three wavelength
samples. The mesh-30 uniform-lead probe spread is 0.01371 dB at 1.55 µm and
0.00904–0.01752 dB across its three wavelengths. Reflections do not meet the
same convergence tolerance (all-entry final change 0.50777 dB); complex phase,
PML/domain convergence, and other device families remain outside this claim.
[Report, plots, and reproduction](benchmarks/BEAMZ_CONVERGENCE.md).
[Example 06](examples/06_convergence_and_caching/06_convergence_and_caching.ipynb)
now replays the 0.5.3 JSON records with editable convergence and monitor plots,
checks the reported criteria, and labels the older 0.4.3 figures as historical.
The notebook also compares the mesh-30 0.5.3 field intensity with the recorded
commercial fields in linear/log views, using a portable cropped JSON map
with source provenance; no new complex NPZ archives are committed.
PR #154 remains draft for maintainer review; generated NPZ archives stay local.

**BeamZ 0.5.3 fix verified (2026-10-04):** the released material-snapshot
correction works through the integration without adapter changes. Exact probes
reduce uniform-lead monitor spread from 0.315 to 0.097 dB at mesh 10 and from
0.130 to 0.027 dB at mesh 20. The specific upstream defect is addressed;
residual numerical sensitivity and absolute mesh convergence remain separate
validation concerns. Full mesh-10 y-branch/escalator matrices and mesh-10/20
S-bend runs also pass finite, incident-power, and temporal-convergence checks.
After merging the upstream maintenance/security updates and requiring
BeamZ >=0.5.3, the local suite passes (372 passed, 27 skipped); repository-wide
lint/formatting, spelling, strict source typing, and lock checks pass. The migration is suitable for merge after normal checks/review within its
fundamental-TE/x-facing scope; extra convergence studies are follow-up accuracy
work, not an unresolved upstream-fix blocker. PR #154 remains draft for
maintainer review. [Follow-up report](benchmarks/BEAMZ_053_RESULTS.md)
contains versioned JSON/plot artifacts; original 0.5.2 measurements are preserved.
Generated `.npz` archives are excluded from Git and can be recreated by the
benchmark scripts; the duplicated upstream issue body is linked on GitHub.

**Device validation completed (2026-10-04):** fresh RTX 3090 / BeamZ 0.5.2
runs cover the sharp S-bend (meshes 6/10/14/20), full three-port y-branch,
and Si→SiN escalator (meshes 6/10). Forward y-branch paths agree within
0.05 dB and escalator transmission within 0.09 dB of recorded commercial
results; weak matrix entries still differ. The y-branch reverse paths work.
On 0.5.2, S-bend monitor-plane sensitivity remained unresolved: a mesh-20
probe varies by 0.130 dB along the straight output lead. Filed
[BeamZ #309](https://github.com/beamzorg/beamz/issues/309) with a verified
standalone reproducer. [Results and reproduction](benchmarks/DEVICE_RESULTS.md)
include full matrices, fields, convergence diagnostics, and limitations.
No fresh cloud or licensed runs were performed.


`v0.6.3` is a maintenance release over `v0.6.2`: dependency floors and pinned
GitHub Actions moved to current releases (including `setup-uv` v10, whose new
cache-poisoning default is a no-op here). No API change. The `pip-audit`
exception from 0.6.2 still stands — tidy3d remains at 2.12.0, so #115's exit
criteria are unmet.

`v0.6.2` is the maintenance release beneath it, over `v0.6.1`: dependency floors raised to
current releases, a protective `beamz < 0.5` cap ahead of that project's
breaking 0.5 API (#85), and a documented `pip-audit` exception for three
`cryptography` advisories that cannot be remediated while tidy3d 2.12 pins
`cryptography==48.0.1` (#115). No API change.

`v0.6.1` (2026-07-21) is the feature release beneath it, tagged with a signed
GitHub release (Sigstore bundles + SBOM) and building on `v0.6.0`
(2026-07-15). It adds the interactive 3D viewer
(`viewer3d.show_3d` / `save_3d` / `render_static`), steerable field monitors
(`field_monitor_positions` / `field_monitor_wavelengths`, `plot_monitor_planes`),
and examples `05b_field_monitors` + `11_bragg_grating`; it hardens the viewer
against HTML/JS injection from GDS-derived names and makes its embed render
across JupyterLab / VSCode / static docs. The PyPI upload is the one step
still blocked — see the owner actions below.

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
      publish step cannot run (re-verified 2026-07-21: `invalid-publisher`,
      PyPI still serves 0.4.0). Once registered, re-run the failed publish job
      of the latest (v0.6.3) Release run.
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
