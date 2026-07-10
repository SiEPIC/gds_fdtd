# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Continuous fuzzing via ClusterFuzzLite + atheris (`fuzz/`), with fuzz
  targets on the technology-YAML and INTERCONNECT `.dat` parsers; runs on
  PRs that touch the library.
- Release artifacts are now Sigstore-signed (keyless, GitHub OIDC); the
  `.sigstore` bundles are attached to each GitHub Release.

### Fixed
- `SMatrix.from_entries` (and therefore `SMatrix.from_dat`) rejected a
  corrupt mode/port index only by attempting an unbounded allocation — a
  malformed `.dat` claiming a ~10**6 mode id tried to allocate 623 TiB and
  raised `MemoryError`. It now raises a clean `ValueError` (found by the new
  `.dat` fuzz target).

### Security
- Branch protection enabled on `main`; solo-friendly (PR + green CI +
  linear history required, force-pushes blocked).

## [0.5.0] - 2026-07-08

The solver-agnostic release: one component, one technology file, any engine —
validated live on all three. 73 commits (PR [#23](https://github.com/SiEPIC/gds_fdtd/pull/23)).

### Added

#### Solver-agnostic core
- `Solver` ABC with a strict lifecycle contract: `validate()` / `build()` /
  `estimate()` are offline and free; **only `run()` spends** money, licenses,
  or GPU time. Engine registry + `gds_fdtd.solvers` entry-point group so
  third-party engines plug in (`docs/adding_a_solver.md` is the guide).
- Adapters: **tidy3d** (≥ 2.11, ModalComponentModeler), **Lumerical FDTD**
  (2024/2025, offline `.lsf` generation + licensed run), **beamz** (≥ 0.4,
  free/Apache-2.0, JAX CPU/GPU) — identical setup on all three:
  `get_solver(name)(component, tech, spec)`.
- Canonical `SMatrix` (complex, multi-port, multi-mode, NaN partials) with
  reciprocity/passivity/power-balance checks and I/O: INTERCONNECT `.dat`,
  Touchstone `.sNp`, HDF5, npz. Validated `SimulationSpec` (pydantic)
  replaces ~15 loose solver kwargs.
- **Technology schema v2**: named materials defined once, referenced by
  layers; each material carries a neutral `nk`, optional
  refractiveindex.info `rii:` source, and per-engine hints — one YAML for
  every solver. `gds-fdtd convert-tech` migrates v1 files (round-trip
  guarded). Offline rii database reader.
- Tier-B pipeline for kernel-level engines: `grid.rasterize` (sub-pixel
  averaged permittivity, sidewall angles), `modes` (local tidy3d mode
  solver: Si-strip n_eff(TE0) = 2.451), `extraction` (bidirectional mode
  overlap, exact identities tested).
- Convergence sweeps (`convergence.sweep` with recommended-value picking),
  job-hash **result caching** (`run_cached`: repeat runs are free), and
  cross-solver validation (`validation.validate_across`).
- Serializable `JobSpec` + `LocalBackend`/`SubprocessBackend` + `gds-fdtd`
  CLI (`validate|build|estimate|run|convert|convert-tech|solvers`; exit
  codes 0/2/3/4; secrets from environment only) — remote-compute ready
  (Modal/AWS/SLURM recipes in `docs/remote_compute.md`, beamz-first).
- Standardized example flow (geometry+ports+FDTD-region plot → offline
  setup → S-parameters → field profile), RdBu as the package-wide
  visualization scheme, port-extension stubs drawn in every geometry plot.
- gdsfactory (≥ 9) converter with port auto-detection; SiEPIC and PreFab
  integrations updated; 16 runnable examples incl. three free beamz ones.
- Exception hierarchy (`GdsFdtdError` tree, backward-compatible),
  `GDS_FDTD_*` settings (pydantic-settings), JSON-lines logging option.

#### Validation (all on real engines, recorded into the repo)
- **Three-engine agreement on the identical job**: tidy3d ↔ Lumerical
  within **0.0033 dB**, beamz within 0.052 dB (recorded in
  `tests/recorded/`, asserted every PR). Per-solver status in
  `SOLVER_STATUS.md`.
- Honest branch coverage **80%** (was a gamed ~100% badge over a 16%
  reality); coverage gate ratchets up only. Recording mocks run both
  legacy adapters' full setup paths offline; recorded real artifacts
  replay the results pipelines.

#### Infrastructure
- CI: 9-leg OS/Python matrix + all-extras leg + required typecheck
  (mypy --strict on the typed core) + single `pass` gate; weekly
  lowest-floors job; pip-audit/dependency-review/zizmor/Scorecard;
  SHA-pinned actions with hardened permissions; SBOM attached to
  releases; tag-driven trusted publishing (hatch-vcs — no version bumps).
- Budget-gated tidy3d cloud-smoke workflow (human-approved) and a
  self-hosted Lumerical nightly lane (off until a runner exists).

### Changed
- Examples rewritten on the modern API and ONE technology file; legacy
  15-kwarg constructor examples removed (the classes remain, deprecated,
  until v1.0).
- README rebuilt around real solver output (field profiles, curated
  S-parameters, three-engine agreement plot).

### Fixed
- 19 audited legacy bugs (B-series) plus findings F1–F14 discovered by
  live validation — highlights: `tidy3d.web` lazy-import crash in
  examples (F10), beamz reference-monitor mis-normalization (+2 dB, F9),
  beamz y-normal monitor mis-normalization (F14 — y-facing ports now
  rejected loudly), eager tidy3d import breaking engine-free loading
  (F13), test fixture/GDS layer mismatch that silently mis-assigned a
  port to the superstrate (F4).
- 646 lines of dead code removed (write-only field-monitor plumbing,
  uncalled log-metrics chains, legacy `sparam`/`s_parameters`).
- Example `09_smatrix` no longer requires the optional h5py on a clean
  install; library modules log instead of printing; shipped docstrings
  scrubbed of internal project references (pre-PyPI polish audit).

### Removed
- Fake-coverage test that exec-compiled `pass` statements to inflate
  codecov; jekyll/python-publish workflows; per-solver example
  technology files.

### Note on 0.4.0
The 0.4.0 entry below described several features that did not exist at the
time (Touchstone/JSON export, energy-conservation checks, group-delay
tools). Those capabilities ship — for real, with tests — in this release.

## [0.4.0] - 2025-08-05

### Major Release - New Architecture

This release completely rewrites how the solvers work. Everything is now more modular and easier to extend.

### Added

#### New Solver System
- Base `fdtd_solver` class that both Tidy3D and Lumerical solvers inherit from
- `fdtd_solver_tidy3d` class that uses Tidy3D's ComponentModeler for S-parameters
- `fdtd_solver_lumerical` class that works with Lumerical FDTD (with GPU support)
- `fdtd_port` class so ports work the same way in both solvers
- Better field monitor system that lets each solver handle visualization differently

#### Multi-Modal Simulations
- Both TE and TM polarizations in the same simulation
- Tidy3D ComponentModeler integration for better S-parameter accuracy
- S-parameter extraction for all mode combinations
- PDL and mode conversion analysis tools

#### Logging System (`logging_config.py`)
- Logs everything that happens during simulations
- Writes detailed logs to files in the working directory
- Shows different amounts of detail in console vs. file logs
- Creates log files with timestamps and component names automatically

#### Better S-Parameter Handling (`sparams.py`)
- Tools to validate S-parameter results
- Export to multiple formats (.dat, JSON, Touchstone)
- Check energy conservation and reciprocity
- Calculate group delay and bandwidth

#### Documentation
- Complete documentation website using Sphinx
- Guides for how the solvers work and how to use them
- Setting up multi-modal simulations
- Working with S-parameters
- Creating technology files
- Troubleshooting common problems
- Automatic API documentation
- Deployed to GitHub Pages

### Changed

#### Breaking Changes
- Removed old `lum_tools.py` and `t3d_tools.py` modules
- All solvers now inherit from the base `fdtd_solver` class
- Import solvers from `gds_fdtd.solver_tidy3d` and `gds_fdtd.solver_lumerical` now
- Changed some method names to be consistent between solvers

#### How Things Work Now
- Both solvers use the same parameter names and structure
- Simulation domain size is calculated automatically from your component
- Each component gets its own subdirectory for output files
- Better error messages when parameters don't make sense

#### Tidy3D Solver Changes
- Now uses ComponentModeler instead of manual S-parameter calculation
- Better handling of cloud simulation submission and monitoring
- Field visualization uses Tidy3D's native plotting functions
- Supports all TE/TM mode combinations in one simulation

#### Lumerical Solver Changes
- Works with Lumerical 2024 GPU acceleration syntax
- Better integration with technology files for layer building
- S-parameter sweeps are set up automatically
- Shows estimated memory and computation requirements

### Fixed

- Port detection from GDS files works better now
- Material assignment from technology files is more reliable
- Fixed buffer calculations and port extensions
- S-parameter magnitudes and phases are calculated correctly
- Field monitors are placed exactly where they should be
- File organization works the same on different operating systems

### Migration Guide from v0.3.x to v0.4.0

#### Import Changes
```python
# Old way (v0.3.x)
from gds_fdtd.lum_tools import lumerical_fdtd
from gds_fdtd.t3d_tools import sim_tidy3d

# New way (v0.4.0)
from gds_fdtd.solver_lumerical import fdtd_solver_lumerical
from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d
```

#### Solver Initialization
```python
# Old way (v0.3.x)
sim = sim_tidy3d(in_port=component.ports[0], device=component)

# New way (v0.4.0)
solver = fdtd_solver_tidy3d(
    component=component,
    tech=technology,
    port_input=[component.ports[0]],
    # ... other parameters
)
```

#### Running Simulations
```python
# Old way (v0.3.x)
sim.upload()
sim.execute()

# New way (v0.4.0)
solver.run()  # Everything happens automatically
```

#### Accessing Results
```python
# Old way (v0.3.x)
# Results were accessed differently for each solver

# New way (v0.4.0)
# Same interface for both solvers
sparams = solver.sparameters
wavl = sparams.wavelength
s41 = sparams.S(in_port=1, out_port=4, in_modeid=1, out_modeid=1)
```

---

## [0.3.10] - 2024-05-13

### Added
- Buffer parameter for Lumerical solver

---

## [0.3.9] - 2024-05-11

### Changed
- Python requirements changed from 3.11 to 3.10

---

## [0.3.8] - 2024-05-11

### Fixed
- Path for halfring function

---

## [0.3.7] - 2024-05-10

### Changed
- Updated examples
- Updated unit tests
- Updated Update_Halfring_CML for ebeam_dc_halfring_straight

---

## [0.3.6] - 2024-05-08

### Changed
- Updated compact model library for ebeam_dc_halfright_straight PCell

---

## [0.2.0] - 2024-03-07

### Added
- from_gdsfactory: create simulation recipes with gdsfactory component instance
- Multi-polarization support and examples

### Fixed
- Plotting dimensions
- Source and monitor placement
- Setup file
- S-parameters prep
- Other minor improvements

---

## [0.1.0] - 2023-12-03

### Added
- Base usable version of the package
