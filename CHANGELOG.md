# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Field monitors are steerable and visible: `SimulationSpec.field_monitor_positions`
  pins any monitor plane at an absolute coordinate along its normal axis, and
  `SimulationSpec.field_monitor_wavelengths` restricts what the monitors record
  (tidy3d), decoupling field-download size from S-parameter spectral density.
  `plot_monitor_planes(solver)` draws the domain, the layer stack, and every
  monitor plane (labelled default/custom) offline before anything runs;
  `plot_field` gained an `outline=` geometry overlay via the new
  `component_outlines()`, and tidy3d's `plot_fields` selects the excitation
  (`task=`) and the recorded wavelength (`wavelength_um=`).
- Two examples: `05b_field_monitors` (the placement machinery on the Si→SiN
  escalator, whose side view shows the light climbing from the Si core into
  the SiN core) and `11_bragg_grating` (a 95 µm SiEPIC Bragg grating from
  `devices.gds` on tidy3d: the 101-point stopband spectrum, and one field
  monitor showing reflection in-band vs transmission out-of-band from a
  single run). Both ship recorded artifacts with provenance.

### Fixed
- The `filterwarnings = ["error::DeprecationWarning:gds_fdtd.*"]` guard sat
  under `[tool.coverage.report]`, where pytest never read it — moved to
  `[tool.pytest.ini_options]`, so in-package deprecation warnings fail tests
  again as intended.
- `from_gdsfactory` now merges abutting polygons per device layer before
  extrusion (same KLayout merge pipeline as the GDS loader), so hierarchical
  components no longer open wedge-shaped gaps at internal junctions under
  angled sidewalls (#1, #58).

### Changed
- Repo cleanup: the modernization arc's temporary ruff ignore list
  (`E501`/`B006`/`E722`/`B904`/`F841`/`UP031`) was worked off and removed —
  the remaining violations were fixed (mutable default arguments in
  `lyprocessor`, a swallowed exception chain in `simprocessor`, a dead
  variable, 36 long lines wrapped) and the lint config now carries no ignores.
  Unused dev dependencies (`twine`, `watchdog`, `wheel`) were dropped, stale
  docs figures deleted, and `HANDOFF.md`/`ROADMAP.md`/`SOLVER_STATUS.md`
  refreshed to the post-release state.
- The reference `examples/tech.yaml` now demonstrates material-source pinning
  live: the substrate material (`SiO2_rii`) carries eda + rii + nk sources and
  sets `source: rii`, so every engine models the buried oxide from the same
  refractiveindex.info page. `02b_rii_to_engines` walks through the selection
  with `select_source`, and the technology docs mirror the shipped file.

## [0.6.0] - 2026-07-15 (breaking)

### Removed (BREAKING)
- The pre-0.5 solver classes and their modules were removed:
  `gds_fdtd.solver_lumerical.fdtd_solver_lumerical`,
  `gds_fdtd.solver_tidy3d.fdtd_solver_tidy3d`, and the `gds_fdtd.solver`
  module (`fdtd_solver`/`fdtd_port`). They duplicated the modern adapters
  and nothing internal used them. **Migration:**

  ```python
  # before (removed)
  from gds_fdtd.solver_lumerical import fdtd_solver_lumerical
  solver = fdtd_solver_lumerical(component=c, tech=t, wavelength_start=1.5, ...)

  # after
  from gds_fdtd.solvers import get_solver
  from gds_fdtd.spec import SimulationSpec
  solver = get_solver("lumerical")(c, t, SimulationSpec(...))
  smatrix = solver.run()
  ```

  Per SemVer this is a pre-1.0 minor bump (0.5 → 0.6); 0.5 shipped the
  modern `get_solver` API and moved all docs/examples to it.

- The legacy dict-flavored `gds_fdtd.core.technology` class (and its
  `from gds_fdtd import technology` re-export) was removed. Use the validated
  `gds_fdtd.technology.Technology` model (`Technology.from_yaml(...)`); its
  `.to_solver_dict()` reproduces the old dict shape if you need it. Nothing
  inside the package used the class.
- Dead code: the internal Tidy3D engine's unused results path
  (`run`/`get_resources`/`get_results`/`get_log`/`visualize_field_monitors`
  and the `sparameters` conversion) was deleted. The supported `Tidy3DSolver`
  adapter runs the cloud job and builds the canonical `SMatrix` itself; the
  engine only builds the scene. This also removes the engine's last dependency
  on the legacy `sparams` module.
- `gds_fdtd.core.parse_yaml_tech` was removed. It was a one-line bridge to
  `Technology.from_yaml(path).to_solver_dict()`; call sites now use
  `gds_fdtd.technology.Technology.from_yaml(path)` directly and pass the
  `Technology` to `load_component_from_tech`/`load_device`/the solvers (all of
  which accept it). Use `.to_solver_dict()` if you specifically need the dict.
- The `gds_fdtd.core` module itself was removed — the last thing it held was
  the PEP-562 shim serving the pre-0.5 lowercase geometry names (deprecated in
  0.5). **Migration:** import the geometry classes from `gds_fdtd.geometry`
  (`port`→`Port`, `structure`→`Structure`, `region`→`Region`,
  `component`→`Component`, `layout`→`LayoutSource`); the non-deprecated helpers
  (`c0_um`, `calculate_polygon_extension`, `initialize_ports_z`,
  `is_point_inside_polygon`) are exported from `gds_fdtd.geometry` too.
- The `gds_fdtd.sparams` module is no longer public — it moved to the internal
  `gds_fdtd._sparams` (the legacy `sparameters` container + INTERCONNECT `.dat`
  reader/writer). The supported surface is the `SMatrix` API (`SMatrix.from_dat`/
  `to_dat`, `from_hdf5`/`to_hdf5`, `from_npz`/`to_npz`, `to_touchstone`);
  `import gds_fdtd.sparams` now raises `ModuleNotFoundError`.
- Removed the stray pre-0.5 `examples/notebooks/faid/` notebook (unreferenced,
  superseded by the standardized `examples/0X_*` set; it had also once logged a
  license token).

### Added
- Per-engine **material-source selection**. A technology material can name up to
  three optical-constant sources — the engine's own database model (`tidy3d` /
  `lumerical`), a `refractiveindex.info` page (`rii`), and a neutral constant
  (`nk`) — and each engine picks one: an explicit `MaterialSpec.source`
  (`eda`/`rii`/`nk`) wins, else the precedence `eda → rii → nk`, else a clear
  `MaterialSourceError`. tidy3d builds a dispersive medium from `rii`
  (`RiiMaterial.to_tidy3d_medium`); Lumerical emits an `(n,k)` material; beamz a
  constant. New module `gds_fdtd.materials.select`; documented in
  `docs/technology.rst` and example `02b`.
- Seeded atheris fuzzing of the technology-YAML and INTERCONNECT `.dat`
  parsers (`fuzz/`), run directly on PRs that touch the library.
- Release artifacts are now Sigstore-signed (keyless, GitHub OIDC); the
  `.sigstore` bundles are attached to each GitHub Release.
- **Executed-notebook examples curriculum** (`00_quickstart` …
  `10b_polarization`, 13 notebooks): jupytext-paired `.py` sources committed
  alongside executed `.ipynb` with real solver outputs. Includes the
  three-engine y-branch comparison, the sharp-S-bend convergence study
  (converged ≠ correct), the frontend × engine matrix (gdsfactory / SiEPIC /
  raw GDS on beamz / tidy3d / Lumerical), and polarization devices (a
  directional-coupler PBS and gdsfactory's splitter-rotator, TE0+TM0 on two
  engines, with the symmetric-cladding negative control). Recorded artifacts
  ship with PROVENANCE notes so every notebook reproduces for free.
- **Visualization toolkit** (`gds_fdtd.plotting`): `plot_component` (geometry +
  auto-detected ports + FDTD region), `plot_tech_stack`, `plot_permittivity`,
  `plot_mode`, `smatrix_summary`, and `plot_field` with linear and log (dB)
  scales; every solver's `plot_fields()` gained a `scale=` argument and draws
  on the engine's true (non-uniform) grid. `gds_fdtd.modes.waveguide_mode`
  solves strip modes offline via tidy3d's local plugin.
- **Documentation overhaul**: the API reference now covers the whole public
  surface (27 module pages, grouped by topic); new frontends guide (what a
  frontend is, the built-ins, and how to write your own — verified example);
  the technology page rebuilt around the three material sources and
  refractiveindex.info; real figures throughout; sphinx-design landing page.
- **Top-level API exports**: `from gds_fdtd import Technology, SimulationSpec,
  get_solver` (joining `SMatrix` and the geometry classes).
- **Convenience install extras**: `pip install gds_fdtd[all]` (every engine +
  frontend) and `[engines]` (tidy3d + beamz).
- **Property-based tests** (hypothesis, new dev dependency) for the numeric
  core — S-matrix I/O round-trips, reciprocity/passivity invariants,
  port-extension geometry, technology v1↔v2 equivalence — plus a real beamz
  end-to-end test (slow-marked) in the CI all-extras leg. All-extras branch
  coverage rose 81% → 90.2%; the CI gate ratcheted 80 → 90.
- A written **deprecation policy** (CONTRIBUTING.md): two-minor-release
  window, `DeprecationWarning` + CHANGELOG + a warning-asserting test.

### Fixed
- `SMatrix.from_entries` (and therefore `SMatrix.from_dat`) rejected a
  corrupt mode/port index only by attempting an unbounded allocation — a
  malformed `.dat` claiming a ~10**6 mode id tried to allocate 623 TiB and
  raised `MemoryError`. It now raises a clean `ValueError` (found by the new
  `.dat` fuzz target).
- `compare_smatrices`/`max_delta_db` no longer return `+inf` when one engine
  has a dead (all-zero) column: values are clamped to the floor and compared
  in dB, so partial results read as a bounded disagreement.
- The tidy3d adapter crashed on lossy constant materials (`nk` with k > 0) by
  building a complex permittivity; both call sites now use `Medium.from_nk`
  (found live by the air-clad PSR run, whose superstrate is `nk: 1.0`).
- beamz now honors `spec.buffer` and `spec.z_min`/`z_max` whenever they exceed
  its physical guard-band floors (computed over all device layers, so
  multilayer stacks get full clearance); default jobs are bit-identical.
- Field plots render on the engine's true grid coordinates: tidy3d's adaptive
  mesh drawn with a uniform `imshow` stretched the finely-meshed core ~2× and
  made the waveguide look wider than simulated (setup-parity audit, example
  06). All field figures and `plot_fields` now use `pcolormesh`.
- `lyprocessor` port detection: a >2-point pin path *returned* an exception
  instead of raising it, diagonal pins fell through silently, and an unlabeled
  pin produced a `None` port name — all three now raise `LayoutError` (caught
  by the whole-package strict-typing pass).
- The tidy3d workdir now defaults under the current directory instead of the
  system temp dir (often a small RAM-backed tmpfs), and result downloads are
  routed there too — multi-GB field downloads no longer die with
  `OSError 28: No space left on device`.
- `smatrix_summary` uses an engineering reciprocity/passivity tolerance
  (1e-2) suited to real FDTD output instead of flagging every physical result.

### Changed
- (BREAKING) Renamed the technology-to-dict method: `Technology.to_legacy_dict()`
  → `Technology.to_solver_dict()`, and the per-material `MaterialSpec.to_legacy()`
  → `MaterialSpec.to_solver_dict()`. The returned schema-v1 dict — consumed by the
  simprocessor, the gdsfactory bridge, the CLI, and the tidy3d/lumerical/beamz
  adapters — is unchanged; only the misleading "legacy" name is gone. No
  deprecation alias (consistent with the rest of this 0.6.0 cleanup). Migration:
  replace `tech.to_legacy_dict()` with `tech.to_solver_dict()`.
- The Tidy3D scene-building engine is now fully internal, with a name that reflects its role.
  The generic `fdtd_solver`/`fdtd_port` engine classes (the last pre-0.5 names
  in the tree) were renamed to `_TidyEngineBase`/`_TidyPort`, and the module
  `solvers/_engine_base.py` → `solvers/_tidy3d_base.py`. The supported
  `Tidy3DSolver` adapter is the only entry point; the engine base no longer
  imports the legacy `core.technology` type (it took the modern `Technology`
  in practice). No public API changed — all of these names are internal.
- Errors raised on user-input paths (files, tech dicts, layouts, jobs,
  results I/O) now derive from the `GdsFdtdError` hierarchy — including the
  new `SMatrixError` — while dual-inheriting the builtin they replaced
  (`ValueError`, `RuntimeError`, …), so existing `except` clauses keep
  working. `materials.rii`'s missing-page error changed base from
  `FileNotFoundError` to `TechnologyError` (still a `ValueError`).
- `mypy --strict` now passes on the **whole package** (34 files) and is a
  required CI gate (was: 13-module core + advisory rest).
- The internal `_sparams` module logs through the package logger instead of
  printing (library etiquette: silent unless logging is configured).
- One unified project description across pyproject/README/docs.

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
