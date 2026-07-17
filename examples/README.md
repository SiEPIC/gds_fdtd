# gds_fdtd examples — a guided tour

A learning path from *"load a layout"* to *"run it on any FDTD engine and read
the S-parameters."* Work through it in order the first time; after that, each
notebook stands alone as a reference.

Every simulation in gds_fdtd is the same three inputs → one output:

```python
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

solver  = get_solver("beamz" | "tidy3d" | "lumerical")(component, technology, SimulationSpec())
smatrix = solver.run()      # the ONLY call that spends money / a license / GPU
```

`validate()`, `build()`, and `estimate()` are always **offline and free** — you
preview the whole simulation before `run()` ever touches an engine.

## The path

| # | Notebook | You'll learn | Reproduce |
|---|----------|--------------|-----------|
| **00** | `00_quickstart` | Layout → S-matrix in ten lines | beamz · free |
| **01** | `01_layout_to_component` | Load a GDS / gdsfactory cell, auto-detect ports, read the geometry view | offline · free |
| **02** | `02_technology` | Materials, `refractiveindex.info`, the vertical layer stack | offline · free |
| **03** | `03_first_simulation` | The full flow end-to-end: geometry → permittivity → build → run → S-params → fields | beamz · free |
| **04** | `04_reading_results` | `SMatrix`: insertion loss, crosstalk, phase, reciprocity/passivity, Touchstone/HDF5/npz I/O | recorded · free, offline |
| **05** | `05_fields_and_modes` | Waveguide mode profiles, effective indices, permittivity cross-sections | local mode solver · free |
| **05b** | `05_fields_and_modes` | Field monitors: axes (top/side views), pinned positions, recorded wavelengths, and `plot_monitor_planes` — the escalator's side view shows light climbing from Si into SiN | recorded (tidy3d) · free |
| **06** | `06_convergence_and_caching` | Mesh convergence, `run_cached` (repeat runs free), and cross-engine validation on a device where *converged ≠ correct* | beamz + recorded · free |
| **07** | `07_choosing_an_engine` | The identical job on beamz / tidy3d / Lumerical, and how they agree | recorded (3 engines) · free, offline |
| **08** | `08_frontends` | Any EDA in, any engine out: gdsfactory / SiEPIC / raw-GDS frontends, then the full **frontend × engine matrix** (3 devices × beamz/tidy3d/Lumerical) | offline + recorded · free |
| **09** | `09_cli_and_jobs` | The `gds-fdtd` CLI and serializable `JobSpec` for remote/batch compute | offline · free |
| **10** | `10_cookbook` | Reference devices with known-good S-params — the **Si→SiN escalator**, a multi-layer device the free engine handles, cross-checked against recorded tidy3d/Lumerical | beamz + recorded · free |
| **10b** | `10b_polarization` | Polarization devices from gdsfactory: TE0/TM0 modes, a directional-coupler **PBS**, and the **polarization splitter-rotator** (incl. why it needs air cladding) — multi-mode S-params + per-polarization fields | recorded (Lumerical) · free |
| **11** | `11_bragg_grating` | A 95 µm SiEPIC **Bragg grating** from `devices.gds`: the stopband spectrum, and the same field monitor showing reflection in-band and transmission out-of-band, from one run | recorded (tidy3d) · free |

**Everything here reproduces for free.** No cloud account, license, or GPU is
required to run any notebook — the live runs use [beamz](https://github.com/beamzorg/beamz)
(Apache-2.0 JAX FDTD, CPU/GPU) or tidy3d's free *local* mode solver, and the
cross-engine comparisons (07, 10) load **recorded** tidy3d/Lumerical artifacts.
Those recorded results are the paid engines' real output, regenerated on a
licensed machine and committed as an artifact of record.

## How the notebooks are built

- **Source of truth is the paired `.py`** (jupytext *percent* format) — diff-friendly
  and reviewable. The committed `.ipynb` is that script **executed, with real
  outputs** (plots and numbers), so the gallery shows actual results rather than
  source alone.
- **Checked by CI.** `tests/test_examples_importable.py` parses every
  example on every PR and fails if it imports a `gds_fdtd` symbol that no longer
  exists — so an API change can't silently rot the gallery. Full execution is
  run locally and the executed `.ipynb` committed; a dedicated notebook-execution
  CI job is tracked in `ROADMAP.md` (WS1).
- **beamz is free** (Apache-2.0, JAX on CPU/GPU) and is the default engine for
  the live outputs here — no cloud account, license, or GPU required.

## Running one yourself

```bash
uv sync --extra dev --extra beamz --extra gdsfactory   # + tidy3d for the mode solver (05)
uv run jupyter execute examples/03_first_simulation/03_first_simulation.ipynb
# or open the paired .py directly in Jupyter (jupytext round-trips it to a notebook)
```

See the top-level `HANDOFF.md` for the live-validation runbook and per-engine
setup (tidy3d credits, Lumerical license, GPU).
