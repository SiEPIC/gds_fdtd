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

| # | Notebook | You'll learn | Engine | Runs in CI? |
|---|----------|--------------|--------|-------------|
| **00** | `00_quickstart` | Layout → S-matrix in ten lines | beamz (free) | build-only |
| **01** | `01_layout_to_component` | Load a GDS / gdsfactory cell, auto-detect ports, read the geometry view | none | ✅ offline |
| **02** | `02_technology` | Materials, `refractiveindex.info`, the vertical layer stack | none | ✅ offline |
| **03** | `03_first_simulation` | The full flow end-to-end: geometry → permittivity → build → run → S-params → fields | beamz (free) | build-only |
| **04** | `04_reading_results` | `SMatrix`: insertion loss, crosstalk, phase, reciprocity/passivity, Touchstone/HDF5/npz I/O | none | ✅ offline |
| **05** | `05_fields_and_modes` | Waveguide mode profiles, permittivity cross-sections, field maps | tidy3d-local (free) | ✅ offline |
| **06** | `06_convergence_and_caching` | Mesh/resolution convergence sweeps and `run_cached` (repeat runs are free) | beamz (free) | build-only |
| **07** | `07_choosing_an_engine` | The identical job on beamz / tidy3d / Lumerical, and how they agree | all three | build-only |
| **08** | `08_frontends` | gdsfactory, SiEPIC/KLayout, and PreFab (litho-prediction) front ends | mixed | ✅ offline |
| **09** | `09_cli_and_jobs` | The `gds-fdtd` CLI and serializable `JobSpec` for remote/batch compute | none | ✅ offline |
| **10** | `10_cookbook` | Reference devices with known-good S-params — starting with the **Si→SiN escalator** (a multi-layer device the free engine handles, cross-checked against recorded tidy3d/Lumerical) | beamz + recorded | build-only |

## How the notebooks are built

- **Source of truth is the paired `.py`** (jupytext *percent* format) — diff-friendly
  and reviewable. The committed `.ipynb` is that script **executed, with real
  outputs** (plots and numbers) so the gallery shows genuine physics, not code.
- **Two execution tiers.** *Offline* notebooks (01, 02, 04, 05, 08, 09) run
  end-to-end in CI on every PR. *Engine* notebooks (00, 03, 06, 07, 10) are
  executed **locally on a real engine** and committed as an artifact of record;
  CI runs them only through `build()` (offline) and checks structure/imports.
- **beamz is free** (Apache-2.0, JAX on CPU/GPU) and is the default engine for
  the executed outputs here — no cloud account, license, or GPU required to
  reproduce them.

## Running one yourself

```bash
uv sync --extra dev --extra beamz --extra gdsfactory   # + tidy3d / siepic / prefab as needed
uv run jupyter execute examples/03_first_simulation/03_first_simulation.ipynb
# or open the paired .py directly in Jupyter (jupytext round-trips it to a notebook)
```

See the top-level `HANDOFF.md` for the live-validation runbook and per-engine
setup (tidy3d credits, Lumerical license, GPU).
