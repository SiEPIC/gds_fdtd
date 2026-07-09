# Recorded solver artifacts (replay testing — WP7.1.4)

Real solver outputs, committed so CI exercises the full results pipeline with
zero cloud/license access. All files are scrubbed before commit
(`grep -riE 'licens|handler|api[_-]?key|token' tests/recorded/` must be empty).

| file | origin | refresh cost |
|---|---|---|
| `si_sin_escalator_smatrix.h5` | tidy3d 2.8.5 cloud run, 2026-07-07: escalator fixture, mesh=6, 11 wavelength pts, 2 tasks | ≈0.05 FlexCredits |
| `si_sin_escalator.dat` | same run, exported via SMatrix.to_dat | — |
| `si_sin_escalator_lum_smatrix.h5` | Lumerical FDTD v252 (2025 R2) local run, 2026-07-07: same fixture/settings | local license |

Refresh procedure: with `TIDY3D_API_KEY` configured (`~/.tidy3d/config`), run a
solver on `tests/si_sin_escalator.gds` + `tests/tech_tidy3d.yaml` (coarse mesh,
few wavelengths), export via `sparameters.to_smatrix()` → `to_hdf5()`/`to_dat()`,
re-run the scrub grep, and update this table.

NOTE (finding F5): tidy3d 2.8.x requires numpy<2 (undeclared) and therefore a
Python ≤3.12 environment — the recording env was a py3.11 venv. tidy3d ≥2.11
(WP4.1) resolves this.

## straight_mesh10_{tidy3d,lumerical,beamz}.npz

The SAME gdsfactory straight (L=5 um), unified `examples/tech.yaml`,
`SimulationSpec(wavelength_points=11, mesh=10, z_min=-1, z_max=1.22)`,
zero engine-specific kwargs, recorded 2026-07-08:
tidy3d 2.11.2 cloud (~0.05 FC), Lumerical 2025 R2 local license,
beamz 0.4.3 CPU. Refresh with scratch script `threeway/run_one.py`
(same three commands, one argument changed).
