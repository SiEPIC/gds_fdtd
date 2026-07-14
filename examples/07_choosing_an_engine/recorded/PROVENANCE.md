# Recorded three-engine results — `ebeam_y_1550`

Real S-matrices from one run of each engine on the **identical** job. Loaded by
`07_choosing_an_engine`, so the notebook re-executes for free without touching a
cloud account, a license, or much memory.

## The job

- **Device:** SiEPIC EBeam `ebeam_y_1550` 1×2 y-branch (3 ports, all x-facing),
  loaded from the KLayout/SiEPIC PDK — no gdsfactory.
- **Technology:** `examples/tech.yaml` (Si core n≈3.48, SiO₂ cladding).
- **Spec:** `SimulationSpec(wavelength_start=1.5, wavelength_end=1.6,`
  `wavelength_points=5, mesh=6, z_min=-1.0, z_max=1.11, buffer=0.8)` — kept
  small (~3.1 M cells, ~0.15 GB) so the local engines fit a modest laptop.

## Files

| File | Engine | Runtime | Cost |
|------|--------|---------|------|
| `ybranch_beamz.npz` | beamz 0.4.3 | local (JAX, CPU) | free |
| `ybranch_tidy3d.npz` | tidy3d 2.11.2 | cloud | ~0.075 FlexCredits |
| `ybranch_lumerical.npz` | Ansys Lumerical FDTD (2025) | local workstation | one license seat |

Recorded 2026-07-10 on the development environment. Regenerate with
`solver.run_cached(cache_dir)` on the same job (the cache key includes engine
versions, so a version bump re-runs).

## Summary

Input split (opt1 → opt2/opt3), mean over the band, vs the −3.01 dB ideal:

| Engine | opt2←opt1 | opt3←opt1 |
|--------|-----------|-----------|
| beamz | −3.206 dB | −3.293 dB |
| tidy3d | −3.233 dB | −3.233 dB |
| lumerical | −3.249 dB | −3.249 dB |

Three independent FDTD engines agree on the split to well under 0.2 dB.

## Known limitation (beamz v1)

beamz's `opt3` reverse-excitation column underflowed to zero (its mode source
under-injects on one of the two same-direction output ports), so beamz reads as
non-reciprocal and its reverse columns are not trustworthy at this mesh. This is
a beamz engine-level mode-injection issue, not the gds_fdtd adapter; tidy3d and
Lumerical resolve every column. The reported input-split metric uses only the
`opt1` excitation and is unaffected.
