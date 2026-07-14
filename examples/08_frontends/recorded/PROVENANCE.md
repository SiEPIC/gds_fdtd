# Recorded results — `gf_bend_s` on three engines (08 §6)

Real S-matrices for the gdsfactory-authored gentle S-bend
(`gf.components.bend_s(size=(5.0, 1.0))`, gpdk cross-section, 2 x-facing
ports), converted with `layout.gdsfactory.from_gdsfactory` and run with the
**identical** `(component, technology, spec)` on all three engines. This is
the gdsfactory row of 08's frontend × engine matrix; the SiEPIC and raw-GDS
rows reuse the artifacts recorded for `07_choosing_an_engine` and
`06_convergence_and_caching`.

## The job

- **Technology:** `examples/tech.yaml` (Si core n≈3.48, SiO₂ cladding).
- **Spec:** `SimulationSpec(wavelength_start=1.5, wavelength_end=1.6,`
  `wavelength_points=5, mesh=6, z_min=-1.0, z_max=1.11)`.
- Deliberately small (~1.0 M cells on beamz) so the local engines stay well
  inside a modest laptop's memory.

## Files

| file | engine | where it ran | cost |
|------|--------|--------------|------|
| `gf_bend_s_beamz.npz` | beamz 0.4.3 (JAX, CPU) | this machine | free |
| `gf_bend_s_tidy3d.npz` | tidy3d 2.11.2 | cloud | ~0.05 FC (budget-gated at 0.5) |
| `gf_bend_s_lumerical.npz` | Ansys Lumerical FDTD 2025 | this machine (license seat) | one seat·run |

Recorded 2026-07-12. Regenerate with the script recipe in the notebook (§6) or
`solver.run_cached(...)` on the same job.

## Summary

A gentle (adiabatic) bend is beamz-friendly physics: all three engines agree
on S21 across the band (see the executed notebook for the exact spread),
in contrast to the sharp `sbend_dontfabme` where beamz never
converges (that story is `06_convergence_and_caching` §3).

## Field artifacts (added with the fields/modes gap-fill)

`gf_bend_s_field_{beamz,tidy3d,lumerical}.npz` — z-plane |E|² at 1.55 µm for
the `o1` excitation of the same job. tidy3d/Lumerical were extracted from the
original run artifacts (no re-run, no cost; coords are each engine's true
grid); beamz from one identical re-run. Rendered in 08 §6 as the three-engine
field row.
