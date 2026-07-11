# Recorded escalator S-matrices (reference for cross-validation)

Real engine output for the Si→SiN escalator (`../si_sin_escalator.gds`), used to
cross-check the free beamz run in `10_cookbook`. These are the **record of an
earlier live validation** on the licensed engines — the `10_cookbook` notebook
reads them for comparison and does **not** re-run tidy3d or Lumerical (that needs
cloud credits / a license). Re-generating them is the job of a machine with those
engines; see the live-validation runbook in the repo's `HANDOFF.md`.

- `si_sin_escalator_tidy3d.npz` — tidy3d 2.11.2 (cloud), full 2-port matrix.
- `si_sin_escalator_lum_smatrix` → `si_sin_escalator_lumerical.npz` — Lumerical FDTD 2025 R2 (v252).

Converted (npz, no extra deps to read) from `tests/recorded/si_sin_escalator_smatrix.h5`
and `si_sin_escalator_lum_smatrix.h5`. See `SOLVER_STATUS.md` for the last-verified
dates. Sanitized: contain only f / S / port names — no credentials or hostnames.
