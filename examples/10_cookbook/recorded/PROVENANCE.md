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

## Polarization records (10b_polarization)

Two gdsfactory devices, `modes=(1, 2)` (TE0 + TM0), recorded on **two engines**
from the identical `(component, technology, spec)`:

| file | what | engine | how |
|------|------|--------|-----|
| `pbs_coupler_{lumerical,tidy3d}.npz` | 4-port, 2-mode S-matrix of `gf.components.coupler(length=6, gap=0.2)` | Lumerical 2025 (local seat, ~11 min) / tidy3d 2.11.2 (cloud) | mesh 10 ppw, 1.5–1.6 µm ×5 |
| `pbs_coupler_field_{TE,TM}_in[_tidy3d].npz` | z-plane \|E\|² per polarization excitation | both | `profile_z` / `z_field` at 1.55 µm |
| `psr_{lumerical,tidy3d}.npz` | 3-port, 2-mode S-matrix of `gf.components.polarization_splitter_rotator()` with an **air superstrate** (`nk: 1.0`) | Lumerical (local, **10.9 h** — air-clad forces a fine graded mesh) / tidy3d (cloud, 200 s) | mesh 10 ppw, 1.5–1.6 µm ×3 |
| `psr_field_{TE,TM}_in[_tidy3d].npz` | z-plane \|E\|² per polarization excitation | both | at 1.55 µm |
| `psr_symclad_lumerical.npz` | the same PSR buried in **symmetric oxide** — the negative control | Lumerical | TM0 passes through unconverted (−0.59 dB): vertical mirror symmetry forbids TM0↔TE1 coupling |

Recorded 2026-07-12/13. tidy3d total for both devices ≈ **1.98 FC**
(budget-gated at 2.0; estimates verified before running).

Summary @1.55 µm:
- **PBS** splits: TE through −0.54 dB / TM cross −1.77 dB (Lumerical; tidy3d
  overlaid in the notebook). At length 12 µm TM beats across *and back*
  (recorded during tuning, not committed).
- **PSR (air-clad)**: TE0 → bus at −0.01 dB on **both** engines (exact
  agreement). TM0 conversion is **partial** with the stock cell in this stack
  (converted TE0 on the arm: −14.7 dB tidy3d / −21.3 dB Lumerical; the rest
  radiates in the taper) — vs the oxide control's zero interaction. The weak
  converted paths are mesh-sensitive and differ between engines: re-optimize
  stock cells for your stack, and cross-validate (the `06` lesson).
