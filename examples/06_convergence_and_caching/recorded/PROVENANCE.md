# Recorded results — `sbend_dontfabme` cross-engine convergence (06 §3)

Real solver output for the `sbend_dontfabme` device (a deliberately sharp S-bend
from `examples/devices.gds`), used by `06_convergence_and_caching` §3 so the
cross-validation study reproduces **for free** — the notebook loads these; no
tidy3d key or FlexCredits and no multi-hour beamz run are needed to re-render it.

| file | what | how |
|------|------|-----|
| `sbend_beamz_convergence.json` | \|S21\|²/\|S11\|² in dB at 1.55 µm vs mesh | beamz `run_cached` per mesh, S at 1.55 µm |
| `sbend_tidy3d_convergence.json` | same, vs mesh (6→25) | tidy3d `run()` per mesh via `ModalComponentModeler` |
| `sbend_lumerical_convergence.json` | same, vs mesh 6→22 ppw (Lumerical mesh accuracy 1→5) | Lumerical `run()` per mesh on a licensed workstation (2026-07-12; ~5 min total) |
| `sbend_beamz_field.npz` | z-plane \|E\|² at mesh 12 (+ frame size) | beamz `run()` field monitor |
| `sbend_tidy3d_field.npz` | z-plane \|E\|² at mesh 12 (+ x/y coords) | tidy3d `z_field` monitor at 1.55 µm |
| `sbend_lumerical_field.npz` | z-plane \|E\|² at ppw 14 / accuracy 3 (+ x/y coords) | Lumerical `profile_z` monitor at 1.55 µm, opt1 excitation |
| `sbend_injected_modes.npz` | launched TE0 lateral profile + n_eff, each engine | each engine's mode solver on the 0.5×0.22 µm Si guide (0.02 µm grid) |

- **Engines:** beamz 0.4.3 (CPU, JAX); tidy3d 2.11.2 (cloud). Both: single
  mode/port, 1.5–1.6 µm (3 pts), `z_min=-1.0`, `z_max=1.11`, `mesh` =
  points/wavelength.
- **beamz mesh range:** 3→20 (mesh 20 alone took ~37 min; 25 is ~hours on CPU). beamz is 3D FDTD on CPU with cost ∝
  ~mesh⁴; mesh 16 already takes ~15 min and mesh 25 ~1.5 h, so the high end is
  recorded rather than re-run live (the notebook runs beamz live only for the
  fast straight sweep in §1–2). tidy3d (cloud) reaches mesh 25 cheaply.
- **Recorded:** 2026-07-11. tidy3d cost ≈ 0.45 FlexCredits total (this device is
  small — every task hit the ~0.05 FC/task floor).
- **Headline:** tidy3d converges cleanly to S21 ≈ −5.64 dB / S11 ≈ −27.6 dB, and
  **Lumerical independently confirms it** — S21 ≈ −5.63 dB / S11 ≈ −27.7 dB, flat
  from mesh-accuracy 2 onward, i.e. the two references agree within ~0.03 dB on
  the identical job (~72 % radiated/mode-converted by the sharp bend). beamz
  never converges — its S21 wanders between ≈ −4.7 and −2.0 dB across the sweep
  (mesh 20, its finest here, is −1.99 dB, its *farthest* from the references),
  i.e. it under-counts the mode conversion (v1 single-mode per-direction
  normalization) and more mesh does not close the gap. That non-convergence is
  itself the finding. See the §3 discussion.

To regenerate on a licensed machine: set `TIDY3D_API_KEY`, sweep both engines
over `sbend_dontfabme` at the spec above, and rewrite these files (see the
scratch scripts, or the recipe in the notebook).

## Setup-parity audit (2026-07-12)

Prompted by "tidy3d looks like a wider waveguide than beamz" in the field
figure, the full per-engine setup was audited:

- **Geometry parity confirmed.** The identical polygons reach both engines;
  beamz's rasterized core measures 0.483 µm vs the 0.500 µm drawn (dx=37 nm
  quantization). Launched TE0 and materials (@1.55 µm, <0.001) also match.
- **The "wider waveguide" was a rendering bug**, not a setup difference:
  tidy3d's z-plane grid is non-uniform (35.7–89.2 nm cells), and drawing it
  with `imshow(extent=…)` (uniform pixels) stretched the finely-meshed core —
  measured: true mode FWHM 0.396 µm rendered as 0.793 µm. The figure now uses
  `pcolormesh` on the true grid coordinates; in true coordinates the two
  engines' guided fields have matching FWHM (0.32–0.72 µm at the same
  stations) and matching output peak positions.
- **Documented engine floors that legitimately differ** (PML-absorbed, no
  material effect on this job): beamz in-plane guard band 3.0 µm/side vs the
  1.0 µm buffer; beamz mode planes 2.7×2.2 µm vs `width_ports×depth_ports`
  2.0×1.5 µm; uniform vs adaptive grid; constant-nk vs dispersive materials;
  vertical vs 85° sidewalls. beamz now honors `spec.z_min/z_max` (and
  `spec.buffer`) whenever they exceed its floor — for THIS job the resulting
  geometry is bit-identical to what was recorded, so these artifacts remain
  valid.
- The S-parameter verdict is unchanged by the audit: tidy3d converges to
  S21 ≈ −5.6 dB; beamz wanders and never converges (its v1 single-mode
  normalization under-counts the bend's mode conversion).
