# Recorded results — `sbend_dontfabme` cross-engine convergence (06 §3)

Real solver output for the `sbend_dontfabme` device (a deliberately sharp S-bend
from `examples/devices.gds`), used by `06_convergence_and_caching` §3 so the
cross-validation study reproduces **for free** — the notebook loads these; no
tidy3d key or FlexCredits and no multi-hour beamz run are needed to re-render it.

| file | what | how |
|------|------|-----|
| `sbend_beamz_convergence.json` | \|S21\|²/\|S11\|² in dB at 1.55 µm vs mesh | beamz `run_cached` per mesh, S at 1.55 µm |
| `sbend_tidy3d_convergence.json` | same, vs mesh (6→25) | tidy3d `run()` per mesh via `ModalComponentModeler` |
| `sbend_beamz_field.npz` | z-plane \|E\|² at mesh 12 (+ frame size) | beamz `run()` field monitor |
| `sbend_tidy3d_field.npz` | z-plane \|E\|² at mesh 12 (+ x/y coords) | tidy3d `z_field` monitor at 1.55 µm |
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
- **Headline:** tidy3d converges cleanly to S21 ≈ −5.64 dB / S11 ≈ −27.6 dB
  (~72 % radiated/mode-converted by the sharp bend). beamz never converges — its
  S21 wanders between ≈ −4.7 and −2.0 dB across the sweep (mesh 20, its finest
  here, is −1.99 dB, its *farthest* from tidy3d), i.e. it under-counts the mode
  conversion (v1 single-mode per-direction normalization) and more mesh does not
  close the gap. That non-convergence is itself the finding. See the §3 discussion.

To regenerate on a licensed machine: set `TIDY3D_API_KEY`, sweep both engines
over `sbend_dontfabme` at the spec above, and rewrite these files (see the
scratch scripts, or the recipe in the notebook).
