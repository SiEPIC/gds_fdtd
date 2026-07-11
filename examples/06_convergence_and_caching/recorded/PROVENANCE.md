# Recorded tidy3d results — `sbend_dontfabme` cross-engine convergence

These are **real tidy3d cloud results** for the `sbend_dontfabme` device (a
deliberately sharp S-bend from `examples/devices.gds`), used by
`06_convergence_and_caching` so the cross-engine section reproduces **for free**
(the notebook runs beamz live and loads these; no tidy3d key or FlexCredits
needed to re-run it).

| file | what | how |
|------|------|-----|
| `sbend_tidy3d_convergence.json` | \|S21\|²/\|S11\|² in dB at 1.55 µm vs mesh (6→25) | tidy3d `run()` per mesh via the `ModalComponentModeler`, S at 1.55 µm |
| `sbend_tidy3d_field.png` | \|E\| top-down field profile at mesh 10 | tidy3d `z_field` monitor from the same run |

- **Engine:** tidy3d 2.11.2 (cloud), `min_steps_per_wvl = mesh`, single mode/port,
  1.5–1.6 µm (3 points), domain = device bounds + 1 µm buffer.
- **Recorded:** 2026-07-11. Total cost ≈ 0.45 FlexCredits (this device is small
  enough that every task hit the ~0.05 FC/task floor).
- **Headline:** tidy3d converges cleanly to S21 ≈ −5.64 dB / S11 ≈ −27.6 dB
  (~72 % of the power is radiated/mode-converted by this sharp bend). beamz — run
  live in the notebook — plateaus near −2.8 dB instead, i.e. it under-counts the
  mode-conversion loss (its v1 single-mode per-direction normalization is not
  suited to a strongly mode-converting bend). See the notebook discussion.

To regenerate on a licensed machine: set `TIDY3D_API_KEY`, run the tidy3d sweep
`[6, 8, 10, 12, 16, 20, 25]` on `sbend_dontfabme` with the spec above, and rewrite
these files.
