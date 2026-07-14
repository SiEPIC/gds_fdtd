# Recorded results — `crossing_te1550` (README/docs figures)

Real tidy3d output for the waveguide crossing shipped in `examples/devices.gds`,
used for the README hero and the docs. The `01_layout_to_component` notebook
itself is offline (it only loads and inspects the geometry); these artifacts
back the figures next to it.

## The job

- **Device:** `crossing_te1550` from `examples/devices.gds` (4 ports: opt1 west,
  opt4 east, opt2 north, opt3 south).
- **Technology:** `examples/tech.yaml`.
- **Spec:** `SimulationSpec(wavelength_start=1.5, wavelength_end=1.6,`
  `wavelength_points=51, mesh=14, modes=(1, 2), z_min=-1.0, z_max=1.11,`
  `field_monitors=("z",))` — TE + TM, 51 wavelengths.
- **Engine:** tidy3d 2.11.x (cloud), ~0.54 FlexCredits.

## Files

| file | what |
|------|------|
| `crossing_te1550_tidy3d.npz` | full 4-port × 2-mode S-matrix over 51 wavelengths |
| `crossing_te1550_field_tidy3d.npz` | z-plane \|E\|² at 1.55 µm, opt1 TE0 excitation |

## Result @ 1.55 µm (TE0 into opt1)

Insertion loss (opt1→opt4) −0.49 dB (TE) / −0.50 dB (TM); crosstalk
(opt1→opt2, opt1→opt3) −41 dB; reflection (opt1→opt1) −33 dB. The through-loss
sits near 0 dB while crosstalk and reflection are tens of dB lower, which is why
the README plots them on a dual y-axis.

Recorded 2026-07-14. Regenerate by running the crossing on tidy3d at the spec
above (`get_solver("tidy3d")(...).run()`), or on Lumerical for a license-based
reference.
