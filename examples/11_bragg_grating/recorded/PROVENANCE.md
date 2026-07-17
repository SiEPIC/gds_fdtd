# Recorded artifacts — 11 Bragg grating

| file | what |
|---|---|
| `bragg_tidy3d.npz` | `bragg_te1550` S-matrix, 2 ports × 1 mode, 101 points over 1.50–1.60 µm |
| `bragg_field_z.npz` | \|E\|² on the z-plane pinned at z = 0.11 µm, recorded at λ = 1.51, 1.545, 1.55, 1.555, 1.59 µm, opt1 excitation |

Run: tidy3d 2.11.2 cloud, 2026-07-17. Layout `examples/devices.gds` (cell
`bragg_te1550`, SiEPIC EBeam, 95.1 µm), technology `examples/tech.yaml`
(`GDS_FDTD_RII_DB=examples/02_technology/rii_db` at build time).

Spec: 1.50–1.60 µm, 101 points, mesh 10, z −1.0…1.0, `field_monitors=("z",)`,
`field_monitor_positions={"z": 0.11}`,
`field_monitor_wavelengths=(1.51, 1.545, 1.55, 1.555, 1.59)`.
Cost: ~0.205 FC/sim × 2 sims ≈ 0.41 FC (cloud estimate at submission).

Measured: stopband (S21 < −10 dB) 1.542–1.545 µm, S21 min −14.3 dB at
1.5430 µm with S11 ≈ 0 dB there; of the recorded field wavelengths, 1.545 µm
is in-band, the rest out-of-band. Power balance 0.84–1.12 (band-edge
radiation, and mesh-10 passivity overshoot at the reflection peak).

npz schema (fields): `h`, `v` — plane coordinates [µm]; `mag2` — (n_λ, len(v), len(h))
\|E\|²; `wavelength_um`; `axis`; `task` (excitation task name).
