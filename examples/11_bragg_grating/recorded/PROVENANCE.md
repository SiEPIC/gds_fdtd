# Recorded artifacts — 11 Bragg grating

| file | what |
|---|---|
| `bragg_tidy3d.npz` | `bragg_te1550` S-matrix, 2 ports × 1 mode, 501 points over 1.50–1.60 µm |
| `bragg_field_z.npz` | \|E\|² on the z-plane pinned at z = 0.11 µm, recorded at λ = 1.5424, 1.5427, 1.5430, 1.5433, 1.5436 (comb on the Bragg wavelength) + 1.51, 1.59 µm, opt1 excitation |

Run: tidy3d 2.11.2 cloud, 2026-07-17. Layout `examples/devices.gds` (cell
`bragg_te1550`, SiEPIC EBeam, 95.1 µm), technology `examples/tech.yaml`
(`GDS_FDTD_RII_DB=examples/02_technology/rii_db` at build time).

Spec: 1.50–1.60 µm, 501 points (0.2 nm), mesh 10, z −1.0…1.0,
`field_monitors=("z",)`, `field_monitor_positions={"z": 0.11}`,
`field_monitor_wavelengths=(1.5424, 1.5427, 1.5430, 1.5433, 1.5436, 1.51, 1.59)`
— the comb center 1.5430 is the Bragg wavelength measured by a first coarse
(101-pt) pass on the same mesh.
Cost: ~0.272 FC/sim × 2 sims ≈ 0.54 FC (cloud estimate at submission); the
superseded coarse pass spent ~0.41 FC.

Measured: stopband (S21 < −10 dB) 1.5414–1.5450 µm (3.6 nm), S21 min
−14.3 dB at 1.5432 µm with S11 ≈ 0 dB there; the recorded comb wavelength
nearest the minimum is 1.5433 µm (0.1 nm off — inside the sweep's own 0.2 nm
resolution). Power balance 0.84–1.12 (band-edge radiation, and mesh-10
passivity overshoot at the reflection peak).

npz schema (fields): `h`, `v` — plane coordinates [µm]; `mag2` — (n_λ, len(v), len(h))
\|E\|²; `wavelength_um`; `axis`; `task` (excitation task name).
