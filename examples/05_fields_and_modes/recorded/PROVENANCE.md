# Recorded artifacts — 05b field monitors

| file | what |
|---|---|
| `escalator_tidy3d.npz` | Si→SiN escalator S-matrix, 2 ports × 1 mode, 11 points over 1.50–1.60 µm |
| `escalator_field_y.npz` | \|E\|² on the y-normal side plane (x–z), recorded at 1.55 µm, o1 excitation |
| `escalator_field_z.npz` | \|E\|² on the z-normal plane pinned at z = 0.11 µm (Si-core mid-plane), 1.55 µm, o1 excitation |

Run: tidy3d 2.11.2 cloud, 2026-07-17. Layout `examples/10_cookbook/si_sin_escalator.gds`
(cell `si_sin_escalator`), technology `examples/tech.yaml` (substrate pinned to rii,
`GDS_FDTD_RII_DB=examples/02_technology/rii_db` at build time).

Spec: 1.50–1.60 µm, 11 points, mesh 10, z −1.2…1.5, `field_monitors=("y","z")`,
`field_monitor_positions={"z": 0.11}`, `field_monitor_wavelengths=(1.55,)`.
Cost: ~0.026 FC/sim × 2 sims ≈ 0.05 FC (cloud estimate at submission).
Result sanity: through path −0.07…−0.06 dB across the band.

npz schema (fields): `h`, `v` — plane coordinates [µm]; `mag2` — (n_λ, len(v), len(h))
\|E\|²; `wavelength_um`; `axis`; `task` (excitation task name).
