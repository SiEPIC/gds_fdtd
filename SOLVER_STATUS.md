# Per-solver verification status

CI cannot run licensed or credit-spending engines per-PR, so this file is
the equivalent of a build badge: when each adapter was last verified
against the REAL engine, with what, and how well it agreed.

**Historical three-engine agreement (2026-07-08, identical job, zero engine-specific kwargs):**
tidy3d ↔ Lumerical within **0.0033 dB**; beamz 0.4.3 within 0.052 dB of both
(gf straight, mesh 10, unified tech — recorded in `tests/recorded/straight_mesh10_*.npz`,
asserted every PR by `test_three_engine_agreement`).

| engine | last verified | version | evidence |
|---|---|---|---|
| tidy3d (cloud) | 2026-07-13 | 2.11.2 | crossing 4-port×2-mode matrix re-recorded at 51 wavelength points, TE+TM (~0.54 FC; `examples/01`/`04` recorded); S-bend convergence + injected-mode overlay (`examples/06`); PBS + PSR polarization matrices (`examples/10b` via `10_cookbook/recorded`); budget-gated cloud smoke; artifacts replayed every PR |
| Lumerical FDTD | 2026-07-13 | 2025 R2 (v252) | PBS + PSR full polarization matrices on local license (PSR: 10.9 h); S-bend convergence + injected-mode overlay within **0.03 dB** of tidy3d (`examples/06`); escalator full matrix; artifacts replayed every PR |
| beamz | 2026-10-04 | 0.5.0 / 0.5.1 / 0.5.2 | Local RTX 3090: full 2-port 5 µm straight, mesh 10, 11 wavelengths; all runs converged. Latest S21 within 0.0211 dB of recorded Tidy3D and 0.0191 dB of recorded Lumerical. [Results, plots, and reproduction](benchmarks/README.md). Mesh-6 short-waveguide regression covers fields and cache. Latest also runs full y-branch and escalator matrices and S-bend mesh sweep: [device results](benchmarks/DEVICE_RESULTS.md). S-bend monitor-plane sensitivity remains open in [BeamZ #309](https://github.com/beamzorg/beamz/issues/309). |

**tidy3d version note (0.6.2):** the `tidy3d` extra now floors at **2.12.0**,
but the live cloud evidence in the table above was gathered on **2.11.2** — the
dates and FlexCredit costs are unchanged and should not be read as a 2.12 live
validation. What *was* checked on 2.12.0 is the offline half: the tidy3d-facing
suites pass (25 passed, 1 skipped) and a real scene builds correctly against it
(modeler construction, media, and the 0.6.1 monitor controls — pinned plane
position and single recorded wavelength). The cloud `run()` path on 2.12.0 is
still unexercised; the next `cloud-smoke` run is what refreshes this row.

Refresh procedure: `cloud-smoke` workflow (tidy3d, budget-gated, human
approval), `lumerical-nightly` workflow (self-hosted lane, see
`docs/self_hosted_runner.md`), the beamz examples (`00_quickstart`,
`06_convergence_and_caching`, `10_cookbook`) locally. Update this table in
the same PR that lands refreshed `tests/recorded/` artifacts.
