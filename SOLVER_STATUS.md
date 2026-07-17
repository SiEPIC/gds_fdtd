# Per-solver verification status

CI cannot run licensed or credit-spending engines per-PR, so this file is
the equivalent of a build badge: when each adapter was last verified
against the REAL engine, with what, and how well it agreed.

**Three-engine agreement (2026-07-08, IDENTICAL job, zero engine-specific kwargs):**
tidy3d ↔ Lumerical within **0.0033 dB**; beamz within 0.052 dB of both
(gf straight, mesh 10, unified tech — recorded in `tests/recorded/straight_mesh10_*.npz`,
asserted every PR by `test_three_engine_agreement`).

| engine | last verified | version | evidence |
|---|---|---|---|
| tidy3d (cloud) | 2026-07-13 | 2.11.2 | crossing 4-port×2-mode matrix re-recorded at 51 wavelength points, TE+TM (~0.54 FC; `examples/01`/`04` recorded); S-bend convergence + injected-mode overlay (`examples/06`); PBS + PSR polarization matrices (`examples/10b` via `10_cookbook/recorded`); budget-gated cloud smoke; artifacts replayed every PR |
| Lumerical FDTD | 2026-07-13 | 2025 R2 (v252) | PBS + PSR full polarization matrices on local license (PSR: 10.9 h); S-bend convergence + injected-mode overlay within **0.03 dB** of tidy3d (`examples/06`); escalator full matrix; artifacts replayed every PR |
| beamz | 2026-07-15 | 0.4.3 | slow-marked real end-to-end straight run executes in the CI all-extras leg on every PR (thru > −0.5 dB, reflections < −15 dB, reciprocity); beamz examples (`00`/`03`/`06`/`08`/`10`) executed 2026-07-14 |

Refresh procedure: `cloud-smoke` workflow (tidy3d, budget-gated, human
approval), `lumerical-nightly` workflow (self-hosted lane, see
`docs/self_hosted_runner.md`), the beamz examples (`00_quickstart`,
`06_convergence_and_caching`, `10_cookbook`) locally. Update this table in
the same PR that lands refreshed `tests/recorded/` artifacts.
