# Per-solver verification status

CI cannot run licensed or credit-spending engines per-PR, so this file is
the honest equivalent of a build badge: when each adapter was last verified
against the REAL engine, with what, and how well it agreed.

**Three-engine agreement (2026-07-08, IDENTICAL job, zero engine-specific kwargs):**
tidy3d ↔ Lumerical within **0.0033 dB**; beamz within 0.052 dB of both
(gf straight, mesh 10, unified tech — recorded in `tests/recorded/straight_mesh10_*.npz`,
asserted every PR by `test_three_engine_agreement`).

| engine | last verified | version | evidence |
|---|---|---|---|
| tidy3d (cloud) | 2026-07-08 | 2.11.2 | crossing 4-port×2-mode matrix on the UNIFIED tech (example 03a as committed, 0.292 FC); live mesh-convergence sweep (04b); budget-gated cloud smoke; recorded artifact in `tests/recorded/` replayed every PR |
| Lumerical FDTD | 2026-07-07 | 2025 R2 (v252) | escalator full matrix on local license; agrees with tidy3d within **0.081 dB** (worst pair, `compare_smatrices`); recorded artifact replayed every PR |
| beamz | 2026-07-08 | 0.4.3 | example 06a as committed, fully agnostic setup (indices auto-resolved from the unified tech): thru ≈0 dB, S11 −31…−41 dB; F9 (reference-monitor mis-normalization) found and fixed earlier |

Refresh procedure: `cloud-smoke` workflow (tidy3d, budget-gated, human
approval), `lumerical-nightly` workflow (self-hosted lane, see
`docs/self_hosted_runner.md`), `examples/06_beamz` locally. Update this
table in the same PR that lands refreshed `tests/recorded/` artifacts.
