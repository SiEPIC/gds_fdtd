# Per-solver verification status

CI cannot run licensed or credit-spending engines per-PR, so this file is
the honest equivalent of a build badge: when each adapter was last verified
against the REAL engine, with what, and how well it agreed.

| engine | last verified | version | evidence |
|---|---|---|---|
| tidy3d (cloud) | 2026-07-08 | 2.11.2 | escalator full matrix + live mesh-convergence sweep (example 04b, ≤0.15 FC); crossing 4-port×2-mode matrix (03a); recorded artifact in `tests/recorded/` replayed every PR |
| Lumerical FDTD | 2026-07-07 | 2025 R2 (v252) | escalator full matrix on local license; agrees with tidy3d within **0.081 dB** (worst pair, `compare_smatrices`); recorded artifact replayed every PR |
| beamz | 2026-07-08 | 0.4.3 | straight-waveguide S21 ≡ S12 (reciprocity), S11 −34 dB, passivity — live JAX CPU run; finding F9 (reference-monitor mis-normalization) found and fixed in the process |

Refresh procedure: `cloud-smoke` workflow (tidy3d, budget-gated, human
approval), `lumerical-nightly` workflow (self-hosted lane, see
`docs/self_hosted_runner.md`), `examples/06_beamz` locally. Update this
table in the same PR that lands refreshed `tests/recorded/` artifacts.
