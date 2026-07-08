# Security Policy

## Supported versions

Only the latest release on PyPI receives security fixes.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting
([Security → Report a vulnerability](https://github.com/SiEPIC/gds_fdtd/security/advisories/new))
rather than a public issue. If that is unavailable, email the maintainer at
mustafa@siepic.com. You should receive a response within a week.

## Scope notes

- `gds_fdtd` executes **no** untrusted code at import time; solver `run()`
  methods contact exactly one external service each (tidy3d cloud) or a
  local licensed binary (Lumerical) — see `SolverCapabilities.execution`.
- Never commit API keys or license tokens. Recorded solver artifacts under
  `tests/recorded/` are sanitized before commit (see `tests/recorded/README.md`);
  CI secret scanning covers that directory.
