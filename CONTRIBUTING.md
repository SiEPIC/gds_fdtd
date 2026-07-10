# Contributing to gds_fdtd

## Dev setup

```bash
git clone https://github.com/SiEPIC/gds_fdtd && cd gds_fdtd
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run prek install          # pre-commit hooks (ruff, codespell, ...)
```

Common tasks live in the `justfile` (`just test`, `just lint`, `just docs`).

## The gate (run before every push)

```bash
uv run pytest tests                        # capture pytest's own exit code
uv run ruff check . && uv run ruff format --check .
uv run codespell src tests README.md
```

CI runs the same checks plus a 9-leg OS/Python matrix; the single `pass`
fan-in job is the required status check.

## Test taxonomy

Markers (see `pyproject.toml`): `physics` (runs a real engine locally),
`gpu`, `cloud` (**spends tidy3d credits**), `licensed` (**needs a Lumerical
license**), `slow`. Default CI deselects all of these. Never mark a test
`cloud`/`licensed` without also documenting its cost in the PR.

Coverage is enforced (`fail_under` in `pyproject.toml`) and only ratchets
up. The single sanctioned exclusion comment is
`# pragma: no cover - requires <license|cloud|gpu>`.

## Solver adapters

New engines implement the `Solver` ABC (`gds_fdtd.solvers.base`) and register
via the `gds_fdtd.solvers` entry-point group. Non-negotiables:

- constructor is cheap and pure (no disk, no network, no license checks);
- `validate()`/`build()`/`estimate()` are offline and deterministic;
- `run()` is the ONLY method that may spend money/licenses/GPU time;
- pass the conformance suite (`tests/conformance/`) — it parametrizes over
  every registered solver automatically.

## Releasing

Releases are tag-driven: `git tag vX.Y.Z && git push --tags`. The release
workflow requires green CI on the tagged commit, builds+inspects, publishes
to PyPI via trusted publishing with attestations, and attaches an SBOM.
There is no version to bump (hatch-vcs derives it from the tag).

## Repo settings checklist (maintainer-applied)

These require admin permissions; PRs cannot change them:

- [x] Branch protection/ruleset on `main`: required checks `pass` + CodeQL,
      linear history, no force pushes
- [ ] Merge queue enabled with `pass` as its gate
- [ ] Secret scanning + push protection enabled
- [ ] CODEOWNERS: `* @mustafacc`
- [ ] Tag protection for `v*`
- [ ] Pages source = "GitHub Actions" (not the main branch)
- [ ] PyPI Trusted Publisher registered (workflow `release.yml`, env `pypi`);
      legacy `PYPI_API_TOKEN` secret deleted
