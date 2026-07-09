# Self-hosted Lumerical runner

The `lumerical-nightly` workflow runs the `licensed`-marked test suite on a
lab machine that holds a Lumerical license. The repo works fine without one:
the workflow is guarded by the `LUMERICAL_RUNNER` repository variable and
stays skipped until it is set.

## One-time setup (lab machine, e.g. a SiEPIC/UBC workstation)

1. Install Lumerical FDTD (2024+) and verify `lumapi` imports:
   `python -c "import lumapi"` (add the Lumerical `api/python` dir to
   `PYTHONPATH` system-wide, e.g. in `/etc/environment`).
2. Install [uv](https://docs.astral.sh/uv/) and clone the repo.
3. Register a GitHub Actions runner
   (repo → Settings → Actions → Runners → New self-hosted runner) and give
   it the extra label **`lumerical`**.
4. Run it as a service so it survives reboots:
   `./svc.sh install && ./svc.sh start`.
5. Headless machines: install `xvfb` and prefix the run step with
   `xvfb-run -a` if your Lumerical version needs a display.
6. Set the repository variable `LUMERICAL_RUNNER=true`
   (Settings → Secrets and variables → Actions → Variables).

## What it runs

`pytest -m licensed` — tests that need a real license seat, which the normal
CI matrix deselects. Nightly at 07:41 UTC plus on-demand via
*Run workflow*. Real `.dat`/log artifacts are uploaded so the free replay
tests (`tests/recorded/`) can be refreshed from genuine solver output —
**sanitize before committing** (no license tokens/hostnames; see
`tests/recorded/README.md`).

## Security notes

- Self-hosted runners should only build trusted refs: keep the default
  "runners run on workflows from this repository only" setting, and do NOT
  enable them for pull requests from forks.
- The license never leaves the machine; CI secrets are not needed for this
  lane.
