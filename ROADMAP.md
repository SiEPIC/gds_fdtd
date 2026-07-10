# gds_fdtd roadmap

Living plan for the post-0.5.0 development arc. Any agent or contributor
should be able to read this, understand the current state, and pick up work
without losing context. Keep it current; move granular tracking to GitHub
Issues under the `v0.5.1` milestone as items are picked up.

## Where we are — v0.5.0 (released 2026-07-09)

Solver-agnostic FDTD: one `Component` + one technology file + one
`SimulationSpec`, any engine (tidy3d / Lumerical / beamz) behind
`get_solver(name)(component, tech, spec)`. Three-engine agreement within
0.052 dB on identical jobs (tidy3d↔Lumerical within 0.0033 dB), recorded and
asserted every PR. Honest branch coverage 81% (full package). Docs on GitHub
Pages. OpenSSF Scorecard 7.0.

The modernization arc (Phases 0–7, ~80 commits) is complete and shipped; its
plan has been retired. Deferred items from that arc are captured in the
[Carry-forward backlog](#carry-forward-backlog) below.

## Guiding principles (non-negotiable)

1. **Validate through the exact artifact users run** — example files in a
   fresh interpreter/venv, not bespoke scripts sharing session state (this
   is how the released 0.5.0 was found to break example 09 on a clean
   install; see the de-slop PR #24).
2. **No coverage theatre.** Every test asserts real behavior. A number that
   went up because a `pass` got executed is a regression, not progress.
3. **Only `run()` spends** money / licenses / GPU. Constructors and
   `validate`/`build`/`estimate` stay offline, pure, deterministic.
4. **Deferrals are documented, not silent.** If something can't be validated
   here (no GPU, no container runtime), it goes to the backlog with the
   verified facts a future executor needs — never a blind implementation.
5. **Every change is a PR into `main`.** No direct pushes. This is both good
   hygiene and the path to the Scorecard Code-Review signal.

## v0.5.1 — the polish, hardening & trust arc

Five workstreams. They are independent; parallelize freely.

### WS1 — Examples become executed, verified notebooks

The 18 example scripts are the front door. Turn them into notebooks with
real, committed outputs (plots inline), executable end-to-end, and kept
honest by CI.

- **Author format:** keep the `.py` scripts as source of truth and pair them
  to notebooks with [jupytext](https://jupytext.readthedocs.io/) (percent
  format), so notebooks are diff-friendly and never drift from the scripts.
- **Execution tiers:**
  - *offline* (01b, 04c, 09, 10): execute in CI on every PR with
    `jupyter execute` / nbmake; outputs must match (nbval-style tolerance for
    floats/plots).
  - *engine* (tidy3d/Lumerical/beamz runs): execute **locally** with the
    real engine, commit the executed notebook with outputs as an artifact of
    record; CI runs them only up to `build()` (offline) and validates the
    notebook *structure*/imports.
- **Gallery:** render executed notebooks into the Sphinx docs
  (`nbsphinx` or `myst-nb`) so the website shows real field profiles and
  S-parameters, not just code.
- **Cleanup:** the stray `examples/notebooks/faid/` notebook predates the
  standardized flow and once carried a base64 license token in its logs —
  sanitize and fold it into the standard set, or remove it.
- *Done when:* every example has a committed executed notebook; the offline
  set runs green in a dedicated CI job; the docs gallery shows real outputs.

### WS2 — Real >90% coverage

Current: 81% full-package. The gap is concentrated in the engine adapters
(`solvers/beamz` 16%, `solvers/tidy3d` 45%, `layout/gdsfactory` 12% —
low only because those extras aren't in the base test env) and in the legacy
`solver_*` internals, `cli`, `caching`, `modes`.

- **Measure the right thing:** the >90% bar is enforced on the **all-extras**
  CI leg (engines installed), not the base profile. Raise
  `--cov-fail-under` there step by step as real tests land; ratchet only up.
- **Property-based tests** (hypothesis) for the numeric core — `SMatrix`
  (reciprocity/passivity invariants, round-trip through every I/O format),
  `geometry` (dilate/extension math), `technology` (v1↔v2 equivalence),
  `grid`/`extraction` (energy/overlap identities). These double as the
  fuzzing signal (see WS5).
- **beamz** is free and CPU-runnable → real (small, `@pytest.mark.slow`)
  end-to-end tests in the all-extras leg, not just mocks.
- **gdsfactory / mode solver** run offline locally → real conversion and
  n_eff tests.
- *Done when:* all-extras leg ≥ 90% branch coverage, every new test asserts
  behavior, and the base-profile floor also rises where honest.

### WS3 — De-slop & refactor for robustness

The released 0.5.0 is clean at the surface; the depth to attack is the
**legacy `solver_*` layer** the modern adapters wrap.

- **Flatten the legacy wrap:** `solvers/tidy3d.py` and `solvers/lumerical.py`
  delegate into 400–600-line `solver_tidy3d.py` / `solver_lumerical.py`.
  Fold the still-used logic into the adapters and delete the wrapper modules
  (they are the last home of the deprecated class API — schedule with the
  v1.0 shim removal, but the internal cleanup can start now behind the
  stable adapter surface).
- **Errors everywhere:** audit every `raise` in user-input paths → the
  `GdsFdtdError` hierarchy; no bare `except:`.
- **Types:** extend `mypy --strict` module-by-module beyond the current
  typed core until the whole package is strict-clean; then make it a hard
  gate.
- **Consistency pass:** logging (no residual prints), docstring style,
  public API surface (`__all__`), and a documented deprecation policy.

### WS4 — Workflow, tooling & release modernization

- **Signed releases** (also WS5): sign wheels+sdist with Sigstore in
  `release.yml`, attach `.sigstore` bundles to the GitHub Release.
- **Fuzzing** (also WS5): ClusterFuzzLite + atheris targets on the parsers.
- **Coverage reporting:** wire codecov `project`/`patch` gates so PRs that
  drop coverage fail; surface the number on the PR.
- **Dependency freshness canary:** alongside the weekly lowest-floors job,
  add an allowed-failure "latest unpinned" job so upstream breakage is seen
  early.
- **Notebook CI** (WS1) and **docs link-check** jobs.
- **Merge queue** once branch protection lands (WS5).

### WS5 — OpenSSF Scorecard: 7.0 → as high as a solo-maintained repo can go

Current failing/low checks and the concrete action for each. Weights:
CRITICAL/HIGH move the needle most.

| Check | Now | Weight | Action | Who |
|---|---|---|---|---|
| **Branch-Protection** | 0 | HIGH | Protect `main`: require PR + passing `pass`/CodeQL checks, up-to-date branches, ≥1 review, dismiss stale approvals, linear history, block force-push, enforce for admins, require conversation resolution. | owner (admin API/settings) |
| **Code-Review** | 0 | HIGH | Route *all* changes through PRs (principle 5) and get an approving review before merge. Solo dev is the ceiling here — a second reviewer/maintainer or a review bot is the only way to fully satisfy it. | owner + process |
| **Signed-Releases** | 0 | HIGH | Sigstore-sign release artifacts in `release.yml`; attach signatures to the GitHub Release. (PyPI already gets PEP 740 attestations via trusted publishing.) | executor (workflow) |
| **Fuzzing** | 0 | MEDIUM | ClusterFuzzLite with atheris fuzz targets on `technology` YAML, `.dat`/Touchstone/`SMatrix` parsers, port-id parsing. | executor |
| **Packaging** | ? | MEDIUM | Resolves to 10 once the package publishes to PyPI via the recognized workflow — pending Lukas registering the trusted publisher. | pending PyPI |
| **CII-Best-Practices** | 0 | LOW | Register at bestpractices.dev and complete the passing questionnaire (we already meet most: VCS, tests, CI, docs, license, CONTRIBUTING, SECURITY, no known vulns). Executor can pre-fill answers. | owner registers |

Everything else is already 10 (Dangerous-Workflow, Maintained,
Dependency-Update-Tool, Binary-Artifacts, Token-Permissions, Vulnerabilities,
Security-Policy, Pinned-Dependencies, SAST, License, Contributors, CI-Tests).

*Honest target:* Branch-Protection + Signed-Releases + Fuzzing + Packaging
should lift the score to ~8.5–9.0. A perfect 10 is unrealistic for a
single-maintainer project (Code-Review and full Fuzzing credit are the
structural ceilings) — we max what's real and don't game the rest.

## Feature ideas (menu — pick as inspiration strikes)

Not committed; a palette to choose from. Roughly ordered by leverage.

- **fdtdz adapter** — finishes the free-GPU story; the whole
  rasterize→modes→extraction pipeline is already built and tested (blocked
  only on GPU hardware; see backlog).
- **Sweeps & optimization** — parameter sweeps over geometry/spec producing
  a tidy results table (pandas/xarray), and a hook for inverse-design loops.
- **S-parameter post-processing** — group delay, dispersion, insertion
  loss / crosstalk summaries, `scikit-rf` `Network` interop both directions.
- **Component library / PDK bridge** — a small set of parametric reference
  devices (crossing, DC, MMI, ring) with known-good S-params as fixtures.
- **Results caching backend** — content-addressed store beyond the local
  npz cache (e.g. an S3/GCS-backed cache for cluster sweeps).
- **Interactive report** — an HTML/notebook report per run (geometry,
  convergence, S-params, fields) as a single shareable artifact.
- **Richer technology** — anisotropic/dispersive material helpers,
  gdsfactory `LayerStack` / KLayout `.lyt` import into technology v2.
- **Multi-frequency / broadband mode tracking**, bend-mode solving, PML
  convergence diagnostics.

## Carry-forward backlog (from the modernization arc)

Deferred with documented rationale; resume when the blocker clears.

- **fdtdz adapter (D9)** — needs an NVIDIA GPU + CUDA (fdtdz ships a
  CUDA-building sdist; won't even import without it). Verified kernel API and
  constraints recorded in git history. Natural to pair with a GPU CI lane.
- **femwell mode-solver backend (D8)** — second `ModeSolver` backend; the
  tidy3d local plugin already covers Tier-B needs at zero extra deps.
- **MEEP adapter** — deprioritized by owner; beamz fills the free-engine
  role.
- **Container images (ghcr) + conda-forge feedstock** — no container runtime
  on the dev machine to validate images; feedstock is owner-level.
- **tenacity retries on cloud calls** — modifying validated money-spending
  paths deserves its own live-revalidation session; make task submission
  idempotent (`task_name = gdsfdtd-{job_hash[:12]}`) first.
- **CITATION.cff** + Zenodo DOI for academic citation.
- **v1.0** — remove the deprecated `core`/`solver_*` shims, freeze the public
  API, finalize the deprecation policy.

## Owner-only actions (need admin / external accounts)

Tracked here so they don't get lost; none block executor work.

- [ ] **Branch protection on `main`** (WS5) — the single biggest Scorecard
      lever.
- [ ] **PyPI trusted publisher** (project `gds_fdtd`, owner `SiEPIC`,
      workflow `release.yml`, env `pypi`) — in progress with Lukas; then
      re-run the release build's publish job. Unlocks Packaging (WS5).
- [ ] **OpenSSF Best Practices badge** — register at bestpractices.dev.
- [ ] **`cloud-tests` environment** with a required reviewer (guards the
      budget-gated tidy3d smoke).
- [ ] **`LUMERICAL_RUNNER` repo variable** when a lab self-hosted runner
      exists (enables the nightly licensed lane).
- [ ] A **second reviewer/maintainer** would meaningfully raise the
      Code-Review score.

## Tracking

- This file = the durable plan.
- Granular work = GitHub Issues under a **`v0.5.1`** milestone, one issue per
  task above, linked from the PR that closes it.
- Every PR targets `main`, passes the full matrix, and is squash- or
  merge-committed only after review.
