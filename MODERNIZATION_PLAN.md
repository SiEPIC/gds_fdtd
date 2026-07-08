# gds_fdtd — Deep Audit & Modernization Playbook (v2)

> **Audience:** the AI developer (assume a modest model) or human contributor executing the
> modernization. This document is the single source of truth for the effort. It contains the
> project's identity and history, a line-referenced audit, verified 2026 ecosystem facts, the
> target architecture, and an ordered set of **work packages (WPs), each written as a
> ready-to-paste prompt** with acceptance criteria and verification commands.
>
> **Prepared:** 2026-07-07 against `main` @ `ab2cd9b` (v0.4.0, 212 commits since 2023-01-13).
> Ecosystem versions verified online at that date: tidy3d **2.11.2**, gdsfactory **9.44.0**,
> beamz **0.4.3**, fdtdz (PyPI, low-level kernel), fdtdx (JOSS-published), MEEP (conda-forge).

---

## Part −1 — EXECUTION LOG & HANDOFF STATE (update after every WP — mandatory)

> Any agent resuming this effort: read this section first, then Part 0. Update this section
> in the same commit as (or immediately after) each WP so an interruption at any point leaves
> a resumable state.

**Standing user directives (override anything below):**
- Do NOT add `Co-Authored-By` trailers (or any AI attribution) to commits.
- Keep this execution log current with progress, findings, and plan deviations.
- All modernization work happens on the **`modernization` branch** — never commit to `main`
  (`main` sits at origin/main `ab2cd9b`).

**Executor environment:** dedicated venv at `.venv/` (Python 3.13.5, gitignored) created to
avoid touching the user's `gdsfactory` conda env; `uv` is installed *inside* it
(`.venv/bin/uv`). Gate: `.venv/bin/ruff check . && .venv/bin/python -m pytest -q tests`.
Commits are local only — nothing pushed without the user's say-so.

**WP status:**
| WP | Status | Commit | Notes |
|---|---|---|---|
| WP0.1 | ✅ done | `3fe9984` | see deviations D1–D4 |
| WP0.2 | ✅ done | (next commit) | ci.yml (SHA-pinned, uv, alls-green `pass` gate); fake-coverage test + commented blocks deleted; README fence/badges fixed; **honest coverage measured: 16%** (core 90%, lyprocessor 21%, sparams 11%, simprocessor 6%, solvers+logging 0%) — this is the WP7.1 `fail_under` baseline |
| WP0.5 | ✅ done (code) / ⏸ cleanup+settings pending owner | (next commit) | jekyll + python-publish workflows deleted; build_docs.yml → official artifact Pages flow. **NEW FINDING (F1):** Pages source is the `main` BRANCH — the live site is a *Jekyll render of the README*; the Sphinx docs pushed to `gh-pages` were never served at all. **Owner actions at merge: (1) flip Pages source to "GitHub Actions" (`gh api -X PUT repos/SiEPIC/gds_fdtd/pages -f build_type=workflow`), (2) delete `gh-pages` branch, (3) approve the deployment-deletion pass** (list-only pass done: 80 rows = 75×main + 5×tags; newest id 2827934461). |
| WP0.6, WP0.3, WP0.4 | pending | — | in that intended order |
| Phase 1+ | not started | — | |

**WP0.2 execution notes for successors:** `dev` is an *extra*, not a PEP 735 group — CI must
`uv sync --locked --extra dev` (plain `uv sync` gives no pytest). The old
`build_and_test.yml` is deleted (replaced by ci.yml). Owner must later flip the required
status check to `pass` in branch protection (WP7.2 checklist).

**Deviations from the plan as written (accepted, keep):**
- **D1:** instead of `# noqa` spam, temporary rule-level ruff ignores live in
  `pyproject.toml [tool.ruff.lint] ignore` with comments mapping each to its owning WP
  (E501/B006/E721/E722/B904/F841/UP031). Remove each ignore inside its owning WP.
- **D2:** `[tool.mypy] python_version` pin removed — numpy 2.x type stubs use `type`
  statements (3.12+ syntax) and break mypy pinned to 3.11.
- **D3:** `examples/` is ruff-`extend-exclude`d until the WP6.1 rewrite.
- **D4:** mypy currently reports **71 errors** on the legacy code → the CI `typecheck` job
  ships as **advisory** (`continue-on-error: true`, in alls-green `allowed-failures`), to be
  made required during Phase 2. (Same deadlock class the adversarial review fixed for docs `-W`.)
- **D5:** WP2.2 gained item 2b — refractiveindex.info material support (`rii:` key), owner
  request 2026-07-07. Additive v1-schema change; offline-first with committed page fixture.

**Verified action SHAs (resolved via `gh api` 2026-07-07; Dependabot maintains after merge):**
```
actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd            # v5
astral-sh/setup-uv@37802adc94f370d6bfd71619e3f0bf239e1f3b78          # v7
codecov/codecov-action@0fb7174895f61a3b6b78fc075e0cd60383518dac      # v5
re-actors/alls-green@05ac9388f0aebcb5727afa17fcccfecd6f8ec5fe        # release/v1
j178/prek-action@0bb87d7f00b0c99306c8bcb8b8beba1eb581c037            # v1
hynek/build-and-inspect-python-package@d44ca7d91762de7a7d5436ddae667c6da6d1c3df  # v2
```

**Environment facts discovered:** user's active conda env `gdsfactory` has gdsfactory 9.14.0,
tidy3d 2.8.5, klayout 0.30.3, pydantic 2.11.7, numpy 2.2.0, python 3.13.5 — useful later for
WP4.x verification, but not the executor env. Baseline test suite: 51 passed in ~2 s.

---

## Part 0 — Rules of engagement for the AI developer

These rules override any instinct to "be helpful" by doing more. Violating them is failure.

1. **One work package per session/PR.** Never combine WPs. Never start a WP before its listed
   prerequisites are merged.
2. **Run the gate before and after every WP:**
   ```bash
   pip install -e .[dev] && ruff check . && pytest -q tests
   ```
   If the gate fails *before* you start, stop and report — do not fix unrelated breakage inside
   your WP. **Exception, WP0.1 only:** the pre-gate omits `ruff check .` (there is no ruff
   config yet and the default rules report ~100 errors); WP0.1 itself must end with
   `ruff check .` green via config scoping + `--fix` + `# noqa`.
3. **Behavior-preserving = golden fixtures and the full test suite pass unchanged** (except
   tests the card explicitly names). WPs marked `[REFACTOR]` must not change any public
   behavior; WPs marked `[FIX]` change exactly the behavior named in the WP; WPs marked
   `[FEATURE]` add new surface only.
4. **Never delete or rewrite a test to make it pass** — unless the WP card explicitly names the
   test expectations to change. If any *other* test seems wrong, report it in the PR
   description and skip with `@pytest.mark.xfail(reason=...)` — a human decides.
5. **No drive-by changes.** No renaming, reformatting, comment editing, or "cleanup" outside the
   files listed in the WP. Ruff-format only the files you touched.
6. **No new dependencies** unless the WP explicitly lists them.
7. **Commit message format:** `WP<id>: <imperative summary>` (e.g., `WP1.3: fix dilate_1d sign
   multiplication bug`). One WP may have several commits; all share the WP id.
8. **When a WP says "verify against installed X"** — actually run Python and inspect the object
   (`python -c "import tidy3d; print(tidy3d.__version__)"`); never trust memory of an API.
9. **If blocked** (missing license, unclear spec, failing prerequisite), write your findings in
   the PR/issue and stop. A wrong guess costs more than a question.
10. **Solver code must be testable offline.** Any code path that talks to a cloud service or
    licensed binary must be reachable only from `run()`-family methods, never from constructors
    or `setup()`/`build()` paths, so that CI can exercise everything else.

**Prompting strategy for the human orchestrator:** feed the model *one WP card verbatim* plus
Part 0, plus the "Architecture contract" (Part 4) when the WP touches interfaces. Do not feed the
whole document per task. After each WP, have a second session review the diff against the WP's
acceptance criteria before merging (reviewer prompt template at the end of Part 5).

---

## Part 1 — What this project is, and what it should become

**gds_fdtd** converts a photonic chip layout (GDS) plus a small **technology YAML** (layer stack:
z-heights, materials, sidewall angles, pin/devrec conventions) into ready-to-run 3D FDTD
simulations, runs them on a chosen solver, and returns standardized S-parameters.

Pipeline: `GDS + technology.yaml → Component (polygons+ports+bounds) → solver adapter → run →
S-matrix → export (.dat / plots)`.

**The three ideas worth preserving at all costs:**
1. The **technology YAML** as a solver-neutral layer-stack contract (materials carry per-solver
   entries: `tidy3d_db`, `lum_db`; the v1 key names stay frozen through 1.x — generalizing them
   into neutral `solver_hints` is a schema-v2 decision, Part 8 Q13).
2. **Port auto-discovery** from SiEPIC-style pin paths (path on pinrec layer + text label), so
   sources/monitors are never hand-placed.
3. **One constructor for every solver** — switch engines by switching an import.

**Vision.** Become the *vendor-neutral layout-to-S-matrix layer for photonics*. Nobody owns this
today: gdsfactory's `gplugins`/`gsim` are gdsfactory-front-end-only; SiEPIC-Tools is
Lumerical-only; each solver's own helpers are captive. gds_fdtd's defensible niche is
**EDA-agnostic ingestion (KLayout/SiEPIC/gdsfactory) × solver-agnostic execution
(Tidy3D/Lumerical/MEEP/beamz/fdtdz/fdtdx/…)** with the technology file as the neutral middle.
A 2025 arXiv paper (2506.16665) literally benchmarks Lumerical vs Tidy3D on passive silicon
components — cross-solver validation is a real community need this tool can automate.

**Why the new JAX solvers change the architecture.** beamz / fdtdz / fdtdx are GPU-first,
open-source, and orders of magnitude cheaper per run than cloud/licensed solvers. But they demand
different things from us than Tidy3D/Lumerical do (see capability matrix, Part 3). Supporting
them properly — not as bolted-on hacks — is the core architectural driver of this plan: the
intermediate representation must serve both **polygon-consuming** solvers and **grid/kernel**
solvers, and gds_fdtd must own the surrounding machinery (rasterization, mode solving, S-parameter
normalization) that kernel solvers deliberately omit.

---

## Part 2 — Repository archaeology (what history teaches)

212 commits, 13 tags (v0.1.0 → v0.4.0), work in bursts (Dec 2023: 38 commits, May 2025: 45,
Jul–Aug 2025: 32, then dormant since Sep 2025). Four identities:

| Era | Name | Dates | What happened |
|-----|------|-------|----------------|
| 1 | `klayout_tidy3d.py` single script | Jan 2023 | First DC example; KLayout→Tidy3D proof of concept |
| 2 | `siepic_tidy3d` (+ a parallel `siepic_tidy3d_v2/`!) | 2023 | Package-ified; Nov 2023 restructure into core/lyprocessor/simprocessor/sparams modules that survive today |
| 3 | `gds_tidy3d` | Sep 2024 | flit + pyproject migration, gdsfactory made optional, docs |
| 4 | `gds_fdtd` | May 2025 | Lumerical added (`lum_tools.py`), tidy3d code split to `t3d_tools.py`, SiEPIC/tidy3d/prefab all made optional extras |
| 4.5 | solver-class rewrite | Jul–Aug 2025 | `75edfd0 "major refactor towards unified solver"`: `fdtd_solver` base + `solver_lumerical` + `solver_tidy3d`; `lum_tools.py` deleted (Jul 23), `t3d_tools.py` deleted (Aug 4) — **but examples 01/05/07/08 still import them**, and `tests/test_{solver,sparams,simprocessor}.py` were deleted without replacement |

**Lessons that shape this plan:**
- **Every past rename/refactor was abandoned mid-flight** (a `siepic_tidy3d_v2/` directory once
  coexisted with `siepic_tidy3d/`; today's examples import deleted modules). Countermeasure:
  WPs are small, ordered, and each leaves the repo green — plus an examples-import CI check
  (WP0.4) so stale examples can never merge again.
- **Solver additions historically forked the codebase** (`lum_tools` vs `t3d_tools`) instead of
  extending an interface. The Jul-2025 solver base class was the right correction — this plan
  finishes it.
- **Coverage numbers were gamed, not earned:** `tests/test_lyprocessor.py:196`
  (`test_force_full_coverage`) exec-compiles a `pass` at every line number of the module so
  codecov reports ~100% while the real tests in that file are mostly commented out (lines
  151–192) and the whole `pya` module is stubbed. The codecov badge is fiction for this module.
- **CHANGELOG 0.4.0 describes features that don't exist** (Touchstone/JSON export, energy
  conservation and reciprocity checks, group-delay tools). Treat the changelog as aspiration,
  not record; those aspirations are folded into real WPs below (WP2.4, WP6.2).

---

## Part 3 — Audit findings

### 3.1 Confirmed bugs (each becomes a WP1.x fix; file:line against `ab2cd9b`)

| ID | Location | Defect | Consequence |
|----|----------|--------|-------------|
| B1 | `lyprocessor.py:49,58` | `[x, y-abs(e)] * sign` multiplies the *list* by sign; sign=−1 → `[]` | `dilate_1d("y"/"xy")` silently returns corrupt geometry |
| B2 | `simprocessor.py:196` | `from_gdsfactory` appends `l.get_polygons()[1]` — hardcoded `1`, loop var `s` unused | every structure is the same second polygon |
| B3 | `simprocessor.py:213` | port `name=c.name` (component name) | all ports share a name; `port.idx` garbage |
| B4 | `simprocessor.py:162-265` | written for pre-gf-8 API (`get_polygons()` list-of-arrays in um; `p.center`/`p.width` floats) | incompatible with pinned gf ≥9.5.7 (see §3.3 rules) |
| B5 | `solver.py:196,240` | default `port_input=[None]` is truthy; fallback unreachable; `_get_active_ports` raises on `None` | default solver construction crashes |
| B6 | `core.py:186` | `port.idx` = digits of name **reversed** (`"opt12"`→21, `"opt10"`→1≡`opt1`) | scrambled S-labels for ≥10-port or multi-digit-named devices; same ambiguity in `sparams.py:52-56` (`idn` concatenates digits: S`21` = (2,1) or port 21?) |
| B7 | `simprocessor.py:36-39` | `lum_db` without `model` key → `UnboundLocalError: mat_lum` | crash on valid-looking YAML |
| B8 | `lyprocessor.py:191-199` | devrec shape neither box nor polygon → `polygon` undefined (`NameError`); only first devrec shape used | crash / wrong region |
| B9 | `solver_tidy3d.py:261-267,297-303,352` | field monitors centered at `[0,0,z]` regardless of component position; substrate/superstrate squares origin-centered (sized `abs(center)+span/2`, so still covering but O(\|center\|) oversized) | monitors miss off-origin components entirely; substrate wastefully oversized |
| B10 | `solver_tidy3d.py:193` | `boundary` argument ignored — PML hardcoded all sides (Lumerical solver honors it) | solver parity broken |
| B11 | `solver_tidy3d.py:190,227` | `run_time` ignores `_calculate_simulation_time` (no group-index factor; Lumerical uses it); `mode_indices` computed then unused | same `run_time_factor` ⇒ different physics per solver |
| B12 | `solver_tidy3d.py:410-441` | `_set_field_data_in_monitors` probes `component_modeler.batch_data` internals that don't exist in this form | field visualization dead code; warnings at runtime |
| B13 | `sparams.py:84`, `core.py:543,564` | `np.angle(phase)**2` (meaningless, and discarded); `log10(complex**2)` | wrong/warning-spewing plots |
| B14 | `solver_tidy3d.py:638-674` + `sparams.py:566-577` | `.dat` export assumes strict n²(out,in) ordering that `_sparameters.data` doesn't guarantee | exported `.dat` mislabels ports whenever ordering differs / matrix partial |
| B15 | `lyprocessor.py:81-127` | `load_device` returns `None` (builds component, drops it), mutates the input-adjacent GDS, and example 07 calls `load_cell(..., prefab=...)` — a kwarg that doesn't exist | broken prefab path |
| B16 | `logging_config.py:33-38` | wipes **root logger** handlers, sets root to DEBUG on every solver construction; `sparams.py:15` logger literally named `"dreamcompiler"` | hijacks host app logging; library anti-pattern |
| B17 | `solver.py:19-62` etc. | mutable default args everywhere (`[None]`, `["PML",...]`, `[0]`, `[1,0]`) | shared-state corruption across instances |
| B18 | `tests/test_lyprocessor.py:196-201` | coverage-gaming `exec(compile("\n"*(ln-1)+"pass", filename, "exec"))` | fake coverage; must be deleted, real tests written |
| B19 | `core.py:389,398` / `solver_tidy3d.py:246` | `component.structures` is a mixed `list[structure | list[structure]]` discriminated by `type(s)==list`; substrate identified by name-sniffing incl. the typo `"subtrate"` | fragile; blocks any new solver from consuming structures safely |

### 3.2 Design-level issues (fixed by Part 4/5 architecture, not spot fixes)

- Three coexisting S-parameter representations: `core.sparam/s_parameters` (legacy),
  `sparams.s/sparameters` (current), `sparams.s_parameter_writer` (write-only). One canonical
  type needed.
- PEP8: lowercase class names (`port`, `component`, `technology`, `s`) that get shadowed by
  variables (`solver.py:124` shadows the imported class `component` with a loop variable);
  invalid generics (`list[float, float]`).
- `technology` is an unvalidated dict-in-class; `to_dict()` round-trips everywhere.
- Lumerical adapter is f-string `fdtd.eval(...)` scripting (injection-prone, untestable),
  `while True: try/except` loops, 2024-only GPU syntax (own TODOs at lines 73, 88, 357).
- print()+logger duplication everywhere; 50-line print summaries.
- `mode_freq_pts` unused by tidy3d path; `visualize` flag gates a no-op; placeholder plot
  methods ("Implementation needed") ship in the base class.
- Speed of light redefined in ≥5 places, mixed units.
- Solver constructors take ~20 kwargs; construction also *performs* setup (tidy3d `__init__`
  calls `setup()` which writes GDS to disk) — constructors must be cheap and pure.

### 3.3 Verified 2026 ecosystem facts (do NOT re-derive from memory)

**tidy3d 2.11.2** (pinned today at `>=2.8.3,<2.9` — two majors behind):
- `ComponentModeler` → **`ModalComponentModeler`**; modelers are immutable pydantic objects.
- `verbose`, `path_dir`, `folder_name`, `callback_url`, `batch_cached` **removed from the
  modeler**; passed to the web API instead.
- New run pattern (verify exact signatures against installed 2.11 before coding):
  ```python
  from tidy3d.plugins.smatrix import ModalComponentModeler
  modeler = ModalComponentModeler(simulation=sim, ports=ports, freqs=freqs)
  modeler_data = td.web.run(modeler, task_name="...", verbose=True)   # ModalComponentModelerData
  smat = modeler_data.smatrix()   # xarray DataArray: f, port_in, port_out, mode_index_in/out
  ```
- Migration guide is in the official changelog/docs — read it during WP4.1.

**gdsfactory 9.44** (pin `>=9.5.7,<10` is compatible, code isn't):
- v9 unified units: **`port.center`, `port.width`, `port.x/y` are um floats; `port.orientation`
  is degrees; DBU access is `port.ix/iy`** (the v8 `d`-prefix — `dcenter`, `dwidth` — was
  *removed* in v9; code written for v7 or v8 conventions is wrong in different ways).
- `Component.get_polygons()` returns a dict keyed by layer (DBU polygons);
  **`Component.get_polygons_points()`** returns um point arrays; both accept `by="tuple"` to key
  results by `(layer, datatype)`.
- `Port.layer` is now a layer *index*; use `port.layer_info` / gf helpers for the
  `(layer, datatype)` tuple (verify on installed version).
- gdsfactory has its own `LayerStack` model and `gplugins`/`gsim` — interop target, not template.

**The new solver landscape** (the reason this refactor exists):

| Solver | What it is | Geometry input | Mode solving | S-params | Runs where | License |
|--------|-----------|----------------|--------------|----------|-----------|---------|
| **Tidy3D 2.11** | commercial cloud FDTD | polygons (`PolySlab`, sidewalls) | built-in (`ModeSpec`) | native (`ModalComponentModeler`) | cloud, $ | proprietary client Apache |
| **Lumerical FDTD** | commercial local FDTD | GDS via layer-builder | built-in ports | native sweep + `.dat` | local, license | proprietary |
| **MEEP** | classic open FDTD (C++/Python) | polygons/prisms | MPB integrated; eigenmode source & coefficients | via `get_eigenmode_coefficients` | local CPU (MPI) | GPL. conda-forge only |
| **beamz 0.4.3** (beamzorg/beamz) | JAX FDTD, high-level, "pragmatic engine for chip designers" | **native GDSII import**, parametric shapes | mode sources TE/TM | **built-in S-param workflow** (DFT monitors) | local CPU/GPU | Apache-2.0, pip |
| **fdtdz** (spinsphotonics) | ultra-fast low-level GPU kernel, "does one thing" | **raw permittivity grid** (rasterized), limited z-extent, non-dispersive only | **none — bring your own** | **none — bring your own** (FFT fields yourself) | local GPU (JAX) | open |
| **fdtdx** (ymahlau, JOSS 2025) | JAX FDTD framework, large-scale, autodiff/inverse design | objects/voxel grid | partial — check current API | via detectors, manual | local GPU/multi-GPU | MIT |

**Architectural consequence (the key insight):** solvers split into
- **Tier A — "full-service"** (Tidy3D, Lumerical, MEEP, beamz): accept polygons + layer stack,
  have mode sources, return mode amplitudes. Adapter = translate `Component`+`Technology`+
  `SimulationSpec` into their native scene.
- **Tier B — "kernel"** (fdtdz, largely fdtdx): accept a permittivity grid + raw current
  sources; return raw fields. To support them, **gds_fdtd itself must provide**: (1) polygon →
  permittivity-grid **rasterization** (with sub-pixel averaging), (2) a **local mode solver** to
  synthesize injection profiles and decompose outputs, (3) overlap-integral **S-parameter
  extraction**. These become first-class core modules (WP5.2), which *also* de-risks every Tier A
  adapter (shared postprocessing = cross-solver consistency checks).

Mode-solver options for Tier B (evaluate in WP5.2): tidy3d's local mode solver plugin (free,
runs locally, but drags the tidy3d dependency), MPB (conda), `femwell` (FEM, open, scikit-fem
based), or EMpy. Recommendation: make it a small internal interface (`ModeSolver` protocol) with
a femwell-backed default and tidy3d-backed alternative.

### 3.4 Tooling/pipeline drift

- Build: flit works, but the 2026 standard is **uv + committed `uv.lock`** for workflow/CI and
  **hatchling + hatch-vcs** for build/versioning (version derived from git tags — kills the
  3-file version sync). `bump2version` is officially unmaintained (its repo points to
  `bump-my-version`), and with hatch-vcs no bump tool is needed at all.
- Lint: ruff listed in dev deps but **no configuration**; dead `[tool.flake8]` block
  (flake8 does not read pyproject). No mypy. No pre-commit/prek config.
- CI: single job, py3.10 only, coverage uploaded from partly-gamed tests. `requires-python
  >=3.10,<3.14` while the ecosystem is on 3.12–3.14.
- README: broken unclosed code fence at line ~80 swallows the whole Development section.
- Repo-local clutter (untracked, 2.6 GB): `examples/**/*.hdf5` (1.5 GB + 1 GB + 113 MB),
  `mode_solver.hdf5`, `.coverage`, `docs/_build/`; `docs/_autosummary/` **is tracked** but
  generated.
- Missing entirely: PR template, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md,
  CITATION.cff, CODEOWNERS, `.github/release.yml`, AGENTS.md.

### 3.5 Workflow diagnosis (verified against the live repo, 2026-07)

Five workflows, **two duplicate/conflicting pairs**:

| Workflow | Trigger | Problems |
|---|---|---|
| `build_and_test.yml` | push `'**'` AND `pull_request` `'**'` | every PR runs twice; py3.10 only; no `permissions`, no `concurrency`, no caching |
| `build_docs.yml` | push main | peaceiris pushes built HTML to `gh-pages` **branch** → GitHub's *legacy* branch-based Pages then runs its own "pages build and deployment" → **one extra `github-pages` deployment row per main push**; `force_orphan` rewrites history each time; broad `contents: write` |
| `jekyll-gh-pages.yml` | release published | stock **Jekyll sample misapplied to a Sphinx repo** — builds the repo root as a Jekyll site and deploys it over the docs via `actions/deploy-pages` (the tag-ref deployments v0.3.7…v0.4.0) |
| `python-publish.yml` | release published | publishes to PyPI with legacy `PYPI_API_TOKEN` (`user: __token__`) |
| `release.yml` | tag `v*` | re-runs tests + rebuilds docs at release time instead of gating on CI; archived `actions/create-release`; **also publishes to PyPI** → double-publish race with `python-publish.yml` (second upload fails "file already exists"); `setup-python@v4` |

**The "80 deployments" mystery, solved:** `gh api repos/SiEPIC/gds_fdtd/deployments` returns
exactly 80, all in the `github-pages` environment. Each main push ≈ 2 deployment rows (peaceiris
branch push + legacy Pages build), plus one per release from the Jekyll workflow. Fix in WP0.5;
one-time cleanup: loop the REST API (`gh api`) marking deployments inactive then DELETE — the UI
has no bulk delete.

Versioning today: hand-synced across `pyproject.toml`, `gds_fdtd/__init__.py`, `docs/conf.py`
by dead `bump2version` driven by a 100-line `scripts/release.sh`. Replaced wholesale in WP0.6.

---

## Part 4 — Architecture contract (feed this to every interface-touching WP)

```
src/gds_fdtd/
├── __init__.py            # public API re-exports; __version__ from importlib.metadata
├── constants.py           # C_UM_S, C_M_S — the ONLY definitions of c
├── geometry.py             # Port, Structure, Region, Component (frozen-ish pydantic v2)
├── technology.py            # Technology, DeviceLayer, MaterialSpec (pydantic v2, YAML I/O,
│                            #   json-schema export; materials = neutral spec + per-solver hints)
├── layout/
│   ├── klayout.py           # today's lyprocessor, fixed (KLayout/SiEPIC ingestion)
│   ├── gdsfactory.py         # gf ≥9 converter (lazy import)
│   └── siepic.py             # SiEPIC PDK conveniences (lazy import)
├── smatrix.py               # SMatrix: complex ndarray [f, port_out, port_in, mode_out, mode_in]
│                            #   + port names; I/O: lumerical .dat (r/w), touchstone .snp (w),
│                            #   hdf5/npz (r/w); checks: reciprocity, passivity, unitarity;
│                            #   plotting kept in a separate plotting.py (matplotlib optional)
├── spec.py                  # SimulationSpec (pydantic): wavelengths, mesh, boundaries,
│                            #   symmetry, modes, runtime policy, z-domain, port geometry
├── solvers/
│   ├── base.py              # Solver ABC + SolverCapabilities + registry (entry_points
│   │                        #   group "gds_fdtd.solvers") + SetupArtifacts
│   ├── tidy3d.py            # Tier A (cloud)
│   ├── lumerical.py          # Tier A (licensed local)
│   ├── meep.py               # Tier A (open local)  ← CI's end-to-end engine
│   ├── beamz.py              # Tier A (open GPU/CPU)
│   └── fdtdz.py              # Tier B (kernel; uses grid.py + modes.py + extraction in core)
├── grid.py                   # polygon stack → permittivity voxel grid (sub-pixel averaging)
├── modes.py                   # ModeSolver protocol + femwell/tidy3d backends
├── convergence.py             # sweep any SimulationSpec field across any solver, report
├── execution/
│   ├── jobspec.py             # JobSpec = {component, technology, spec, solver, options},
│   │                          #   100% JSON-round-trippable; JobResult = SMatrix path + meta
│   └── backends.py            # ExecutionBackend protocol; LocalBackend, SubprocessBackend
│                              #   ship at 1.0; Modal/AWS Batch/SLURM/Ray = thin post-1.0 adapters
├── cli.py                     # `gds-fdtd` console script: validate | build | estimate | run |
│                              #   convert | solvers  (headless = deployable anywhere)
└── errors.py                  # GdsFdtdError → TechnologyError, LayoutError, SolverError,
                               #   SolverUnavailableError, BudgetExceededError, ...
```

**The Solver contract** (exact; WP3.1 implements):

```python
class SolverCapabilities(BaseModel):
    tier: Literal["full", "kernel"]
    execution: Literal["local", "cloud"]
    supports_dispersion: bool
    supports_sidewall_angle: bool
    supports_multimode: bool
    supports_gpu: bool
    cost_model: Literal["free", "licensed", "credits"]

class Solver(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[SolverCapabilities]

    def __init__(self, component: Component, technology: Technology,
                 spec: SimulationSpec, workdir: Path | None = None): ...
        # MUST be cheap: no disk writes, no network, no license checks.

    def validate(self) -> list[str]: ...      # human-readable problems; [] = ok
    def build(self) -> SetupArtifacts: ...    # produce native scene OFFLINE (td.Simulation
                                              # json / .lsf script text / meep objects /
                                              # jax arrays); serializable; CI-testable
    def estimate(self) -> ResourceEstimate: ...  # cells, memory, cost hints; offline
    def run(self) -> SMatrix: ...             # the ONLY method allowed to spend money/licenses
```

Non-negotiable invariants:
- `Component.structures: list[Structure]` is **flat**; `Structure.role:
  Literal["device","substrate","superstrate"]` replaces list-nesting and name-sniffing (B19).
- Port indices come from `re.search(r"(\d+)$", name)` — never reversed digits (B6). S-matrix
  coordinates use **port names**, not concatenated digit strings.
- All lengths um, angles degrees, frequencies Hz — documented on every model.
- matplotlib, tidy3d, gdsfactory, meep, beamz, jax are optional; core hard deps: numpy, shapely,
  klayout, pyyaml, pydantic≥2.
- Old import paths (`gds_fdtd.core`, `gds_fdtd.simprocessor`, …) live on for one minor release
  as shim modules emitting `DeprecationWarning`, then die at v1.0.
- **The remote-compute invariant:** everything a solver consumes must round-trip through JSON —
  `JobSpec(component, technology, spec, solver_name).model_dump_json()` fully reconstructs the
  job on another machine, and `run_job(job) -> JobResult` is a pure function of it. Any WP that
  adds state a solver reads from somewhere other than the JobSpec (globals, cwd files, env vars
  other than credentials) violates the contract. This single rule is what makes Modal/AWS
  Batch/SLURM/Ray backends trivial adapters instead of a rewrite (see WP7.3).

---

## Part 5 — Work packages

Legend: `[FIX]` behavior change limited to the named bug · `[REFACTOR]` behavior-preserving ·
`[FEATURE]` new surface · `[INFRA]` tooling/CI. Each card is written to be pasted directly as a
task prompt. Dependencies are strict.

### Phase 0 — Safety net & tooling (do first)

Order by *prerequisites*, not card number: WP0.1 → WP0.2 → then WP0.3/WP0.4/WP0.5/WP0.6 in any
order (all four depend only on WP0.2).

---
**WP0.1 [INFRA] Tooling baseline (ruff, prek, mypy+pydantic, uv)** — no prerequisites
*Files:* `pyproject.toml`, new `.pre-commit-config.yaml`, new `justfile`, delete dead config.
*Do:*
1. In `pyproject.toml`: delete the `[tool.flake8]` block. Add `[tool.ruff]` with
   `line-length = 100`, `target-version = "py311"`, lint rules `["E","F","W","I","UP","B","A","C4"]`,
   and per-file ignore `"tests/*" = ["B011"]`. Add `[tool.mypy]` with
   `python_version = "3.11"`, `ignore_missing_imports = true`, `check_untyped_defs = true`
   (loose start; the pydantic plugin `plugins = ["pydantic.mypy"]` is added when pydantic lands
   in WP2.2). Re-enable `[tool.pytest.ini_options] addopts = "-q"` and add
   `filterwarnings = ["error::DeprecationWarning:gds_fdtd.*"]` only (global
   `"error"` on a legacy codebase is unbounded whack-a-mole — it broadens in WP7.1).
2. Change `requires-python` to `">=3.11,<3.15"`. Move sphinx/theme/docutils deps from `dev`
   extra into a new `docs` extra; leave test/lint tools in `dev`.
3. Add `.pre-commit-config.yaml` (the format is the standard; run it with **prek**, the Rust
   drop-in used by CPython/ruff/fastapi in 2026 — contributors with classic pre-commit are
   unaffected): `astral-sh/ruff-pre-commit` (ruff-check `--fix` + ruff-format),
   `pre-commit-hooks` hygiene set (trailing-whitespace, end-of-file-fixer, check-yaml,
   check-toml, check-merge-conflict, **check-added-large-files with a limit that catches stray
   `.gds`/`.hdf5`/`.fsp` artifacts** — this repo has 2.6 GB of them locally), `codespell`,
   `validate-pyproject`. No mypy/tests in hooks (they belong in CI).
4. Generate and commit `uv.lock` (`uv lock`); document `uv sync --all-extras` +
   `uv tool install prek && prek install` in CONTRIBUTING (created in WP7.2, note for now in
   README).
5. Add a `justfile` (or `noxfile.py`) with canonical tasks — `just test`, `just lint`,
   `just docs`, `just all` — so humans and CI run identical commands. Replace the drifting
   Makefile targets with delegations to it (or port Makefile → justfile wholesale).
6. Run `ruff check . --fix` **only for import-sorting and obvious mechanical fixes in
   `gds_fdtd/`**; add `# noqa` rather than restructuring logic. Zero behavior change.
*Accept:* `prek run --all-files` passes; `pytest -q tests` passes; `uv sync --all-extras` +
`pip install -e .[dev]` both work on 3.11.
*Do NOT:* rename anything; touch examples; fix bugs you notice (file them).

---
**WP0.2 [INFRA] CI architecture v2 + honest coverage** — needs WP0.1
*Files:* replace `.github/workflows/build_and_test.yml` with `ci.yml`; `tests/test_lyprocessor.py`;
README badges.
*Do:*
1. One `ci.yml`, triggers `pull_request` + `push: branches: [main]` (NEVER `'**'` — today every
   PR runs everything twice), with:
   ```yaml
   permissions: { contents: read }
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: true
   ```
2. Job graph (scientific-python guide architecture):
   - `lint`: `j178/prek-action` running all hooks `--all-files`.
   - `typecheck`: mypy (required); plus `ty` (Astral) as advisory `continue-on-error: true` —
     promote to required when ty ships stable pydantic support.
   - `test`: uv-based matrix — `astral-sh/setup-uv` with `enable-cache: true`,
     `cache-dependency-glob: "uv.lock"`, `python-version` from matrix (drop setup-python);
     `uv sync --locked`; python `["3.11","3.12","3.13"]` × ubuntu, plus macos/windows on
     min+max python; **two dependency profiles**: base (no extras — proves the
     lazy-import story) and `--all-extras`. Coverage → `codecov/codecov-action@<SHA>`.
   - `build`: `hynek/build-and-inspect-python-package@<SHA>` (builds sdist+wheel, twine-checks
     metadata, uploads dists as the artifact the release workflow later reuses).
   - `docs`: `sphinx-build` **without `-W`** (build-only smoke against the legacy docs; `-W`
     turns on in WP6.2). **Exclude `docs` from the `pass` needs-list until WP6.2** — record a
     TODO comment in ci.yml.
   - `pass`: fan-in via `re-actors/alls-green@<SHA>` with `needs:` on lint/typecheck/test/build
     — the **single** required status check, so matrix changes never desync branch protection;
     enables a merge queue later.
3. **Pin every third-party action to a full commit SHA** with trailing `# vX.Y.Z` comment
   (2025–2026 supply-chain attacks — tj-actions, actions-cool — made this table stakes).
   Rule 8 applies to action names too: before using `j178/prek-action` or any action named in
   this plan, open its repo and verify it exists at the pinned SHA; record verified SHAs in the
   PR. **Do NOT add a `github-actions` entry to `.github/dependabot.yml` — it already has one**
   (verified at `ab2cd9b`); a duplicate `(ecosystem, directory)` entry invalidates the whole
   dependabot config.
4. **Delete `test_force_full_coverage`** (`tests/test_lyprocessor.py:196-201`) and the
   commented-out test blocks (lines 151–192). Coverage will drop — that is the point; note the
   real number in the PR description.
5. Fix the README broken code fence (line ~80, the unclosed ```` ``` ```` after the pytest
   command); trim badges to signal only: PyPI, Python versions, CI (`pass`), codecov, docs,
   license (drop jekyll/dead-workflow badges).
*Accept:* one green `pass` check gates the PR; matrix runs base and all-extras profiles; all
actions SHA-pinned; coverage number is real; PR runs are single (no push+PR duplication).

---
**WP0.5 [INFRA] Workflow consolidation + Pages/deployments fix** — needs WP0.2
*Purpose:* eliminate the two duplicate pipelines and the deployment noise (see §3.5).
*Do:*
1. **Delete `jekyll-gh-pages.yml`** — it is an unmodified Jekyll sample that deploys a Jekyll
   render of the repo root over the Sphinx docs on every release.
2. **Delete `python-publish.yml`** — publishing consolidates into the single release pipeline
   (WP0.6). Delete the now-unused `PYPI_API_TOKEN` secret after WP0.6 lands.
3. Docs deployment, interim (until the RTD decision in WP6.2): replace peaceiris/branch-based
   Pages in `build_docs.yml` with the official artifact flow — Pages source set to
   "GitHub Actions"; build job `sphinx-build` → `actions/upload-pages-artifact@<SHA>`; deploy
   job with `environment: github-pages`, `permissions: pages: write, id-token: write`,
   `actions/configure-pages@<SHA>` + `actions/deploy-pages@<SHA>`; deploy **only on push to
   main**; then delete the `gh-pages` branch. Result: exactly one deployment per main push,
   zero legacy "pages build and deployment" runs.
4. **One-time deployment cleanup — destructive; guardrails mandatory:**
   (a) FIRST PASS IS LIST-ONLY: `gh api repos/SiEPIC/gds_fdtd/deployments --paginate` → print a
   table (id, environment, ref, created_at, latest status) into the PR for owner approval —
   delete nothing yet. (b) Delete pass (after approval): determine the deployment currently
   serving Pages (`gh api repos/SiEPIC/gds_fdtd/pages`) and **hard-exclude it**; for each other
   id: POST an `inactive` status, then `DELETE /repos/.../deployments/{id}`. (c) The script
   must **abort** if it would delete fewer than 1 or more than 79 rows, and must never touch a
   deployment newer than the excluded active one. Record the script in the PR.
5. Add environment protection rule on `github-pages`: deployable from `main` only.
*Accept:* three workflows remain (`ci.yml`, docs deploy, `release.yml`); deployments list shows
1 active entry and no longer grows by 2 per push; docs site still serves correctly.

---
**WP0.6 [INFRA] Release pipeline v2 (hatch-vcs + trusted publishing)** — needs WP0.2 **and
WP0.5** (WP0.5 deletes `python-publish.yml`; landing WP0.6 first would re-create the
double-publish race: the new release workflow's GitHub Release would trigger the old
`on: release` publish workflow).
**OWNER ACTIONS before starting:** register the GitHub trusted publisher on PyPI (and TestPyPI
for the dry-run); the executor cannot do this.
*Purpose:* single-source the version from git tags; one release path; no long-lived secrets.
*Do:*
1. `pyproject.toml`: build backend → `hatchling` + `hatch-vcs`
   (`[tool.hatch.version] source = "vcs"`, `dynamic = ["version"]`); delete `version =` and the
   hardcoded `__version__` in `gds_fdtd/__init__.py` (replace with
   `__version__ = importlib.metadata.version("gds_fdtd")`); `docs/conf.py` reads the same.
   Add `.git_archival.txt` + `.gitattributes` entry so GitHub tarballs are versioned.
2. **Delete** `.bumpversion.cfg`, `scripts/release.sh`, and the Makefile bump targets
   (bump2version is unmaintained; with hatch-vcs there is nothing to bump — releasing is
   `git tag v0.x.y && git push --tags`).
3. Rewrite `release.yml`: trigger `push: tags: ['v*']` →
   (a) wait-for/require CI `pass` on the tagged commit (don't re-run a private test suite),
   (b) `hynek/build-and-inspect-python-package@<SHA>`,
   (c) publish job: `environment: pypi`, `permissions: id-token: write`,
   `pypa/gh-action-pypi-publish@<SHA>` — **Trusted Publishing** (register the GitHub publisher
   on PyPI first) with default **PEP 740 attestations**; no token input at all,
   (d) `softprops/action-gh-release@<SHA>` with `generate_release_notes: true`.
4. Add `.github/release.yml` with PR-label categories (breaking/feature/fix/docs/infra) so
   auto-generated release notes are structured. (Graduate to towncrier only if release-notes
   quality becomes a real complaint — skip conventional-commit machinery entirely; a small
   scientific team won't sustain it.)
5. Run `repo-review` (scientific-python) against the repo; fix cheap findings, file the rest.
*Accept:* a `v0.4.1.dev` test tag on a fork/test-PyPI walks the whole path: version appears
from the tag, dists attested, release notes generated; `grep -rn "0\.4\.0" pyproject.toml
gds_fdtd/__init__.py docs/conf.py` → no hardcoded versions left.

---
**WP0.3 [INFRA] Golden geometry fixtures (characterization tests)** — needs WP0.2
*Purpose:* pin current behavior so later refactors can prove geometric equivalence.
*Files:* new `tests/test_golden.py`, new `tests/golden/*.json`.
*Do:*
1. Write a helper `component_to_dict(comp)` that serializes a `component` (name; for every
   structure — flattened, preserving order: name, polygon rounded to 1e-9, z_base, z_span,
   sidewall_angle, layer; every port: name, **idx**, center, width, direction, height; bounds
   vertices/z). **Materials: serialize the raw technology-YAML mapping for that layer (e.g.
   `{"lum_db": {...}}`) — NEVER `repr()` of a solver object** (solver reprs change across
   versions and would break the goldens at every dependency bump).
2. Use `tests/tech_lumerical.yaml` (pure-string materials) so goldens build and run with
   **zero extras installed** — the safety net must work in the base CI profile. Fixtures:
   `tests/si_sin_escalator.gds`, and `examples/devices.gds` top cells `crossing_te1550` and
   `directional_coupler_te1550`. Recipe (note `load_cell` returns a tuple):
   ```python
   cell, _ = load_cell(fname, top_cell=...)
   comp = load_component_from_tech(cell=cell, tech=tech)
   ```
   Dump JSON into `tests/golden/`.
3. Test asserts current output == stored JSON (floats via `pytest.approx`, 1e-9).
*Accept:* golden tests pass with no optional extras installed; JSON committed; regenerating
produces identical files.
*Do NOT:* fix any bug (B1 etc.) — capture behavior as-is, bugs and all; note oddities in PR.

---
**WP0.4 [INFRA] Examples can never rot again** — needs WP0.2
*Files:* new `tests/test_examples_importable.py` (or CI step), `examples/**`.
*Do:*
1. Add a test that, for every `examples/**/*.py`, runs
   `ast.parse` + resolves top-level imports against the installed package **without executing**
   the example (walk `ast.Import`/`ast.ImportFrom` for `gds_fdtd.*` names and assert
   `getattr` chain resolves). Mark the currently-broken ones
   (`01a`,`01b`,`05a`,`07a`,`08a_siepic_tidy3d`,`08a_siepic_lumerical`) as `xfail` with reason
   "imports deleted API — rewritten in WP6.1".
2. Scope: this check runs **only in the `--all-extras` CI leg** (examples legitimately import
   tidy3d/gdsfactory); an `ImportError` of a *third-party* optional dependency (not a
   `gds_fdtd.*` symbol) is a skip, not a failure.
*Accept:* test suite green; any future example importing a nonexistent `gds_fdtd` symbol fails
CI.

### Phase 1 — Kill the bugs (all need WP0.3; independent of each other unless noted)

---
**WP1.1 [FIX] B1 `dilate_1d` list-multiplication** — `lyprocessor.py:33-66`
Replace the function body with a correct implementation:
```python
def dilate_1d(vertices, extension=1, dim="y"):
    (x1, y1), (x2, y2) = vertices
    ex = abs(extension)
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    if dim == "x":   return [[x1 - ex*sx, y1], [x2 + ex*sx, y2]]
    if dim == "y":   return [[x1, y1 - ex*sy], [x2, y2 + ex*sy]]
    if dim == "xy":  return [[x1 - ex*sx, y1 - ex*sy], [x2 + ex*sx, y2 + ex*sy]]
    raise ValueError("dim must be 'x', 'y' or 'xy'")
```
(Note the original `"xy"` also failed to extend x1 — verify against golden fixtures whether any
fixture path used `"xy"`; `from_gdsfactory` calls it with `extension=0`, so goldens should hold.)
Add parametrized tests including **descending** vertex order (`[[0,4],[0,0]]`) — the case the old
code corrupted. Delete the now-wrong expectation rows if any; do not delete the test function.
*Accept:* new tests pass; golden fixtures unchanged.

---
**WP1.2 [FIX] B5+B17 solver defaults** — `solver.py`
1. All mutable defaults → `None` sentinels resolved in `__init__`
   (`boundary = list(boundary) if boundary is not None else ["PML"]*3`, etc.).
2. `port_input: port | list[port] | None = None`; `None` ⇒ all ports active (document this);
   normalize to `list[port]` once in `__init__`; simplify `_get_active_ports` accordingly.
3. Same treatment in `fdtd_port.__init__` and `core.structure.__init__` (`layer=[1,0]`).
*Accept:* (a) unit test on the **base class** with a stub component and no solver imports:
no-`port_input` construction normalizes to all component ports; (b) two instances share no
mutable state (mutate one's list attribute, assert the other unchanged); (c) optional bonus,
not the gate: `@pytest.mark.skipif(tidy3d missing)` end-to-end construction test.

---
**WP1.3 [FIX] B6 port index parsing** — `core.py:176-186`, `sparams.py:46-75`
1. `port.idx`: `m = re.search(r"(\d+)$", self.name)`; raise `ValueError` with the port name if
   no trailing digits. Delete the reversed-digits comment/behavior.
2. `sparams.s.idn*`: same trailing-digits rule; change `idn` format to
   `S{out}_{in}@m{mode_out}{mode_in}`? **No** — keep the external format but build it from the
   *parsed integers*, and fix `sparameters.S()` to compare parsed integers
   (`d.out_port_num == out_port and d.in_port_num == in_port and ...`) instead of string
   concatenation, eliminating the 2-digit ambiguity.
3. Tests: `opt1/opt2` (existing behavior), `opt10` vs `opt1` (must differ), `port12` → 12.
*Accept:* `tests/test_core.py::test_port_idx` updated expectations pass (the existing
parametrization at line 302 encodes the reversed behavior — update it and say so in the PR);
goldens for standard ≤9-port fixtures unchanged.

---
**WP1.4 [FIX] B7+B8** — `simprocessor.py:20-41`, `lyprocessor.py:166-207`
1. `get_material`: raise `ValueError(f"technology material for {device} has 'lum_db' without
   'model'")` instead of `UnboundLocalError`; same defensive check for `tidy3d_db` (must contain
   `nk` or `model`).
2. `load_region`: iterate shapes; collect the first box-or-polygon; if none found raise
   `ValueError("no DevRec box/polygon found on layer ...")`; if multiple found, log a warning
   and use their union's bbox.
*Accept:* new unit tests for the three failure modes; goldens unchanged.

---
**WP1.5 [FIX] B9+B10+B11 tidy3d geometry & parity** — `solver_tidy3d.py` (needs WP1.2)
1. Substrate/superstrate extension and `_create_field_monitor` centers: use
   `device.bounds.x_center/y_center` offsets, not origin (compute extended rectangle around the
   component center).
2. Honor `self.boundary`: map `"PML"→td.PML()`, `"Metal"→td.PECBoundary()`, `"Periodic"→
   td.Periodic()` per axis into `td.BoundarySpec`; raise on unknown strings.
3. `run_time`: use `self._calculate_simulation_time(max(sim_size)*1e-6)` like Lumerical.
4. Delete the unused `mode_indices` line and the stray `print(...)` at `solver_tidy3d.py:307`.
*Tests:* construct the solver against `tests/si_sin_escalator.gds` with tidy3d installed but NO
network: assert `base_simulation.boundary_spec` per-axis types, monitor centers, and run_time
value. Mark `@pytest.mark.skipif(tidy3d missing)`.
*Accept:* tests pass offline; no cloud calls anywhere in `setup()`/`_create_*`.

---
**WP1.6 [FIX] B13+B14 sparams math & .dat round-trip**
1. Delete the `np.angle(phase)**2` lines; plot phase as `np.unwrap(s_phase)`.
2. `core.sparam.plot`: magnitude as `10*np.log10(np.abs(s)**2)`.
3. Rewrite `s_parameter_writer` usage: export directly from `sparameters.data` — write one
   header+block per data entry with its **actual** in/out port and mode ids (do not iterate
   `range(n_ports)²`). Keep the file format identical (INTERCONNECT `.dat`).
4. Add the round-trip test: build a synthetic `sparameters` with 2 ports × 2 modes,
   `export → process_dat → compare` all entries (freq, mag, phase to 1e-9).
*Accept:* round-trip test passes; a real-Lumerical `.dat` fixture still parses. **Fixture
handling — safety-critical:** copy ONLY the `.dat` file itself into `tests/recorded/` and
`git add` that single file. NEVER add anything else from `examples/notebooks/faid/` — its
sibling `.log` files contain an embedded ANSYS licensing context token (base64 blob with
license handler + hostname). Before committing, verify:
`grep -iE 'licens|ANSYS|HANDLER|@' tests/recorded/*.dat` → empty.

---
**WP1.7 [FIX] B16 logging hygiene** — `logging_config.py`, all modules
1. Replace `setup_logging` internals: create/configure only the `"gds_fdtd"` logger (file +
   console handlers on it, `propagate = False`); never touch the root logger. Add
   `logging.getLogger("gds_fdtd").addHandler(logging.NullHandler())` in `gds_fdtd/__init__.py`.
2. Rename the `"dreamcompiler"` logger (`sparams.py:15`) to `__name__`.
3. Remove `print()` calls that duplicate an adjacent `logger.*` call in `solver*.py` (keep the
   user-facing summary prints for now — they die in WP3.1).
*Accept:* test that configures a root handler, constructs a solver, and asserts root handlers
are untouched and root level unchanged.

---
**WP1.8 [FIX] B15 prefab path** — `lyprocessor.py:69-127`
Make `load_device` return the component it builds; write the `_with_extensions.gds` into a
`tempfile`/workdir instead of next to the input; give `load_cell` no prefab kwarg (fix example
in WP6.1); `apply_prefab(gds_in, gds_out, model)` signature — never overwrite input.
*Accept:* unit test with mocked `prefab` module (pattern exists in `test_lyprocessor.py`).

---
**WP1.9 [FIX] B12 remove dead field-data plumbing** — `solver_tidy3d.py`
Delete `_set_field_data_in_monitors` and the `tidy3d_field_monitor.set_tidy3d_data` probing;
keep monitor *creation* (monitors still go into the base simulation and their data exists in the
per-task `SimulationData`). Field visualization returns in WP4.1 built on
`ModalComponentModelerData`. Update `run()` accordingly.
*Accept:* no references remain; offline setup tests (WP1.5) still pass.

### Phase 2 — Core refactor (strict order; each needs the previous)

---
**WP2.1 [REFACTOR] src layout + naming** — needs all Phase 1
1. Move package to `src/gds_fdtd/`; update pyproject (flit/hatchling `packages`/`src` config),
   CI, Makefile.
2. `core.py` → `geometry.py`: rename classes `port→Port`, `structure→Structure`,
   `region→Region`, `component→Component`, `layout→LayoutSource`. Keep constructor signatures.
3. Create shim `gds_fdtd/core.py` re-exporting old names:
   `port = _deprecated_alias(Port, "core.port")` emitting `DeprecationWarning` once per name.
   Same shim pattern for module moves in later WPs.
4. Mechanical rename across package/tests/examples via exact-match search; goldens must not
   change (they serialize values, not class names — verify).
*Accept:* full suite + goldens green; `from gds_fdtd.core import component` still works and
warns; `ruff check` clean.
*Do NOT:* change any logic, defaults, or docstring semantics in this WP.

---
**WP2.2 [REFACTOR] Technology as pydantic model** — needs WP2.1
0. **Add `pydantic>=2.7` to `[project.dependencies]`** and `plugins = ["pydantic.mypy"]` to
   `[tool.mypy]` — this is the dependency addition explicitly authorized for this WP (rule 6).
1. New `technology.py`: `MaterialSpec` (fields: `nk: float | tuple[float,float] | None`,
   `solver_hints: dict[str, Any]` — where `tidy3d_db`/`lum_db` YAML keys land), `DeviceLayer`
   (`layer: tuple[int,int]`, `z_base/z_span: float`, `sidewall_angle: float = 90`,
   `material: MaterialSpec`), **`BackgroundLayer`** (`z_base`, `z_span`, `material` — no layer
   tuple; substrate/superstrate parse from the existing list-of-one YAML shape), `Technology`
   (`name`, `substrate/superstrate: BackgroundLayer`, `pinrec/devrec: list[tuple[int,int]]`,
   `device: list[DeviceLayer]`).
   `Technology.from_yaml(path)` parses the **existing YAML format unchanged**, with pydantic
   `ValidationError`s that name the YAML path of the problem. Add an optional
   `schema_version: 1` key (default 1 when absent) and document in the schema page: **v1 keys
   are frozen through the 1.x series**; generalizing `lum_db`/`tidy3d_db` into neutral
   `solver_hints` is a v2 schema change gated on an owner decision (Part 8 Q13) — do NOT
   change YAML key names in this WP.
2. Material resolution moves to solver adapters: `simprocessor.get_material` becomes
   `solvers/…` code later; for now keep a compatibility function
   `technology.to_legacy_dict()` producing exactly the old `to_dict()` shape so
   `simprocessor`/solvers run unmodified.
2b. **refractiveindex.info support (owner-requested):** `MaterialSpec` accepts a new OPTIONAL
   material source — a reference into the refractiveindex.info database:
   ```yaml
   material:
     rii: {shelf: main, book: Si, page: Li-293}        # refractiveindex.info coordinates
   ```
   Adding a new optional key is **additive and backward-compatible** (v1 schema freeze applies
   to existing keys, not to additions). Resolution: a `gds_fdtd/materials/rii.py` loader
   returns tabulated (wavelength, n, k) arrays; wrapper deps to evaluate at implementation
   time (rule 8 — verify APIs on the installed package): `refidx`, `refractiveindex`,
   `PyOptik` — or vendor a direct YAML reader against the official
   `polyanskiy/refractiveindex.info-database` (the DB is plain YAML; a ~100-line reader avoids
   a dependency). Requirements: (a) **offline-first** — the DB (or the used pages) is cached
   under `GDS_FDTD_CACHE_DIR`; CI/docs never download (commit one tiny page fixture, e.g.
   Si Li-293, for tests); (b) solver conversion happens in adapters: tidy3d →
   `td.FastDispersionFitter`/`from_nk_data` fit (verify exact 2.11 API), Lumerical → sampled
   nk import, MEEP/beamz → `validate()` error if dispersive unless single-frequency,
   Tier B kernels → nk interpolated at each simulated wavelength; (c) `validate()` reports
   when a material's tabulated range doesn't cover the simulation band.
3. Add `Technology.model_json_schema()`-based schema doc generation hook (used by docs later).
*Accept:* goldens unchanged (they flow through `to_legacy_dict`); new validation tests: missing
`z_span`, bad layer list length, `lum_db` w/o model → readable errors naming the offending key;
existing `tests/tech_*.yaml` and `examples/tech_*.yaml` all load; a tech YAML with an `rii`
material loads offline from the committed page fixture and returns n(1.55 µm) ≈ 3.48 for Si
(±0.05).

---
**WP2.3 [REFACTOR] Flat structures + roles (kills B19)** — needs WP2.2
1. `Structure` gains `role: Literal["device","substrate","superstrate"] = "device"`.
2. `Component.structures` becomes flat `list[Structure]`; `load_component_from_tech` and
   `from_gdsfactory` build it flat with roles; `initialize_ports_z` matches ports only against
   `role=="device"` structures (preserves current z-resolution semantics — device lists were
   what assigned port z before; confirm against goldens).
3. Replace every `type(s) == list` branch (`geometry.py`, `solver_tidy3d.py:246`,
   `solver.py`) and every name-sniff (`"substrate" in name.lower()`, incl. `"subtrate"`) with
   role checks. Fix the "Subtrate" typo everywhere.
4. Regenerate goldens **only if** ordering changes are provable-equivalent (document the diff in
   the PR; port z/material assignments must be identical).
*Accept:* suite green; `grep -rn "type(s) == list\|subtrate" src/` returns nothing.

---
**WP2.4 [FEATURE] One SMatrix to rule them all** — needs WP2.1 (parallel to WP2.3 ok)
**Execute as FOUR sessions/PRs, in order** (one card fed per session, quoting only its letter):
**2.4a** = item 1 core class + hdf5/npz I/O + accessors + checks; **2.4b** = `.dat` r/w +
item 2 shims + item 3 deletion (gated by the WP1.6 round-trip test); **2.4c** = Touchstone
export — convention fixed here, not delegated: *flatten (port, mode) pairs into Touchstone
ports ordered `(p1,m1),(p1,m2),…,(pN,mM)`, one `.sNp` file with N=P×M ports, header comment
lines documenting the mapping*, validated by reading back with `skrf`; **2.4d** = `plotting.py`.
1. New `smatrix.py`:
   ```python
   class SMatrix:
       f: np.ndarray                     # Hz, shape (F,)
       s: np.ndarray                     # complex, shape (F, P, P, M, M)  [out, in]
       port_names: list[str]             # len P
       n_modes: int
       # constructors: from_entries(...), from_dat(path), from_hdf5(path)
       # exports: to_dat(path)  [INTERCONNECT], to_touchstone(path) [.snp, one file per
       #          mode pair, ports flattened, document convention], to_hdf5(path)
       # accessors: sel(out="opt2", in_="opt1", mode_out=1, mode_in=1) -> complex (F,)
       #            wavelength property (um)
       # checks: is_reciprocal(atol), is_passive(atol), power_balance() -> DataFrame-ish dict
   ```
   numpy-only (no xarray dependency in core).
2. Rewrite `sparams.py` as thin adapters: `process_dat` → returns `SMatrix`;
   keep `sparameters` class as a deprecated shim wrapping `SMatrix` (constructor + `add_data`
   + `S()` + `plot` used by solver code and example 03).
3. Delete `core.sparam`/`core.s_parameters` (shimmed in `core.py` with deprecation).
4. Plotting: `plotting.py` with `plot_smatrix(sm, kind="db"|"phase"|"linear")`; matplotlib
   imported lazily.
*Accept:* `.dat` round-trip test (WP1.6) rewritten against `SMatrix` passes; touchstone file
readable by `scikit-rf` (add `skrf` to dev deps for the test only); reciprocity check flags a
deliberately non-reciprocal fixture.

### Phase 3 — Solver contract

---
**WP3.1 [REFACTOR+FEATURE] Solver ABC v2** — needs WP2.3 + WP2.4
**Execute as FIVE sessions/PRs, in order** — this is the highest-risk card in the plan and
half-porting it is the historical failure mode: **3.1a** = item 1 `spec.py` (old solvers
consume it via a thin adapter; zero behavior change); **3.1b** = item 2 `solvers/base.py` +
`SetupArtifacts` + a `FakeSolver` in tests proving the contract end-to-end; **3.1c** = port
tidy3d onto the ABC; **3.1d** = port lumerical (including moving the module-level `lumapi`
import inside methods); **3.1e** = item 4 registry + item 3 old-constructor shims +
`available_solvers()` + item 5. Each sub-PR leaves BOTH solvers importable and green.
1. New `spec.py`: `SimulationSpec` pydantic model gathering all of today's ~15 numeric solver
   kwargs (wavelength_start/end/points, mesh, boundary (3 enums), symmetry (3 ints in {-1,0,1}),
   z_min/z_max, width_ports, depth_ports, buffer, modes (list[int], 1-based, validated ≥1),
   run_time_factor, mode_freq_pts, field_monitors). All defaults = today's defaults.
   Validators replicate `_validate_simulation_parameters` (then delete that method).
2. New `solvers/base.py` implementing the **Architecture contract** (Part 4) `Solver` ABC:
   port conversion (`_convert_component_ports_to_fdtd_ports` moves here as `injection_plan()`),
   domain computation, `SetupArtifacts` dataclass (`files: dict[str, Path]`,
   `native: Any`, `summary: dict`). `build()` replaces `setup()` and MUST NOT be called from
   `__init__`. `run()` calls `build()` if needed, returns `SMatrix`.
3. Port `solver_tidy3d`/`solver_lumerical` onto the new ABC in `solvers/tidy3d.py` /
   `solvers/lumerical.py`. Old constructors (`fdtd_solver_tidy3d(component=..., tech=...,
   wavelength_start=...)`) remain as shims that assemble a `SimulationSpec` and forward, with
   `DeprecationWarning`.
4. Registry: `[project.entry-points."gds_fdtd.solvers"] tidy3d = "gds_fdtd.solvers.tidy3d:Tidy3DSolver"`
   etc.; `gds_fdtd.get_solver(name)` and `gds_fdtd.available_solvers()` (import errors →
   listed as unavailable with reason, not raised).
5. Replace print-summaries with `Solver.describe() -> str` (called by examples explicitly).
*Accept:* offline tests: `build()` for tidy3d returns artifacts containing a serialized
`td.Simulation` (json) without network; for lumerical WITHOUT lumapi installed,
`available_solvers()` reports it unavailable instead of ImportError at package import
(today `solver_lumerical.py:9` imports lumapi at module top — fix that here); old-style
constructor test passes with a DeprecationWarning.

---
**WP3.2 [FEATURE] Lumerical adapter hardening** — needs WP3.1
1. All `fdtd.eval(f'...')` strings move behind one helper `self._lsf(cmd: str, **params)` that
   escapes/validates params (no raw f-string interpolation of user strings — quote and escape
   `"` in names/paths).
2. `build()` produces the complete `.lsf` setup script *as text* in `SetupArtifacts` (so CI
   asserts on script content without a license); `run()` replays it through lumapi.
3. Version probe (`fdtd.version()` or equivalent): implement 2025 GPU/resource syntax alongside
   2024 (the TODOs at old `solver_lumerical.py:73,88,357`); select by probe; unit-test both
   branches with a mocked lumapi.
4. Replace the `while True: removesweepparameter` loop with a bounded query of existing
   parameters.
*Accept:* mocked-lumapi tests assert generated script for the crossing fixture (ports, layers,
sweep entries); no module-level lumapi import.

### Phase 4 — Ecosystem upgrades

---
**WP4.1 [FEATURE] tidy3d 2.11 migration** — needs WP3.1
1. Bump pin to `tidy3d>=2.11,<3`. Read the official migration notes in the tidy3d changelog
   (search "ModalComponentModeler"). The WP0.3 golden fixtures are tidy3d-independent by
   design (lumerical tech, raw-YAML material serialization) — **they must not change in this
   WP; if they fail, you broke geometry, not materials. Do not regenerate them.**
2. In `solvers/tidy3d.py`: `ComponentModeler` → `ModalComponentModeler`; move
   `verbose`/`path_dir` out of the modeler; `run()` becomes
   `modeler_data = td.web.run(modeler, task_name=f"gds_fdtd-{component.name}")`,
   `smat = modeler_data.smatrix()`; convert the returned DataArray into `SMatrix`
   (coords: `port_in`, `port_out`, `mode_index_in`, `mode_index_out`, `f` — **verify exact coord
   names against installed 2.11**, do not trust this doc).
3. Restore field retrieval properly: per-excitation `SimulationData` from the modeler data
   (check `modeler_data` attributes on the installed version) → `plot_fields()` helper.
4. Update `_create_field_monitor` and boundary code for any 2.9→2.11 renames the import
   surfaces.
*Accept:* offline: modeler constructs & serializes for the crossing fixture; DataArray→SMatrix
conversion unit-tested with a synthetic DataArray; one **manual** cloud smoke test documented in
the PR (human runs it — costs credits).

---
**WP4.2 [FEATURE] gdsfactory 9 converter rewrite** — needs WP2.3 (independent of WP4.1)
Rewrite `layout/gdsfactory.py::from_gdsfactory(c, tech, z_span)` from scratch. Rules
(verified against gf 9 migration guide — re-verify on installed version first):
- polygons: `c.get_polygons_points(by="tuple")` → `{(layer,datatype): [Nx2 float um arrays]}`;
  build one `Structure(role="device")` per polygon for each layer present in `tech.device`.
- ports: `p.center` (um floats), `p.width` (um), `p.orientation` (deg, may be non-90° —
  snap to nearest of {0,90,180,270} if within 1e-6, else raise NotImplementedError naming the
  port), layer tuple via the gf9 helper (verify: `p.layer_info` or
  `gf.get_layer_tuple(p.layer)`).
- bounds from `c.bbox()`/`c.dbbox()` (verify which returns um in gf9) dilated by the same 1.9 um
  rule as before; substrate/superstrate from tech as `role=` structures.
- port names: use the **port's own name**; ensure trailing-digit rule (append `f"{i+1}"` if a
  name lacks digits, warn).
*Tests* (gate on `pytest.importorskip("gdsfactory")`): `gf.components.straight(length=10)`
→ 2 ports at x=0 and x=10 (um!), width 0.5, directions {180, 0}; polygon count ≥1 on layer
(1,0); `bend_circular` smoke; goldens for these two committed.
*Accept:* the two component tests pass with installed gf 9.x; no use of removed `d*` attrs.

---
**WP4.3 [INFRA] Dependency floor refresh** — needs WP4.1, WP4.2
Bump floors: numpy≥1.26, shapely≥2.0, klayout≥0.30, pydantic≥2.7; extras: `tidy3d>=2.11,<3`,
`gdsfactory>=9.40,<10`, `prefab>=1.2`, `siepic>=0.5.25` (verify each still installs together);
add extras `meep` (documented as conda-only — extra is a no-op marker), `beamz>=0.4`,
`fdtdz`, `jax`. Re-lock (`uv lock`) and verify the WP7.1 lower-bounds job passes at the new
floors.
*Accept:* CI green on 3.11–3.13 at both locked and lowest-direct resolutions;
`pip install -e .[dev]` still works for non-uv users.

### Phase 5 — New solvers (the agnosticism payoff)

Order matters: MEEP first (free end-to-end CI), then beamz (Tier A, easy), then the Tier B
enablers (grid/modes), then fdtdz.

---
**WP5.1 [FEATURE] MEEP adapter** — needs WP3.1
`solvers/meep.py::MeepSolver` (Tier A, local, free):
- `build()`: structures → `mp.Prism` lists (vertices um→meep units with a=1 um length scale;
  z from z_base/z_span; sidewall angle unsupported by Prism — document, `validate()` warns when
  angle ≠ 90), materials from `MaterialSpec.nk` (`mp.Medium(index=n)`; dispersive model hints
  unsupported → `validate()` error), PML from spec.boundary.
- ports: `mp.EigenModeSource` at input port; `mp.ModeRegion`/`get_eigenmode_coefficients` at
  all ports; run until fields decay (`stop_when_fields_decayed` driven by
  spec.run_time_factor).
- one forward run per active (port, mode) as in the other adapters; assemble `SMatrix` from
  eigenmode coefficient ratios (normalize by input-port forward coefficient).
- CI: meep from conda-forge in a dedicated workflow job (`mamba-org/setup-micromamba`), run a
  **tiny 3D straight-waveguide** fixture (2–3 um, coarse mesh, resolution ~15) and assert
  |S21|² > 0.8, |S11|² < 0.1 across the band. Mark `@pytest.mark.slow`; run on a schedule +
  when `solvers/meep.py` changes.
*Accept:* offline geometry tests (no meep needed — assert Prism vertex math via mocks) + the
conda CI job passing.

---
**WP5.2 [FEATURE] Tier-B enablers: grid.py + modes.py** — needs WP2.3
**Execute as THREE sessions/PRs:** **5.2a** = item 1 rasterizer (+analytic rectangle test);
**5.2b** = item 2 `ModeSolver` protocol + femwell backend (+n_eff test); **5.2c** = tidy3d
mode-solver backend + item 3 extraction (+self-overlap=1 test).
1. `grid.py::rasterize(component, technology, dx, dy, dz, wavelength) -> PermittivityGrid`
   (dataclass: `eps: np.ndarray (Nx,Ny,Nz)`, origin, spacing). Use shapely per z-slab +
   sub-pixel averaging (supersample 4× or analytic polygon-cell overlap via
   `shapely.intersection` area fraction). Sidewall angle: per-z-slice polygon offset
   (`shapely.buffer` with distance `dz*tan(90-angle)` per slice).
2. `modes.py`: `ModeSolver` protocol (`solve(eps_cross_section, dl, wavelength, n_modes) ->
   list[Mode]`, `Mode` = fields E/H arrays + n_eff); backends: `FemwellModeSolver`
   (extra `femwell`), `Tidy3DModeSolver` (uses tidy3d's **local** mode solver plugin — free,
   offline; verify import path `tidy3d.plugins.mode.ModeSolver` on 2.11).
3. `extraction.py` (or inside smatrix.py): overlap integrals of recorded DFT fields against
   mode profiles → complex amplitudes → SMatrix entries. This must be solver-independent
   (numpy arrays in, complex out) so beamz/fdtdz/fdtdx and even MEEP cross-checks share it.
*Tests:* rasterizer vs analytic rectangle (area fraction exact to 1e-3 at 10 nm grid);
mode solver: 220×500 nm Si strip in SiO2 @1.55 um → n_eff(TE0) ≈ 2.4±0.1 (literature value);
overlap of a mode with itself = 1.
*Accept:* all offline; femwell in dev-test extra.

---
**WP5.3 [FEATURE] beamz adapter** — needs WP3.1, WP5.2 (for extraction fallback)
`solvers/beamz.py::BeamzSolver` (Tier A, Apache-2.0, pip, JAX CPU/GPU). beamz 0.4.x advertises
native GDS import, mode sources, DFT monitors and an S-parameter workflow — but it is a 0.x
package: **rule 8 applies with force — before writing any code, install it, read its actual
API surface, and record the verified class/function names in the PR description**; treat every
beamz claim in this plan as a hypothesis. Prefer its native high-level API: `build()` assembles the beamz simulation from `Component` polygons (or write a
temp GDS via `Component.export_gds` and use beamz's GDS import — decide by trying both against
the installed version; document the choice), materials from `MaterialSpec.nk` (beamz is
non-dispersive-first; `validate()` errors on dispersive-model-only materials), sources/monitors
at fdtd_port positions. `run()` executes locally (JAX; honor
`capabilities.supports_gpu` = jax backend detection) and returns `SMatrix` either from beamz's
native S-param output or, if its conventions don't match, from `extraction.py` overlap on its
DFT fields — validate which during implementation.
*Accept:* pip-installable in CI (add job); straight-waveguide fixture: |S21|² > 0.8 at coarse
resolution (mark slow); offline scene-construction tests (mock jax not needed — CPU jax fine).

---
**WP5.4 [FEATURE] fdtdz adapter (Tier B)** — needs WP5.2
`solvers/fdtdz.py::FdtdzSolver`. fdtdz exposes one low-level `fdtdz()` GPU kernel: permittivity
grid in, raw E-fields at sampled timesteps out; **no mode solver, no monitors, no dispersive
materials, limited z cells** (design constraint of the kernel).
- `validate()`: reject dispersive materials, z-extent beyond kernel limit (read limit from
  fdtdz docs/installed constants), missing GPU jax.
- `build()`: `grid.rasterize(...)` at spec-derived resolution → kernel arrays; mode profiles
  from `modes.py` at each port cross-section; source = time-modulated mode profile of the
  input port (ramped CW per wavelength point, or broadband pulse + FFT — start with per-λ CW,
  simplest correct thing; document cost O(n_wavelengths runs)).
- `run()`: launch kernel (GPU-only; skip in CI), FFT the field record at port planes,
  `extraction.py` overlap → SMatrix.
*Accept:* everything except the kernel launch is unit-tested offline (rasterize→source
synthesis→extraction pipeline with a fabricated analytic field record); GPU smoke test script
committed under `examples/` and documented as manual. Also add `fdtdx` to Part 7 backlog with
notes — same Tier B machinery applies.

---
**WP5.5 [FEATURE] Convergence + caching + cross-solver validation** — needs ≥2 working solvers
1. `convergence.py::sweep(solver_cls, component, tech, spec, field: str, values: list) ->
   ConvergenceReport` (S-param deltas between successive values; plot; recommend value at
   tolerance). Generalizes examples `02b`/`03b`.
2. Simulation caching: `hash = sha256(canonical_json(component) + technology + spec +
   solver.name + solver_version)`; `run(cache_dir=...)` short-circuits to `SMatrix.from_hdf5`.
3. `validate_across(solvers, component, tech, spec) -> report` — the cross-solver comparison
   (max |ΔS| in dB per entry). This is the flagship "agnostic" demo (cf. arXiv 2506.16665).
*Accept:* convergence + cache unit tests with a `FakeSolver` returning canned SMatrix; a
documented MEEP-vs-beamz straight-waveguide comparison in CI (slow job).

### Phase 6 — Examples, docs, release

---
**WP6.1 [FEATURE] Examples rewritten** — needs WP3.1 (solver-specific ones need their WP)
Delete 01a/01b/05a/07a/08a legacy examples; new set, each ≤80 lines, each starts with the
solver-availability guard, each runnable headless up to (not including) `run()`:
`01_load_and_inspect.py` (GDS→Component→describe/plot), `02_tidy3d.py`, `03_lumerical.py`,
`04_meep.py`, `05_beamz.py`, `06_fdtdz.py`, `07_gdsfactory.py`, `08_siepic_pdk.py`,
`09_convergence.py`, `10_cross_solver.py`, `11_prefab.py`. Remove the WP0.4 xfails.
*Accept:* import-check green with zero xfails; each example's build path (through
`solver.build()`) executed in CI where deps allow.

---
**WP6.2 [FEATURE] Docs overhaul (the "best docs there is" blueprint)** — needs WP6.1
**OWNER ACTIONS before starting:** connect the repo on readthedocs.org and enable PR previews
in the RTD project settings — the executor cannot; a committed `.readthedocs.yaml` alone does
not make previews appear.
**Execute as FOUR sessions/PRs:** **6.2a** = RTD skeleton (`.readthedocs.yaml`, theme, MyST-NB
pipeline, existing content migrated, `-W` turned on and the `docs` job joins the CI `pass`
gate); **6.2b** = Get-started tutorials + how-tos, CI-executed; **6.2c** = API reference +
autodoc-pydantic + generated schema page; **6.2d** = README rewrite + llms.txt + Sybil +
lychee.
*Stack decision* (researched 2026-07; note **mkdocs-material is in maintenance mode** since
Nov 2025 and its successor Zensical is pre-1.0 — Sphinx is the only safe scientific-docs bet):
**Sphinx + pydata-sphinx-theme + MyST-NB (+jupyter-cache) + numpydoc/napoleon +
autodoc/autosummary + autodoc-pydantic, hosted on Read the Docs.** RTD gives versioned docs
(`stable`/`latest` switcher), **PR preview builds with visual diff comments**, and server-side
search with zero workflow code — replacing the GitHub Pages pipeline entirely (WP0.5's Pages
setup then gets deleted or reduced to a redirect; deployment noise → zero). Intersphinx to
python/numpy/matplotlib/pydantic/tidy3d/gdsfactory (tidy3d and gdsfactory are both Sphinx —
cross-links to `tidy3d.Simulation`/`gf.Component` come free).
*Structure* — Diátaxis with photonics-native names:
1. **Get started** — install + one ≤15-min end-to-end GDS→S-params tutorial **per free solver
   path** (MEEP/beamz), CI-executed so it can never break.
2. **How-to guides** — task recipes: write a technology YAML, import from gdsfactory, run on
   each solver, export Touchstone/.dat, sweep a parameter, run remotely via the CLI/JobSpec.
3. **Examples gallery** — the WP6.1 examples as **jupytext-paired `.py` sources with committed
   executed `.ipynb` outputs** (the tidy3d-notebooks pattern, in-repo). MyST-NB
   `nb_execution_mode = "cache"`: cheap notebooks execute in the docs build; expensive ones
   (cloud credits / licenses / GPU) render from committed outputs and carry a standard header
   cell stating cost ("~2 min, ~0.1 FlexCredit"). A `just run-examples SOLVER=tidy3d` target
   re-executes + re-commits outputs, run by a maintainer pre-release (or via the WP7.5 cloud/
   self-hosted lanes). **The docs build must never require a license, API key, or GPU.**
4. **Explanation** — how the GDS→simulation pipeline works, port auto-detection, meshing &
   boundary choices, S-parameter conventions/normalization, solver tiers (steal MEEP's depth).
5. **API reference** — curated `api.rst` grouped by workflow stage (Layout / Technology / Spec /
   Solvers / S-Matrix / Execution), autosummary-templated pages, numpydoc style enforced via
   ruff pydocstyle rules; **autodoc-pydantic** renders Technology/SimulationSpec/JobSpec fields,
   validators, and collapsible JSON schema; plus a generated **"Technology file format"
   reference page** from `Technology.model_json_schema()` and the raw `.json` schema published
   as a downloadable artifact (editors/agents can then validate tech YAMLs via
   yaml-language-server association).
*Also:* "Add a solver in 200 lines" guide walking MeepSolver; architecture page kept in sync
with Part 4; honest CHANGELOG for 1.0 (explicitly noting the 0.4.0 claims that ship only now);
old→new migration table; `llms.txt` + `llms-full.txt` (gdsfactory-style — a large share of
2026 doc reads are by coding agents); README rewritten to: ≤15-line hero example with an
output S-param plot image, a **solver support matrix** (solver × status/license/extra/
last-verified), ≤6 badges, "what gds_fdtd is NOT" paragraph, citing section.
*Quality gates:* `sphinx-build -W --keep-going` in CI (from day one of the new docs);
**Sybil** executes code blocks in prose pages under pytest; doctests on cheap pure functions;
**lychee** link check as a weekly cron with auto-filed issue; `.readthedocs.yaml` committed.
*Accept:* RTD builds green with `-W`; PR preview link appears on a test PR; all Get-started
tutorials execute in CI; schema page regenerates from the model; lychee green; README hero
runs as-is in a fresh venv (`pip install gds-fdtd[meep-or-beamz-path]`).

---
**WP6.2b [FEATURE] Citation & academic credibility** — needs WP6.2 (cheap, high leverage)
1. Add `CITATION.cff` (GitHub renders "Cite this repository"); enable the Zenodo–GitHub
   integration so every release mints a versioned DOI; DOI badge in README.
2. Prepare a **JOSS submission at v1.0**: statement-of-need, install docs, executed examples,
   API docs, tests, CONTRIBUTING — all already produced by this plan; JOSS review doubles as an
   external docs/packaging audit and yields a citable paper (photonics users cite tools in
   papers; this materially drives adoption for a SiEPIC-ecosystem package).
*Accept:* DOI minted on next tagged release; JOSS checklist issue filed with all items green.

---
**WP6.3 [INFRA] Release v1.0.0** — needs everything through Phase 7
Remove nothing yet: ship 1.0 with deprecation shims; file a tracking issue to delete shims in
1.1. Trusted-publisher PyPI workflow (replace token if applicable); tag; announce upgrade notes
(SiEPIC-Tools consumers).

### Phase 7 — Hardening & production readiness

Constraints this phase is designed around (verified 2026-07):
- **Lumerical cannot run on GitHub-hosted runners** — no license server access, no display, no
  install media. All hosted-CI Lumerical coverage is mock/replay; real runs need a self-hosted
  runner on a licensed machine (WP7.5).
- **Every real tidy3d run costs FlexCredits** (minimum ≈ 0.025 FC even for tiny sims), and
  `td.web.estimate_cost(task_id)` returns the max charge *before* running — so cloud tests are
  budget-guarded and manual/scheduled, never per-PR (WP7.5).
- **GitHub GPU runners are GA** (`gpu-t4-4-core`, T4 16 GB, ≈$0.07/min, billed always) — a
  weekly ~10-min GPU physics job costs under $1, viable for beamz/fdtdz smoke tests (WP7.5).

Scheduling: WP7.1 and WP7.2 can start any time after Phase 3 and run in parallel with Phases
4–5. WP7.3 needs WP3.1. WP7.4–7.6 slot between Phase 5 and WP6.3.

---
**WP7.1 [INFRA] Test taxonomy & maximum honest coverage** — needs WP3.1
*Goal:* make "how much is tested" a designed quantity, not an accident, under the
no-license/no-credits constraint.
*Do:*
1. **Marker taxonomy** in `pyproject.toml` `[tool.pytest.ini_options].markers`:
   `unit` (default, no marker needed), `physics` (runs a real solver binary — meep/beamz),
   `gpu`, `cloud` (spends credits), `licensed` (needs lumapi), `slow`.
   Default CI run = `pytest -m "not physics and not gpu and not cloud and not licensed"`.
2. **Solver conformance suite** `tests/conformance/test_solver_contract.py`: one parametrized
   test class run against *every* registered solver (`gds_fdtd.available_solvers()`):
   constructor is side-effect-free (no files created, no sockets — assert via tmp cwd +
   `socket` monkeypatch), `validate()` returns list[str], `build()` is deterministic (two calls
   → equal artifacts), artifacts serialize/deserialize, `estimate()` returns without network,
   `capabilities` complete. New solvers inherit ~30 tests for free.
3. **Mock lumapi** `tests/mocks/lumapi.py`: records every `eval`/`set`/`addport` call into a
   script transcript; `sys.modules["lumapi"]` injection fixture. Assert full generated setup
   transcript for the crossing fixture against a golden transcript file.
4. **Recorded cloud artifacts (replay testing):** commit tiny real result files produced once
   by a human: a `ModalComponentModelerData`/smatrix HDF5 from one minimal tidy3d run and a
   real Lumerical `.dat` + solver log. Tests exercise the *entire* results pipeline
   (native → SMatrix → export → checks) from these files with zero network. Document the
   refresh procedure in `tests/recorded/README.md` (who, how, expected cost ≈0.025 FC).
   **Mandatory sanitization before any commit:** run a scrub script over every recorded
   artifact stripping lines matching `ANSYS_LICENSING_CONTEXT|licens|HANDLER|hostname|user|@`
   (real Lumerical logs embed a base64 license token — one exists in this repo's untracked
   `examples/notebooks/faid/ID_sparams/` logs today); acceptance requires
   `grep -riE 'licens|handler|api[_-]?key|token' tests/recorded/` → empty, and WP7.2's
   secret-scanning/pre-commit hooks must cover `tests/recorded/`.
5. **Coverage policy:** `[tool.coverage]` with `branch = true`;
   `exclude_also` for `pragma: no cover — requires <license|cloud|gpu>` (the ONLY sanctioned
   exclusion pattern, greppable); codecov `project` and `patch` status checks required;
   set `fail_under` to the **measured honest coverage at merge time, rounded down** (record the
   number and the measuring command in the PR — do NOT hard-code a target the current suite
   can't meet, which is exactly the gaming incentive that produced B18); ratchet +1 per release
   toward 92, never down.
6. **Lower-bounds job:** weekly CI job installing floors via
   `uv pip install --resolution lowest-direct -e .[dev]` — catches "works only on latest dep"
   rot in both directions.
7. Optional (create issue, don't build now): mutation testing (`mutmut`) as a monthly job on
   `geometry.py`/`smatrix.py` only.
*Accept:* default `pytest` passes with no solver binaries installed beyond klayout;
conformance suite green for tidy3d (offline parts), lumerical (mocked), meep, beamz, fdtdz
(as each lands); coverage gate enforced on PRs; replay tests cover ≥90% of each solver's
results-processing code.
*Do NOT:* mark anything `no cover` outside the three sanctioned pragmas; skip tests to hit the
gate.

---
**WP7.2 [INFRA] Supply-chain & security hardening** — needs WP0.2 (parallel-safe)
**OWNER ACTIONS:** items 4's repo settings (branch protection/ruleset, merge queue, secret
scanning) and the Scorecard score depend on settings only the owner can change — the
executor's acceptance for those items is "hand-off checklist documented and delivered", not
the score itself.
*Do:*
1. **Workflows:** add CodeQL (python) on PR + weekly; `pip-audit` (or `uv audit`) job;
   dependency-review action on PRs; OpenSSF Scorecard workflow + badge; `zizmor` (GitHub
   Actions workflow linter) since workflows are now numerous.
2. **Actions hygiene:** pin every third-party action by commit SHA; set top-level
   `permissions: contents: read` in all workflows, escalating per-job only where needed;
   `concurrency` groups to cancel superseded PR runs.
3. **Releases (extends WP0.6):** attach an SBOM (`cyclonedx-py`) to each GitHub Release
   alongside the WP0.6 attested dists; release notes come from `.github/release.yml` label
   categories — no more aspirational changelogs (see Part 2).
4. **Repo settings checklist** (documented in `CONTRIBUTING.md`, applied by owner): branch
   protection (or ruleset) on `main` — required check: the single `pass` fan-in job from
   WP0.2, plus CodeQL; linear history; **enable the merge queue** with `pass` as its gate;
   secret scanning + push protection, CODEOWNERS (`* @mustafacc`), tag protection `v*`.
5. `SECURITY.md` (report channel, supported versions), `CONTRIBUTING.md` (dev setup via uv +
   prek, justfile tasks, WP workflow, marker taxonomy), `CODE_OF_CONDUCT.md`,
   `.github/PULL_REQUEST_TEMPLATE.md`, and an **`AGENTS.md`** (2026-standard, tidy3d has one):
   build/test commands, marker taxonomy, "never run `cloud`/`licensed` tests", pointers to this
   plan's Part 0 rules — so AI contributors behave.
*Accept:* Scorecard ≥ 7.0; all actions SHA-pinned (`zizmor` clean); a release dry-run produces
signed, attested, SBOM-attached artifacts; merge queue active with `pass` gating.

---
**WP7.3 [FEATURE] JobSpec + ExecutionBackend + CLI (remote-compute readiness)** — needs WP3.1
*Goal:* every simulation is a serializable job; every platform (laptop, SLURM cluster with a
Lumerical license, Modal GPU function, AWS Batch, Lambda-Labs box) is just a backend.
*Do:*
1. `execution/jobspec.py`:
   ```python
   class JobSpec(BaseModel):
       component: Component            # pydantic-serializable after Phase 2
       technology: Technology
       spec: SimulationSpec
       solver: str                     # registry name
       solver_options: dict[str, Any] = {}
       budget: Budget | None = None    # max_flexcredits / max_wall_seconds — enforced by run
   class JobResult(BaseModel):
       smatrix_path: Path              # HDF5 (WP2.4 format)
       job_hash: str                   # canonical hash (reuses WP5.5 cache key)
       solver: str; solver_version: str
       wall_seconds: float; cost: CostReport | None
       log_path: Path | None
   ```
   Round-trip law (tested): `JobSpec.model_validate_json(js.model_dump_json())` reconstructs a
   job that `build()`s identical artifacts.
2. `execution/backends.py`: `ExecutionBackend` protocol —
   `submit(job) -> JobHandle`, `status(handle)`, `result(handle) -> JobResult`, `cancel(handle)`.
   Implement `LocalBackend` (in-process) and `SubprocessBackend` (spawns
   `gds-fdtd run job.json --out dir/` — proves the serialization boundary and gives crash
   isolation + parallelism for sweeps). Convergence (WP5.5) and multi-excitation orchestration
   route through a backend so parallelism/remoting is free.
3. `cli.py` (`[project.scripts] gds-fdtd = "gds_fdtd.cli:main"`), stdlib `argparse`:
   `gds-fdtd validate job.json` · `build` (writes artifacts dir) · `estimate` ·
   `run job.json --out dir/ [--backend local|subprocess]` · `convert results.dat --to snp` ·
   `solvers` (table: name, available?, why not, capabilities). Exit codes: 0 ok, 2 validation
   failed, 3 solver unavailable, 4 budget exceeded. All output to stdout as text or
   `--json`.
4. Credentials policy: backends/solvers read secrets ONLY from env
   (`TIDY3D_API_KEY`, `LUMERICAL_LICENSE_FILE`, …) — never from JobSpec (it gets serialized and
   shipped); document in CLI help.
5. Write `docs/remote_compute.md` with **reference snippets** (not shipped code) for Modal
   (`@app.function(image=..., gpu="T4")` wrapping `run_job`), AWS Batch (container + job-queue
   JSON), and SLURM (`sbatch` template calling the CLI — this is how university clusters with
   Lumerical licenses run it). These stay docs until demand justifies `gds-fdtd[modal]` etc.
*Accept:* round-trip law test; `SubprocessBackend` runs a FakeSolver job end-to-end in CI;
CLI covered by `pytest` + `subprocess` tests incl. exit codes; a convergence sweep runs N jobs
through SubprocessBackend in parallel and matches LocalBackend results.
*Do NOT:* add modal/boto3/ray dependencies; build a job database/queue — handles are in-memory
for 1.0.

---
**WP7.4 [INFRA] Containers & multi-platform distribution** — needs WP7.3
*Do:*
1. **OS/Python matrix:** extend CI tests to `ubuntu-latest`, `macos-latest` (arm64),
   `windows-latest` × 3.11–3.13 for the core suite (klayout, shapely, klayout wheels cover all
   three — verify at setup; mark klayout-text-encoding-sensitive tests accordingly).
2. **Images** (multi-stage, uv-based, non-root user, `ghcr.io/siepic/gds_fdtd`):
   `:core` (python:3.12-slim + gds_fdtd, ~small; runs klayout/geometry/smatrix/CLI),
   `:meep` (micromamba + pymeep + gds_fdtd) and `:gpu` (CUDA runtime + jax[cuda] + beamz +
   fdtdz + gds_fdtd). Build+push on tag; `docker run ghcr.io/...:core gds-fdtd solvers` is the
   image smoke test in CI. These images are exactly what Modal/AWS Batch/RunPod consume — the
   deployability story and the remote-compute story are the same artifact.
3. **conda-forge feedstock** for `gds-fdtd` (photonics users live in conda because of MEEP);
   grayskull-generated recipe; document maintenance duty.
4. `py.typed` marker + `mypy --strict` on `geometry.py`, `technology.py`, `smatrix.py`,
   `spec.py`, `execution/` (the typed public core); documented support policy (SPEC 0-style:
   support latest 3 Python minors, drop on schedule).
*Accept:* 9-cell OS×Python matrix green; three images published and smoke-tested; conda-forge
PR submitted; `mypy --strict` clean on the listed modules.

---
**WP7.5 [INFRA] Licensed & cloud solver test strategy** — needs WP7.1
*Goal:* real-solver confidence at a known, capped cost — answering "can't launch Lumerical on
GitHub, tidy3d burns credits".
*Do:*
1. **Tidy3D cloud smoke** (`.github/workflows/cloud-smoke.yml`): `workflow_dispatch` +
   pre-release trigger only; environment `cloud-tests` with required reviewer (human approves
   every run); uses `TIDY3D_API_KEY` secret. Script: build the tiny straight-waveguide modeler →
   `td.web.estimate_cost` per task → **abort if total estimate > $BUDGET_FC (default 0.5)** →
   run → assert |S21|² > 0.8 → upload SMatrix as artifact → post cost summary to the job
   summary. Also refreshes the WP7.1 recorded artifacts when a `refresh_fixtures` input is set.
2. **Lumerical self-hosted lane:** document + implement a workflow targeting
   `runs-on: [self-hosted, lumerical]` that is **skipped unless the label exists** (repo works
   fine without it). Recipe in `docs/self_hosted_runner.md`: lab machine with license,
   runner as a service, `xvfb-run` for headless GUI-less FDTD, nightly schedule, uploads real
   `.dat` artifacts that CI replay tests then consume. (SiEPIC/UBC lab machines are the
   natural host.)
3. **GPU lane:** weekly job on `gpu-t4-4-core` hosted runner (≈$0.07/min, expect <10 min):
   beamz straight-waveguide on CUDA jax + fdtdz kernel smoke; also runnable via
   `workflow_dispatch`. Keep the CPU-jax beamz test in the normal matrix so GPU lane is
   additive confidence, not sole coverage.
4. **Badges/reporting:** each lane posts to a `SOLVER_STATUS.md` (or shields endpoint) so the
   README shows per-solver "last verified: date + version" — the honest version of a build
   badge for solvers CI can't run per-PR.
*Accept:* cloud smoke runs once end-to-end within budget (document actual FC spent); GPU lane
green once; lumerical lane merged in skipped state with docs; per-solver status surfaced in
README.

---
**WP7.6 [INFRA] Production polish: errors, retries, config, ops** — needs WP7.3
*Do:*
1. `errors.py` hierarchy (see Part 4 tree); sweep the codebase: every `raise ValueError` in
   user-input paths becomes the specific subclass; every solver adapter wraps native errors in
   `SolverError` with the native message attached (`raise ... from e`). CLI maps hierarchy →
   exit codes.
2. **Retries:** cloud/web calls (tidy3d upload/run/download) wrapped in `tenacity` retry
   (exponential backoff, max 5, retry only on transient classes — connection/5xx/timeout;
   NEVER retry a task submission that may have succeeded: make submission idempotent first via
   deterministic `task_name = f"gdsfdtd-{job_hash[:12]}"`). New dep `tenacity` (tiny, stable).
3. **Config:** one `GdsFdtdSettings` (pydantic-settings, env prefix `GDS_FDTD_`):
   cache_dir, default budget, log level, telemetry opt-in (default OFF; if enabled, logs only
   anonymous solver-name + duration locally — no network telemetry in 1.0; the setting exists
   so the policy is explicit).
4. **Structured logging option:** `GDS_FDTD_LOG_FORMAT=json` switches handlers to JSON lines
   (essential for Modal/AWS log aggregation); plain text stays default.
5. **Docs ops:** covered by WP6.2 (Read the Docs versioning + PR previews; Sybil/doctest over
   snippets so docs can't lie) — verify both are live before closing this WP.
*Accept:* `grep -rn "raise ValueError" src/gds_fdtd/solvers/` → zero (all wrapped); retry
tests with a flaky-mock web client (fails twice, succeeds third, idempotent task name
asserted); JSON log smoke test; docs snippets executed in CI.

### Reviewer prompt template (run after every WP)

> You are reviewing PR for WP<id>. Its card says: <paste card>. Check out the diff. Verify:
> (1) only listed files (plus tests) changed; (2) each acceptance criterion — run the commands
> yourself; (3) no test was deleted or weakened except where the card says so; (4) goldens:
> if changed, is the PR's justification specific and correct? (5) run
> `ruff check . && pytest -q tests`. Reply APPROVE or a numbered list of violations. Do not
> suggest improvements beyond the card's scope.

---

## Part 6 — Testing strategy (summary)

Test tiers, from free to expensive — maximize coverage at each tier before reaching for the
next (markers from WP7.1):

| Tier | Marker | What it covers | Mechanism | Runs where | Cost |
|---|---|---|---|---|---|
| 1 | *(none)* | geometry, technology, smatrix, spec, CLI, execution | unit tests + golden JSON fixtures (WP0.3) | every PR, 3 OS × 3 Pythons | free |
| 2 | *(none)* | every solver's `validate()`/`build()`/`estimate()` | **offline artifact assertions** (td.Simulation json, .lsf transcript vs mock lumapi, meep objects, raster grids) + conformance suite (WP7.1) | every PR | free |
| 3 | *(none)* | full results pipeline per solver | **replay of recorded real artifacts** (tidy3d smatrix HDF5, Lumerical .dat+log) (WP7.1) | every PR | free (one-time ≈0.025 FC + one licensed run to record) |
| 4 | `physics` | real physics, open solvers | MEEP (conda job) + beamz (CPU-jax) straight waveguide, \|S21\|²>0.8 | weekly + on solver-file change | free, slow |
| 5 | `gpu` | CUDA paths (beamz, fdtdz kernel) | GitHub `gpu-t4-4-core` runner (WP7.5) | weekly, ~10 min | ≈$0.70/run |
| 6 | `cloud` | tidy3d end-to-end | budget-guarded (`estimate_cost` → abort > 0.5 FC), human-approved environment (WP7.5) | pre-release + manual dispatch | ≈0.025–0.5 FC |
| 7 | `licensed` | Lumerical end-to-end | self-hosted runner on licensed lab machine, nightly, skipped if absent (WP7.5) | opportunistic | license already owned |
| — | — | examples | AST import check (WP0.4) + build-path execution | every PR | free |
| — | — | dependency envelope | lower-bounds resolve job (`--resolution lowest-direct`) + latest | weekly | free |

Per-solver trust is surfaced honestly: README shows "last verified" date+version per solver
from the tier 5–7 lanes instead of pretending a green badge covers them (WP7.5.4).

The historical failure mode was *silent rot between bursts of work* — offline `build()`
testability (rule 10), the conformance suite, replay artifacts, and the examples check are the
structural defenses.

## Part 7 — Backlog (post-1.0, do not start without owner approval)

- **fdtdx adapter** (MIT, JOSS 2025, multi-GPU autodiff) on the Tier B machinery; its autodiff
  opens inverse-design workflows (out of scope for 1.0).
- Compact-model export beyond .dat/.snp (IPKISS/Luceda, SAX-compatible model factories).
- gdsfactory `LayerStack` ⇄ `Technology` bidirectional adapter; gplugins interop shims.
- Litho-aware pipeline: prefab prediction as a `Component → Component` transform stage.
- EME/FDE engines (different physics — new `Solver` tier or sibling ABC).
- GUI/notebook widgets; KLayout SALT package for one-click sim from the editor.

## Part 8 — Open decisions for the owner (answer before Phase 2)

1. **API break at 1.0 with one-release shims** — confirm SiEPIC-Tools / downstream users can
   absorb it (repo lives under the SiEPIC org).
2. **pydantic v2 as a core dep** — recommended (tidy3d & gdsfactory already require it).
3. **Package name stays `gds_fdtd`** — recommended despite the FDTD-specific name; rebrand cost
   > benefit (fifth rename in project history would be on-brand, but no).
4. **`examples/notebooks/faid/`** — personal research content; move out or keep? (Also 2.6 GB of
   local untracked hdf5 — add `make clean-artifacts`.)
5. **License check for MEEP adapter** — MEEP is GPL; an *optional adapter* importing meep at
   runtime is the standard pattern, but confirm you're comfortable (adapter code itself stays
   MIT/this repo's license).
6. **GPU CI budget** — approve the weekly `gpu-t4-4-core` lane (≈$0.07/min ⇒ <$5/month at one
   ~10-min run/week); CPU-jax covers beamz functionally in the meantime.
7. **Cloud test budget & secrets** — approve storing `TIDY3D_API_KEY` as a repo secret behind a
   reviewer-gated environment, and the 0.5 FlexCredit per-run cap (WP7.5). Who are the approved
   reviewers for cloud runs?
8. **Self-hosted Lumerical runner** — is there a SiEPIC/UBC lab machine with a license that can
   host a nightly GitHub runner (WP7.5.2)? If not, Lumerical stays at tier 2–3 (mock + replay)
   and the README says so.
9. **Distribution surface** — confirm appetite for maintaining: ghcr.io Docker images (3
   variants), a conda-forge feedstock (review duty on bot PRs), and Read the Docs — each is
   ongoing maintenance, not one-time (WP7.4, WP6.2).
10. **Remote backends beyond docs** — Modal/AWS Batch/SLURM ship as documented recipes at 1.0
    (WP7.3.5). Promote any of them to a maintained extra (`gds-fdtd[modal]`) only on real user
    demand — which one, if any, do you want first-class?
11. **Docs hosting: Read the Docs vs GitHub Pages** — the plan recommends RTD (PR previews,
    `stable`/`latest` versioning, search, zero deploy code; kills the deployment noise for
    good). WP0.5 fixes Pages properly in the interim; approve the RTD move at WP6.2 and whether
    `siepic.github.io/gds_fdtd` becomes a redirect.
12. **Zenodo + JOSS** — enabling Zenodo DOI minting needs the org owner's authorization; JOSS
    submission at v1.0 needs an author list & affiliations decision (WP6.2b).
13. **Technology YAML schema v2** — the v1 format (incl. `tidy3d_db`/`lum_db` key names) is
    frozen through 1.x (WP2.2). Approve (or reject) a v2 schema that generalizes per-solver
    material hints into neutral `solver_hints`, with a converter and a deprecation window —
    this is the product's public contract; changing it is a bigger decision than any code
    refactor.
