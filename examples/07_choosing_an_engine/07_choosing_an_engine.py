# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 · Choosing an engine — three solvers, one job
#
# The whole promise of gds_fdtd is that **one** `(component, technology, spec)`
# runs on **any** engine and returns the **same** `SMatrix`. This notebook puts
# that to the test: we take a single device and run it on all three engines —
#
# | Engine | Where it runs | Cost | Good for |
# |--------|---------------|------|----------|
# | **beamz** | your CPU/GPU (JAX) | free (Apache-2.0) | prototyping, no account, full control |
# | **tidy3d** | the cloud | FlexCredits | fast, broadband, big jobs, no local hardware |
# | **Lumerical** | your workstation | a license seat | the foundry-standard reference |
#
# — and ask the only question that matters: **do they agree?**
#
# The testbed is the **SiEPIC EBeam `ebeam_y_1550`** 1×2 y-branch, loaded
# straight from the KLayout/SiEPIC PDK — *no gdsfactory anywhere*. That's the
# point: the same solver-agnostic `Component` drives every engine, and beamz
# now reads the KLayout-sourced polygons through a component shim just like the
# others.
#
# > **These results are recorded artifacts.** Each engine was run once
# > (`solver.run_cached(...)`) and the resulting `SMatrix` saved under
# > `recorded/`. This notebook *loads* them, so it re-executes in seconds, for
# > free, on any laptop — nothing here spends a credit, a license, or much
# > memory. See `recorded/PROVENANCE.md` for versions, date, and cost.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gds_fdtd.smatrix import SMatrix
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology
from gds_fdtd.validation import compare_smatrices


def _find(rel: str) -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


HERE = _find("examples/07_choosing_an_engine")
ENGINES = ("beamz", "tidy3d", "lumerical")

# %% [markdown]
# ## 1 · The device
#
# `ebeam_y_1550` is the canonical silicon TE-1550 y-branch: one input (`opt1`,
# facing −x) splits into two outputs (`opt2`, `opt3`, facing +x). We load it
# with the KLayout reader — the identical geometry every engine will see. If
# you don't have the SiEPIC PDK installed (`pip install siepic_ebeam_pdk`), the
# geometry view is skipped and we go straight to the recorded S-parameters.

# %%
tech = Technology.from_yaml(_find("examples/tech.yaml"))
component = None
try:
    import os

    import siepic_ebeam_pdk as pdk

    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.simprocessor import load_component_from_tech

    gds = os.path.join(os.path.dirname(pdk.__file__), "gds", "EBeam", "ebeam_y_1550.gds")
    cell, _layout = load_cell(gds, top_cell="ebeam_y_1550")
    component = load_component_from_tech(cell=cell, tech=tech)
    component.name = "ebeam_y_1550"
    print(f"{component.name}: ports {[p.name for p in component.ports]}")
except Exception as e:  # noqa: BLE001 - geometry is optional for the comparison
    print(f"(SiEPIC PDK not available — skipping geometry views: {e})")

# %% [markdown]
# ## 2 · Geometry and stack the engines share
#
# The top-down view (ports, devrec bounds, FDTD region) and the vertical
# material stack. Both are engine-agnostic — this is exactly what gets handed to
# beamz, tidy3d, and Lumerical alike.

# %%
from gds_fdtd.plotting import plot_component, plot_permittivity, plot_tech_stack  # noqa: E402

if component is not None:
    plot_component(component, spec=SimulationSpec(buffer=0.8))
    plt.show()
    plot_tech_stack(tech, wavelength_um=1.55)
    plt.show()
    # what the mesh actually resolves (coarse grid — just for the picture)
    plot_permittivity(component, dx=0.05, axis="z")
    plt.show()

# %% [markdown]
# ## 3 · Running each engine
#
# Running is the same three lines on every engine — only `run()` (here wrapped
# in `run_cached`, which stores the result so repeats are free) spends anything:
#
# ```python
# from gds_fdtd.solvers import get_solver
# spec = SimulationSpec(wavelength_start=1.5, wavelength_end=1.6,
#                       wavelength_points=5, mesh=6, z_min=-1.0, z_max=1.11, buffer=0.8)
# smatrices = {
#     name: get_solver(name)(component, tech, spec).run_cached(".gds_fdtd_cache")
#     for name in ("beamz", "tidy3d", "lumerical")
# }
# ```
#
# We kept it deliberately **small** — mesh 6, five wavelengths, a tight domain
# (~3 M cells, ~0.15 GB) — so the local engines stay well under a laptop's
# memory. Below we load the `SMatrix` each of those runs produced.

# %%
smatrices = {
    name: SMatrix.from_npz(str(HERE / "recorded" / f"ybranch_{name}.npz")) for name in ENGINES
}
for name, sm in smatrices.items():
    print(f"{name:9s}: {sm.n_ports} ports, {sm.wavelength_um.size} wavelengths")

# %% [markdown]
# ## 4 · Do they agree?
#
# A y-branch's figure of merit is its **input split**: how power from `opt1`
# divides into the two arms (`opt2`, `opt3`). An ideal lossless splitter gives
# −3.01 dB into each. This is the metric all three engines resolve cleanly, so
# it's the one we compare.

# %%
print("input split — mean transmission (dB):")
print(f"  {'engine':10s} {'opt2←opt1':>10s} {'opt3←opt1':>10s}")
for name, sm in smatrices.items():
    s21 = float(np.mean(sm.magnitude_db(out=2, in_=1)))
    s31 = float(np.mean(sm.magnitude_db(out=3, in_=1)))
    print(f"  {name:10s} {s21:>10.3f} {s31:>10.3f}")

# worst pairwise disagreement across engines on the split, per wavelength
worst = 0.0
for out in (2, 3):
    vals = [np.asarray(sm.magnitude_db(out=out, in_=1), float) for sm in smatrices.values()]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            worst = max(worst, float(np.max(np.abs(vals[i] - vals[j]))))
print(f"\n==> three engines agree on the split to {worst:.3f} dB (worst pair, any wavelength)")

# %% [markdown]
# The three independent FDTD engines — a free JAX kernel, a cloud solver, and a
# commercial one — land within a few hundredths of a dB of each other and of the
# −3 dB ideal. That agreement, on the identical job, is the whole thesis of the
# toolbox in one number.

# %%
# compare_smatrices().plot draws the overlay; its default title reports the
# whole-matrix worst pair, which is +inf here because beamz's dead opt3 column
# (see §5) reads -inf dB — so we title it with the trustworthy split metric.
compare_smatrices(smatrices).plot(out=2, in_=1)
plt.gca().set_title(f"opt2 ← opt1 — three engines agree to {worst:.3f} dB")
plt.show()

# %% [markdown]
# ## 5 · The physics behind the numbers — mode and field
#
# The S-parameters compress a lot of physics into a few numbers. Two views to
# keep them honest: the **mode** every engine launches into the 0.5 µm input
# guide (solved offline, free), and the recorded tidy3d **field** through the
# device — the −3 dB split as an actual picture.

# %%
from gds_fdtd.grid import resolve_index  # noqa: E402
from gds_fdtd.modes import waveguide_mode  # noqa: E402
from gds_fdtd.plotting import plot_mode  # noqa: E402

if component is not None:
    n_core = resolve_index(tech.device[0].material, 1.55).real
    n_clad = resolve_index(tech.superstrate.material, 1.55).real
    mode = waveguide_mode(component.ports[0].width, tech.device[0].z_span, n_core, n_clad, 1.55)[0]
    plot_mode(mode)
    plt.show()

fld = np.load(_find("examples/03_first_simulation/recorded/ybranch_tidy3d_field.npz"))
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.pcolormesh(fld["x"], fld["y"], fld["mag2"].T, shading="nearest", cmap="RdBu_r")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, label="|E|²")
ax.set_xlabel("x [µm]")
ax.set_ylabel("y [µm]")
ax.set_title("y-branch |E|² — the −3 dB split as a field (tidy3d, recorded)")
plt.show()

# %% [markdown]
# ## 6 · An honest caveat (beamz v1)
#
# Cross-engine benchmarks are only useful if you report what you *can't* trust.
# At this coarse mesh the reflections (`S11` ≈ −25 dB) sit near each engine's
# numerical floor, and **beamz v1 intermittently under-injects one of the two
# same-direction output ports** — its mode source underflows, forcing that whole
# S-column to zero (visible below as the dead `opt3`-excitation column). It's a
# beamz engine-level mode-injection issue, *not* the gds_fdtd adapter — the
# geometry is correct and tidy3d/Lumerical resolve every column. The input-split
# metric above uses only the `opt1` excitation, which always injects, so it's
# unaffected.

# %%
b = smatrices["beamz"]
print("beamz |S| by excitation (mean over wavelength):")
for i in (1, 2, 3):
    row = [round(float(np.mean(np.abs(b.sel(out=o, in_=i)))), 3) for o in (1, 2, 3)]
    flag = "  <- underflowed to zero" if all(v == 0 for v in row) else ""
    print(f"  excite opt{i}: -> opt1/2/3 = {row}{flag}")

# %% [markdown]
# ## When to reach for which engine
#
# - **beamz** — start here. Free, local, no account; ideal for iterating on a
#   design and for CI. v1 is single-mode TE with x-facing ports.
# - **tidy3d** — when you want speed, broadband, big 3D jobs, or have no local
#   hardware. Cloud FlexCredits; `estimate()` shows the cost before you spend.
# - **Lumerical** — the foundry-standard reference; reach for it to sign off a
#   design against a commercial tool you already license.
#
# Because the API is identical, "choosing an engine" is a one-word change to
# `get_solver(...)` — validate on the free one, confirm on the reference.
#
# **Next:** `08_frontends` (gdsfactory / SiEPIC / PreFab) and `10_cookbook`
# (reference devices with known-good S-parameters).
