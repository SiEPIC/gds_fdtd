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
# # 05 · Fields and modes
#
# Before you run a full FDTD simulation it helps to see two things: the
# **waveguide modes** (the transverse field patterns the guide supports, and
# their effective indices) and the **permittivity** the solver actually
# rasterizes from your layout + technology.
#
# Both are **local and free** — the mode solver is tidy3d's *local* plugin (pure
# on-CPU linear algebra, no cloud task, no credits), and the permittivity grid is
# the same rasterizer the kernel engines use. This whole notebook runs offline.

# %%
from pathlib import Path

import gdsfactory as gf
import matplotlib.pyplot as plt

from gds_fdtd.layout.gdsfactory import from_gdsfactory
from gds_fdtd.modes import waveguide_mode
from gds_fdtd.plotting import plot_mode, plot_permittivity
from gds_fdtd.technology import Technology


def _find(rel: str) -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


# %% [markdown]
# ## 1 · The fundamental mode of a silicon strip
#
# A 500 × 220 nm silicon core (n≈3.48) in oxide (n≈1.44) at 1.55 µm.
# `waveguide_mode` builds the cross-section and solves it; the first mode is the
# fundamental TE-like mode. Its effective index (~2.44) sets the phase velocity.

# %%
modes = waveguide_mode(
    width_um=0.5, height_um=0.22, n_core=3.476, n_clad=1.444, wavelength_um=1.55, n_modes=2
)
for i, m in enumerate(modes):
    print(f"mode {i}: n_eff = {m.n_eff.real:.4f}")

plot_mode(modes[0], field="E")  # total |E| of the fundamental mode
plt.show()

# %% [markdown]
# It's TE-like: the transverse **Ex** component dominates (in-plane
# polarization), while **Ey** is small.

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
plot_mode(modes[0], field="Ex", ax=ax[0])
plot_mode(modes[0], field="Ey", ax=ax[1])
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2 · Index contrast sets the mode
#
# The upper core of the escalator (`10_cookbook`) is silicon **nitride**
# (n≈2.0) — a much lower index, so its mode is larger and its effective index
# lower. Solving both explains why routing between them needs an adiabatic
# taper (their modes must be matched along the transition):

# %%
sin = waveguide_mode(width_um=1.0, height_um=0.4, n_core=1.997, n_clad=1.444, wavelength_um=1.55)
print(f"Si strip  n_eff = {modes[0].n_eff.real:.4f}")
print(f"SiN strip n_eff = {sin[0].n_eff.real:.4f}")
plot_mode(sin[0], field="E")
plt.show()

# %% [markdown]
# ## 3 · The permittivity the solver sees
#
# The mode above is analytic; a real run rasterizes your component + technology
# onto a grid. `plot_permittivity` shows that √ε cross-section — the same one the
# kernel engines (beamz, fdtdz) integrate. Here, a straight waveguide's
# core-in-cladding cross-section:

# %%
gf.gpdk.PDK.activate()
tech = Technology.from_yaml(_find("examples/tech.yaml"))
component = from_gdsfactory(gf.components.straight(length=3), tech)
plot_permittivity(component, axis="x", wavelength_um=1.55)
plt.show()

# %% [markdown]
# ## 4 · From mode to field — the propagating picture
#
# Everything so far was a *cross-section*: the transverse mode and the index it
# rides in. A run stitches those together into the **propagating field**. Here
# is the recorded tidy3d |E|² through the SiEPIC y-branch (the device of `03`
# and `07`): the TE0 mode from §1, launched at `opt1`, splitting into two —
# drawn on the solver's true (non-uniform) grid coordinates.

# %%
import numpy as np  # noqa: E402

fld = np.load(_find("examples/03_first_simulation/recorded/ybranch_tidy3d_field.npz"))
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.pcolormesh(fld["x"], fld["y"], fld["mag2"].T, shading="nearest", cmap="RdBu_r")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, label="|E|²")
ax.set_xlabel("x [µm]")
ax.set_ylabel("y [µm]")
ax.set_title("the mode in flight — y-branch |E|² (tidy3d, recorded)")
plt.show()

# %% [markdown]
# ## Recap & next
#
# `waveguide_mode` + `plot_mode` give you modes and effective indices offline for
# free; `plot_permittivity` shows the discretized index the solver integrates;
# and a run's field profile shows the mode actually propagating. Next:
# **`06_convergence_and_caching`** — how fine a mesh you actually need, and
# how to never pay for the same run twice.
