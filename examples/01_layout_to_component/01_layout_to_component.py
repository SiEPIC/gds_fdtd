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
# # 01 · From a GDS layout to a `Component`
#
# Every simulation starts with geometry. gds_fdtd reads a GDS cell against a
# technology and produces a **`Component`** — device polygons, a material stack,
# and **auto-detected ports** — the object every engine consumes. This first
# step is entirely offline (KLayout + matplotlib), no engine required.
#
# Here we load a waveguide **crossing** shipped with the examples.

# %%
from pathlib import Path

import matplotlib.pyplot as plt

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.plotting import plot_component
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology


def _find(rel: str) -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


# %% [markdown]
# ## 1 · Load the cell against a technology
#
# `load_cell` opens the GDS and picks the top cell; `load_component_from_tech`
# extrudes each layer's polygons to the z-heights and materials the technology
# defines, and detects ports from the pin/DevRec layers. The technology is what
# turns 2-D polygons into a 3-D, materialized device.

# %%
tech = Technology.from_yaml(_find("examples/tech.yaml"))
cell, layout = load_cell(str(_find("examples/devices.gds")), top_cell="crossing_te1550")
component = load_component_from_tech(cell=cell, tech=tech)
print(f"loaded {component.name!r}")

# %% [markdown]
# ## 2 · Ports — detected, never hand-placed
#
# A crossing has four ports. Each carries its own name, µm coordinates, width,
# and a snapped direction (0/90/180/270°) — everything an engine needs to launch
# and collect modes.

# %%
for p in component.ports:
    print(f"  {p.name}: center ({p.center[0]:.2f}, {p.center[1]:.2f}, {p.center[2]:.2f}) µm, "
          f"width {p.width:.3f} µm, facing {p.direction}°")

# %% [markdown]
# ## 3 · Structures — role-tagged, materialized
#
# The component is a flat list of structures, each tagged by **role** (device /
# substrate / superstrate), with its GDS layer, z-extent, and resolved material.
# Backgrounds (substrate/superstrate) are filled by the technology, not the GDS.

# %%
for s in component.structures:
    print(f"  {s.role:<11} {s.name:<10} layer {s.layer}  z {s.z_base:+.2f}…{s.z_base + s.z_span:+.2f} µm")
b = component.bounds
print(f"\nbounds: x [{b.x_min:.2f}, {b.x_max:.2f}]  y [{b.y_min:.2f}, {b.y_max:.2f}]  "
      f"z-center {b.z_center:.3f} µm")

# %% [markdown]
# ## 4 · The geometry view
#
# The standard picture: device polygons (per layer), the detected ports
# (arrows + width ticks), the `devrec` bounds, and — given a `SimulationSpec` —
# the FDTD region and the port-extension stubs the solvers push through the PML.
# You will start every simulation notebook with exactly this view.

# %%
plot_component(component, spec=SimulationSpec(buffer=1.0))
plt.show()

del layout  # release the KLayout objects

# %% [markdown]
# ## Recap & next
#
# One `Component` — geometry + materials + ports — is all any engine needs, and
# it came from a plain GDS file plus a technology. The rest of the toolbox does
# not care where it came from.
#
# - **`02_technology`** — the material side of that technology.
# - **`03_first_simulation`** — put this Component through an FDTD engine.
# - **`08_frontends`** — the same Component from gdsfactory / SiEPIC / raw GDS.
