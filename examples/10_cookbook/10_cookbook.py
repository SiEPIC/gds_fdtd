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
# # 10 · Cookbook — the Si→SiN escalator on the *free* engine
#
# A **cookbook** of reference devices with known-good S-parameters. The first
# entry is the **Si→SiN escalator**: a silicon waveguide that hands its light
# *up* to a silicon-nitride waveguide sitting 0.3 µm above it — a **vertical,
# adiabatic mode transfer** used to route between the Si and SiN layers of a
# photonic stack.
#
# It is the toolbox's signature **multi-layer** device: **two** patterned device
# layers (Si at z 0–0.22 µm, SiN at z 0.3–0.7 µm), not one. The whole point of
# this notebook is that the geometry is simple, so the **free** engine
# ([beamz](https://github.com/beamzorg/beamz), Apache-2.0 JAX FDTD, CPU/GPU) runs
# it end-to-end — and reproduces the same device that the commercial tidy3d and
# Lumerical engines recorded, to within ~0.1 dB on the through path.
#
# Same three inputs → one output, exactly as everywhere else:
# a `Component` + a `Technology` + a `SimulationSpec` → one `SMatrix`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.plotting import plot_component, plot_permittivity, plot_smatrix, plot_tech_stack
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.smatrix import SMatrix
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology


def _find(rel: str) -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


HERE = _find("examples/10_cookbook")
# the SHARED technology — it already defines the Si + SiN + SiO₂ vertical stack
# (Si core [1,0] @ z 0–0.22 µm, SiN core [4,0] @ z 0.3–0.7 µm). No escalator-only
# tech needed; the GDS uses those same layers.
tech = Technology.from_yaml(_find("examples/tech.yaml"))

# %% [markdown]
# ## 1 · The technology — a two-core vertical stack
#
# The escalator uses the **shared** `examples/tech.yaml` — no device-specific
# technology needed. It already defines **two** device layers: `Si` (n≈3.48) is
# the lower core; `SiN` (n≈2.0) is the upper core, 0.3 µm above it; both are clad
# in `SiO₂` (n≈1.44). Each material carries a neutral `nk` so the free beamz engine
# can resolve indices offline, plus per-engine hints (`tidy3d`, `lumerical`) for
# the commercial solvers — **one technology file, every engine**.

# %%
plot_tech_stack(tech, wavelength_um=1.55)
plt.show()

# %% [markdown]
# ## 2 · The device — two patterned cores
#
# We load `si_sin_escalator.gds` (KLayout/SiEPIC) and pair it with the technology
# to build a `Component`. Ports are auto-detected from the pin layer: **opt1** on
# the Si layer (west, z≈0.11 µm) and **opt2** on the SiN layer (east, z≈0.5 µm).
# Passing a `SimulationSpec` also sketches the **FDTD region** (device bounds + a
# buffer) and the **port-extension stubs** that carry each port out through the
# PML. It's a schematic of the setup — the exact margins are engine-specific
# (beamz uses its own fixed guard band); the full run spec is defined in §3.

# %%
cell, _layout = load_cell(str(HERE / "si_sin_escalator.gds"))
component = load_component_from_tech(cell=cell, tech=tech)
component.name = "si_sin_escalator"
print("ports:")
for p in component.ports:
    print(f"  {p.name}: center={[round(float(c), 3) for c in p.center]} µm, dir={p.direction}°")

plot_component(component, spec=SimulationSpec())
plt.show()

# %% [markdown]
# The top-down view collapses the stack. The refractive-index cross-section
# **along** the waveguide (the x–z plane the solver actually rasterizes) shows the
# two cores directly: the **Si** core enters from the left and tapers away, while
# the **SiN** core rises above it and carries the light out to the right — the
# "escalator."

# %%
plot_permittivity(component, axis="y", position=0.0, wavelength_um=1.55)
plt.show()

# %% [markdown]
# ## 3 · Set up and run — on the free engine
#
# `validate()` / `build()` / `estimate()` are **offline and free** — preview the
# whole job before spending anything. beamz then runs one FDTD per port
# excitation (two here) on the CPU. (For repeat runs, `run_cached(cache_dir)`
# hashes the job and reloads the stored `SMatrix` — see
# `06_convergence_and_caching`.)

# %%
from gds_fdtd.solvers import get_solver  # noqa: E402

spec = SimulationSpec(
    wavelength_start=1.5, wavelength_end=1.6, wavelength_points=11, mesh=6, z_min=-1.0, z_max=1.11
)
solver = get_solver("beamz")(component, technology=tech, spec=spec)

print("validate():", solver.validate() or "OK — no problems")
print("build():   ", solver.build().summary)
print("estimate():", solver.estimate())

# %%
smatrix = solver.run()  # one FDTD per port; ~1-2 min on a CPU
print("S-matrix:", smatrix.port_names, "| modes:", smatrix.n_modes, "| freqs:", smatrix.f.size)

# %% [markdown]
# ## 4 · S-parameters
#
# The through path `opt1→opt2` should be near **0 dB** (an efficient adiabatic
# transfer) with deep back-reflection at `opt1`.

# %%
plot_smatrix(smatrix, kind="db")
plt.show()


def thru(sm: SMatrix) -> np.ndarray:
    return sm.magnitude_db(out=sm.port_names[1], in_=sm.port_names[0])


def refl(sm: SMatrix) -> np.ndarray:
    return sm.magnitude_db(out=sm.port_names[0], in_=sm.port_names[0])


back = smatrix.magnitude_db(out=smatrix.port_names[0], in_=smatrix.port_names[1])
print(f"beamz through opt1→opt2: {float(np.nanmax(thru(smatrix))):+.2f} dB peak")
print(f"beamz through opt2→opt1: {float(np.nanmax(back)):+.2f} dB peak  (≈ opt1→opt2: near-reciprocal)")
print(f"beamz reflection opt1:   {float(np.nanmax(refl(smatrix))):+.2f} dB peak")
print("passive:", smatrix.is_passive(atol=0.05))

# %% [markdown]
# ## 5 · The field — light climbing the escalator
#
# The frequency-domain `|E|²` profile at the Si-core plane: the mode enters on the
# Si waveguide (left) and its energy leaves the Si plane as it transfers up into
# the SiN core — the vertical hand-off in action.

# %%
solver.plot_fields(axis="z")
plt.show()

# %% [markdown]
# ## 6 · Cross-validation — free vs commercial
#
# The payoff. **beamz is the only engine executed in this notebook.** The tidy3d
# and Lumerical curves are **pre-recorded reference results** (`recorded/`, see
# `PROVENANCE.md`) from an earlier live validation on those engines for the
# *identical* device — they are *not* re-run here (that needs cloud credits / a
# license). Overlaying the free beamz result on them: the three engines agree on
# the through transfer to within ~0.1–0.2 dB, and all show deep reflection.

# %%
ref_tidy3d = SMatrix.from_npz(str(HERE / "recorded" / "si_sin_escalator_tidy3d.npz"))
ref_lumerical = SMatrix.from_npz(str(HERE / "recorded" / "si_sin_escalator_lumerical.npz"))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for sm, label, style in [
    (smatrix, "beamz (free)", "o-"),
    (ref_tidy3d, "tidy3d", "--"),
    (ref_lumerical, "Lumerical", "--"),
]:
    ax[0].plot(sm.wavelength_um, thru(sm), style, label=label)
    ax[1].plot(sm.wavelength_um, refl(sm), style, label=label)
ax[0].set_title("through  opt1 → opt2  (Si→SiN transfer)")
ax[1].set_title("reflection  opt1")
for a in ax:
    a.set_xlabel("wavelength [µm]")
    a.set_ylabel("|S| [dB]")
    a.grid(alpha=0.3)
    a.legend()
fig.suptitle("Si→SiN escalator — the free beamz engine reproduces tidy3d / Lumerical")
fig.tight_layout()
plt.show()

# %%
print(f"{'wavelength':>10} | {'beamz':>8} {'tidy3d':>8} {'lumerical':>10}   (through, dB)")
grid = ref_tidy3d.wavelength_um
for i, wl in enumerate(grid):
    b = float(np.interp(wl, smatrix.wavelength_um[::-1], thru(smatrix)[::-1]))
    print(f"{wl:>10.3f} | {b:>8.2f} {float(thru(ref_tidy3d)[i]):>8.2f} {float(thru(ref_lumerical)[i]):>10.2f}")

# %% [markdown]
# ## Recap & next
#
# - The Si→SiN escalator is a **multi-layer** device (two cores at different z);
#   `get_solver("beamz")` builds and runs it on the free engine — no cloud
#   account, no license.
# - Through transfer ≈ 0 dB with deep reflection, **matching recorded tidy3d and
#   Lumerical** to ~0.1–0.2 dB: physics you can trust from a zero-cost solver.
# - Same pattern as everywhere: `Component` + `Technology` + `SimulationSpec` →
#   `SMatrix`. See **`07_choosing_an_engine`** for the identical-job three-engine
#   comparison on a single-layer device, and **`04_reading_results`** for the full
#   `SMatrix` analysis surface.
