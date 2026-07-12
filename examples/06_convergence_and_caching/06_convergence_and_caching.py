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
# # 06 · Convergence, caching, and cross-validation
#
# Three questions decide whether you can trust an FDTD number:
#
# 1. **How fine a mesh do I need?** — `convergence.sweep` reruns a job while
#    stepping the mesh and measures how much the S-matrix still moves.
# 2. **How do I avoid paying for the same run twice?** — `run_cached` hashes the
#    whole job and reloads the stored result on a repeat.
# 3. **Is my *converged* answer even *correct*?** — the subtle one. A sweep tells
#    you when an engine has stopped changing, not whether it stopped at the right
#    value. The only way to know is to **cross-check a second engine**.
#
# §1–2 run on free **beamz** (a straight waveguide). §3 tackles question 3 on a
# hard device using **recorded** beamz + tidy3d + Lumerical results, so the
# whole notebook reproduces for free — no cloud account or license needed.

# %%
import json
import tempfile
import time
from pathlib import Path

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

from gds_fdtd.convergence import sweep
from gds_fdtd.layout.gdsfactory import from_gdsfactory
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.plotting import plot_component, plot_permittivity
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology


def _find(rel: str) -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


REC = _find("examples/06_convergence_and_caching/recorded")
tech = Technology.from_yaml(_find("examples/tech.yaml"))

# %% [markdown]
# ## 1 · How fine a mesh? — a convergence sweep
#
# A short straight waveguide, swept over three mesh densities on beamz.
# `sweep` returns a `ConvergenceReport`; `max |ΔS|` is the worst-case change,
# between successive meshes, of any S-parameter carrying real power
# (`floor_db=-10` keeps the metric on the through path, not the deep numerical
# reflection of a well-matched straight). A `cache_dir` means only genuinely new
# points ever cost a run.

# %%
gf.gpdk.PDK.activate()
straight = from_gdsfactory(gf.components.straight(length=1.5), tech)
spec = SimulationSpec(wavelength_start=1.5, wavelength_end=1.6, wavelength_points=3,
                      z_min=-0.6, z_max=0.8)
cache = Path(tempfile.mkdtemp(prefix="gdsfdtd_conv_"))
mesh_values = [4, 6, 8]
TOL_DB = 0.25  # engineering tolerance — convergence is always relative to it

t0 = time.perf_counter()
report = sweep(get_solver("beamz"), straight, tech, spec,
               field="mesh", values=mesh_values, cache_dir=cache, floor_db=-10.0)
cold = time.perf_counter() - t0

for lo, hi, d in zip(mesh_values, mesh_values[1:], report.deltas_db, strict=False):
    print(f"mesh {lo} → {hi}:  max |ΔS| = {d:.3f} dB")
rec = report.recommend(tol_db=TOL_DB)
print(f"\nrecommended mesh (tol {TOL_DB} dB): {rec}   ·   sweep wall time: {cold:.1f} s")

# %%
report.plot(tol_db=TOL_DB)
plt.show()

# %% [markdown]
# The change shrinks as the mesh refines — the coarsest grid is off by ~1 dB,
# then it settles below the tolerance. `recommend` returns the coarsest mesh
# that had already stopped moving: the one to use in production. The tolerance is
# *your* call — a ±0.25 dB spec converges here; a strict 0.05 dB would demand a
# finer sweep.

# %% [markdown]
# ## 2 · Repeats are free — caching
#
# Every point above was stored under a content hash. Re-running the **identical
# sweep** recomputes nothing; change the geometry, technology, spec, or engine
# version and only the genuinely new work reruns.

# %%
t0 = time.perf_counter()
again = sweep(get_solver("beamz"), straight, tech, spec,
              field="mesh", values=mesh_values, cache_dir=cache, floor_db=-10.0)
warm = time.perf_counter() - t0
print(f"cold sweep (new points): {cold:6.1f} s")
print(f"warm sweep (all cached): {warm:6.3f} s   →  {cold / max(warm, 1e-6):.0f}× faster")
print(f"identical result: {again.recommend(TOL_DB) == rec}")

# %% [markdown]
# ## 3 · Converged ≠ correct — cross-validate on a hard device
#
# A sweep only tells you an engine *stopped changing*. On a benign device that's
# enough; on a hard one it can plateau at the **wrong** value. Meet
# **`sbend_dontfabme`** (from `examples/devices.gds`) — a deliberately *sharp*
# S-bend that offsets the guide 0.5 µm in ~1 µm. A bend that tight strongly
# **converts the fundamental mode into higher-order modes + radiation**, so its
# true loss is large and it is a stress test for any solver.
#
# First, the geometry the solvers build — device + cladding + the port
# extensions that carry each port out through the domain edge:

# %%
sbend_cell, _ = load_cell(str(_find("examples/devices.gds")), top_cell="sbend_dontfabme")
sbend = load_component_from_tech(cell=sbend_cell, tech=tech)
sbend.name = "sbend_dontfabme"
plot_component(sbend, spec=SimulationSpec())
plt.show()
plot_permittivity(sbend, axis="z", position=0.11, wavelength_um=1.55)  # top-down √ε at the Si core
plt.show()

# %% [markdown]
# ### The convergence curves — beamz vs tidy3d vs Lumerical
#
# Single wavelength (1.55 µm), swept from low to high mesh on **all three**
# engines (recorded in `recorded/`; beamz on CPU, tidy3d on the cloud, Lumerical
# on a licensed workstation — its `mesh` maps to the nearest of Lumerical's
# discrete *mesh accuracy* settings 1–5). S21 on the left axis, S11 on the right.

# %%
beamz_c = json.loads((REC / "sbend_beamz_convergence.json").read_text())["mesh"]
tidy3d_c = json.loads((REC / "sbend_tidy3d_convergence.json").read_text())["mesh"]
lum_c = json.loads((REC / "sbend_lumerical_convergence.json").read_text())["mesh"]

fig, axL = plt.subplots(figsize=(8, 5))
axR = axL.twinx()
bm = sorted(int(m) for m in beamz_c)
tm = sorted(int(m) for m in tidy3d_c)
lm = sorted(int(m) for m in lum_c)
axL.plot(bm, [beamz_c[str(m)]["s21_db"] for m in bm], "o-", color="tab:blue", label="beamz S21")
axL.plot(tm, [tidy3d_c[str(m)]["s21_db"] for m in tm], "s--", color="tab:red", label="tidy3d S21")
axL.plot(lm, [lum_c[str(m)]["s21_db"] for m in lm], "^:", color="tab:green", label="Lumerical S21")
axR.plot(bm, [beamz_c[str(m)]["s11_db"] for m in bm], "o-", color="tab:blue", alpha=0.4,
         markerfacecolor="none", label="beamz S11")
axR.plot(tm, [tidy3d_c[str(m)]["s11_db"] for m in tm], "s--", color="tab:red", alpha=0.4,
         markerfacecolor="none", label="tidy3d S11")
axR.plot(lm, [lum_c[str(m)]["s11_db"] for m in lm], "^:", color="tab:green", alpha=0.4,
         markerfacecolor="none", label="Lumerical S11")
axL.set_xlabel("mesh (points per wavelength)")
axL.set_ylabel("S21  |through|  [dB]")
axR.set_ylabel("S11  |reflection|  [dB]")
axL.set_title("sbend_dontfabme convergence at 1.55 µm — beamz vs tidy3d vs Lumerical")
axL.grid(True, alpha=0.3)
_ln = axL.get_lines() + axR.get_lines()
axL.legend(_ln, [ln.get_label() for ln in _ln], loc="center right", fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# **tidy3d and Lumerical — two completely independent implementations — converge
# to the same answer**: S21 ≈ −5.6 dB (they agree within ~0.03 dB from Lumerical's
# second-coarsest setting onward) and S11 ≈ −27.7 dB (within ~0.1 dB).
# **beamz never converges** — its S21 *wanders* between about −4.7 and −2.0 dB
# and shows no trend toward the references; its finest point here (mesh 20) is
# −1.99 dB, its *farthest*. When refining the mesh moves the answer around the
# *wrong* value instead of settling on the right one, the error is in the
# **model**, not the resolution — no mesh will fix it.

# %% [markdown]
# ### First rule out the obvious — is it the same launched mode?
#
# A wider field could just mean a different injected mode. It doesn't: both mode
# solvers find the **fundamental TE0** of the 0.5 µm guide at nearly identical
# effective index, and their lateral profiles sit almost on top of each other.
# So the waveguide and the launched mode are the same — whatever differs
# downstream is *not* a wider guide.

# %%
md = np.load(REC / "sbend_injected_modes.npz")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(md["y_tidy3d"], md["e2_tidy3d"], color="tab:red",
        label=f"tidy3d TE0  (n_eff {float(md['neff_tidy3d']):.3f})")
ax.plot(md["y_beamz"], md["e2_beamz"], "--", color="tab:blue",
        label=f"beamz TE0  (n_eff {float(md['neff_beamz']):.3f})")
ax.axvspan(-0.25, 0.25, alpha=0.12, color="gray", label="0.5 µm Si core")
ax.set_xlim(-1.2, 1.2)
ax.set_xlabel("y [µm]")
ax.set_ylabel("|E|² (norm)")
ax.set_title("Injected fundamental (TE0) mode — identical across engines")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ### Now the field through the bend — linear *and* log
#
# Same launched mode, so watch how the field **evolves**. Top row is a **linear**
# scale (only the strong, guided field shows — faint radiation can't inflate it);
# bottom is **log (dB)** (the radiation becomes visible). Each panel is normalized
# to its own peak; cyan rings mark the two ports.
#
# > **Rendering matters.** tidy3d's adaptive mesh is *non-uniform* (35–90 nm
# > cells here), so its field must be drawn on its **true grid coordinates**
# > (`pcolormesh`) — `imshow` with a uniform extent stretches the finely-meshed
# > core ~2× and makes the waveguide *look* wider than it is. An earlier
# > revision of this figure had exactly that bug and read as "tidy3d simulates
# > a wider guide"; it doesn't. beamz's grid is uniform, so either rendering is
# > faithful for it.

# %%
bz = np.load(REC / "sbend_beamz_field.npz")
t3 = np.load(REC / "sbend_tidy3d_field.npz")
lu = np.load(REC / "sbend_lumerical_field.npz")
bw, bh = float(bz["width_um"]), float(bz["height_um"])
cx, cy = sbend.bounds.x_center, sbend.bounds.y_center  # beamz's 0-based frame -> device coords
# beamz: uniform grid -> cell-center coordinate vectors in device µm
b_x = np.linspace(cx - bw / 2, cx + bw / 2, bz["E2"].shape[1])
b_y = np.linspace(cy - bh / 2, cy + bh / 2, bz["E2"].shape[0])
# tidy3d + Lumerical: their own (possibly non-uniform) grid coordinates
panels = [("beamz", b_x, b_y, bz["E2"], float(bz["s21"])),
          ("tidy3d", t3["x"], t3["y"], t3["E2"].T, float(t3["s21"])),
          ("Lumerical", lu["x"], lu["y"], lu["E2"].T, float(lu["s21"]))]

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)
for col, (name, gx, gy, e2, s21) in enumerate(panels):
    en = e2 / e2.max()
    im_lin = ax[0, col].pcolormesh(gx, gy, en, shading="nearest", cmap="magma", vmin=0, vmax=1)
    ax[0, col].set_title(f"{name}  linear   (S21 = {s21:+.2f} dB)")
    im_log = ax[1, col].pcolormesh(gx, gy, 10 * np.log10(np.clip(en, 1e-4, 1)),
                                   shading="nearest", cmap="magma", vmin=-40, vmax=0)
    ax[1, col].set_title(f"{name}  log [dB]")
    for r in (0, 1):
        ax[r, col].scatter([0, 1], [0, 0.5], s=36, edgecolor="cyan", facecolor="none", lw=1.5)
        ax[r, col].set_xlim(cx - 1.9, cx + 1.9)
        ax[r, col].set_ylim(cy - 2, cy + 2)
        ax[r, col].set_aspect("equal")
        ax[r, col].set_xlabel("x [µm]")
        ax[r, col].set_ylabel("y [µm]")
fig.colorbar(im_lin, ax=ax[0, :], label="|E|² (norm)", shrink=0.7)
fig.colorbar(im_log, ax=ax[1, :], label="|E|² [dB]", shrink=0.7)
fig.suptitle("z-plane |E|² in true grid coordinates — three engines, one problem; the gap is in beamz's S-parameters")
plt.show()

# %% [markdown]
# In **true coordinates the three field maps are comparable**: the guided lobe
# has the same ~0.4–0.7 µm lateral FWHM in every engine at matched stations
# along the bend, and all three peak at the same spot on the output guide. We
# audited the *setup* end-to-end for this device: the identical polygons reach
# every engine (beamz's rasterized core measures 0.48 µm vs the 0.50 µm drawn —
# grid quantization), the launched TE0 is the same (previous figure), and the
# materials agree at 1.55 µm to <0.001. What remains different are documented
# engine floors: beamz's uniform grid + own guard band vs the references'
# adaptive/graded meshes, constant vs dispersive materials, vertical vs 85°
# sidewalls. (Note how tidy3d and Lumerical even share the exact same domain,
# x −1…2 µm × y −2.5…3 µm — both are built from `bounds + buffer`.)
#
# So the disagreement is **not** a different simulated structure, and it is not
# obvious in the field picture — it lives in the **energy accounting**: tidy3d
# and Lumerical *both* report S21 ≈ −5.6 dB where beamz's finest mesh says
# −2.0 dB, and the convergence curves above show beamz never trending toward
# the references.
#
# ### The verdict — converged ≠ correct
#
# As an energy budget at 1.55 µm: tidy3d (S21 −5.6, S11 −27.6 dB) and Lumerical
# (S21 −5.63, S11 −27.7 dB) — two fully independent implementations, agreeing
# within 0.03 dB — put a stable ~**72 %** of the input into
# radiation/mode-conversion; beamz's estimate swings with mesh (roughly
# 35–55 % lost) and never reaches it. The gap is a **model** limit: the
# references do a proper multi-mode modal decomposition at the ports, while
# beamz's v1 adapter uses single-mode, per-direction normalization that
# under-counts mode conversion. beamz is excellent for straights and *adiabatic*
# transitions (it matches the `10_cookbook` Si→SiN escalator within ~0.1 dB) —
# but a sharp, radiative bend is outside its comfort zone.
#
# **Lesson:** a convergence sweep is *necessary but not sufficient*. For anything
# strongly multi-mode or radiative, cross-validate against a second engine before
# you trust the number — with two references agreeing, the verdict here is
# unambiguous.
#
# ## Recap & next
#
# `sweep` finds the coarsest converged mesh; `run_cached` makes repeats free; and
# convergence alone doesn't guarantee correctness — cross-check a second engine on
# hard devices. Next: **`07_choosing_an_engine`** — the same job on beamz, tidy3d,
# and Lumerical, and how they line up.
