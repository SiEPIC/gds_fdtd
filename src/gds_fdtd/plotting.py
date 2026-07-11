"""
gds_fdtd simulation toolbox.

Plotting helpers for the canonical SMatrix. matplotlib is imported
lazily so headless / minimal installs never pay for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .geometry import Component
    from .modes import Mode
    from .smatrix import SMatrix
    from .spec import SimulationSpec
    from .technology import Technology

# Package-wide default color scheme (owner directive): the RdBu family.
# Field images use the continuous map; categorical needs (S-parameter
# traces, GDS layers, engines in comparison plots) sample its deep ends.
DEFAULT_CMAP = "RdBu_r"


def rdbu_colors(n: int) -> list[Any]:
    """``n`` distinguishable colors sampled from RdBu, skipping the pale center.

    Alternates the deep-blue and deep-red ends and walks inward, so up to
    ~10 series stay tellable-apart while everything belongs to one palette.
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdBu")
    per_side = max((n + 1) // 2, 1)
    stops = []
    for i in range(n):
        depth = (i // 2) / per_side * 0.30
        stops.append((0.95 - depth) if i % 2 == 0 else (0.05 + depth))
    return [cmap(s) for s in stops]


def plot_smatrix(
    sm: SMatrix, kind: str = "db", ax: Any = None, paths: Any = None
) -> tuple[Any, Any]:
    """Plot S-matrix paths versus wavelength.

    Args:
        sm: an SMatrix.
        kind: "db" (|S|² in dB), "linear" (|S|), or "phase" (unwrapped rad).
        ax: existing matplotlib axes (optional).
        paths: iterable of (out, in_, mode_out, mode_in) tuples; defaults to
            every measured (non-NaN) path.

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    valid = ("db", "linear", "phase")
    if kind not in valid:
        raise ValueError(f"kind must be one of {valid}; got {kind!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    if paths is None:
        # default to measured paths that carry signal: entries whose maximum
        # is below -60 dB are numerical noise and only bury the real traces
        # (a full 4-port x 2-mode matrix has 64 paths; ~half are noise)
        paths = []
        for o in range(sm.n_ports):
            for i in range(sm.n_ports):
                for mo in range(sm.n_modes):
                    for mi in range(sm.n_modes):
                        col = sm.s[:, o, i, mo, mi]
                        if np.all(np.isnan(col)):
                            continue
                        with np.errstate(divide="ignore", invalid="ignore"):
                            peak = 10 * np.log10(np.nanmax(np.abs(col)) ** 2)
                        if peak > -60.0:
                            paths.append((sm.port_names[o], sm.port_names[i], mo + 1, mi + 1))

    wavl = sm.wavelength_um
    trace_colors = rdbu_colors(len(paths))
    for (out, in_, mode_out, mode_in), trace_color in zip(paths, trace_colors, strict=True):
        col = sm.sel(out, in_, mode_out, mode_in)
        label = f"S({out}←{in_})"
        if sm.n_modes > 1:
            label += f" m{mode_out}{mode_in}"
        if kind == "db":
            with np.errstate(divide="ignore", invalid="ignore"):
                y = 10 * np.log10(np.abs(col) ** 2)
            ax.set_ylabel("Transmission [dB]")
        elif kind == "linear":
            y = np.abs(col)
            ax.set_ylabel("|S| [normalized]")
        else:
            y = np.unwrap(np.angle(col))
            ax.set_ylabel("Phase [rad]")
        ax.plot(wavl, y, label=label, color=trace_color)

    ax.set_xlabel("Wavelength [µm]")
    title = sm.name if len(sm.name) <= 40 else sm.name[:37] + "..."
    ax.set_title(f"{title} S-parameters")
    if len(paths) > 8:  # keep dense matrices readable: legend outside
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="x-small", ncol=2)
        fig.tight_layout()
    else:
        ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_component(
    component: Component,
    spec: SimulationSpec | None = None,
    ax: Any = None,
    savefig: str | None = None,
) -> tuple[Any, Any]:
    """Standard geometry view: device polygons, ports, bounds, simulation region.

    This is the first step of the standardized example flow — always offline,
    engine-agnostic.

    Args:
        component: a gds_fdtd Component.
        spec: optional SimulationSpec; when given, the nominal FDTD domain
            (bounds + buffer), the outer stub/background extent
            (bounds + 2*buffer, outside the domain), and the port-extension
            stubs are drawn as well.
        ax: existing matplotlib axes (optional).
        savefig: path to write the figure to (optional).

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.patches import Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # device polygons, colored per GDS layer
    palette = rdbu_colors(8)
    layer_colors: dict[tuple[int, ...], str] = {}
    seen_layers = set()
    for s in component.structures:
        if s.role != "device":
            continue
        key = tuple(s.layer)
        color = layer_colors.setdefault(key, palette[len(layer_colors) % len(palette)])
        label = f"layer {key[0]}/{key[1]}" if key not in seen_layers else None
        seen_layers.add(key)
        ax.add_patch(
            MplPolygon(
                s.polygon,
                closed=True,
                facecolor=color,
                edgecolor="k",
                linewidth=0.4,
                alpha=0.6,
                label=label,
            )
        )

    # bounds (devrec) rectangle
    b = component.bounds
    ax.add_patch(
        Rectangle(
            (b.x_min, b.y_min),
            b.x_span,
            b.y_span,
            fill=False,
            edgecolor="tab:gray",
            linestyle="--",
            linewidth=1.2,
            label="bounds (devrec)",
        )
    )

    # Nominal FDTD domain + the larger geometry extent, drawn only with a spec.
    # INNER box = bounds + buffer/side = the nominal simulation domain. This is
    # engine-AGNOSTIC (tidy3d/Lumerical build bounds + buffer/side; beamz uses
    # its own guard band), so it is a representative preview, not one solver's
    # exact box — hence "nominal". OUTER box = bounds + 2*buffer/side = how far
    # the port-extension stubs and background medium reach; that geometry
    # overhangs the domain and is clipped/absorbed at the domain edge (it is NOT
    # simulated out there). Drawing it gives the 2*buffer stubs a labeled
    # boundary to end on instead of poking into blank space.
    if spec is not None:
        buf = spec.buffer
        ax.add_patch(
            Rectangle(
                (b.x_min - buf, b.y_min - buf),
                b.x_span + 2 * buf,
                b.y_span + 2 * buf,
                fill=False,
                edgecolor=plt.get_cmap("RdBu")(0.02),
                linestyle=":",
                linewidth=1.4,
                label=f"FDTD region (nominal, +{buf:g} µm)",
            )
        )
        ax.add_patch(
            Rectangle(
                (b.x_min - 2 * buf, b.y_min - 2 * buf),
                b.x_span + 4 * buf,
                b.y_span + 4 * buf,
                fill=False,
                edgecolor="tab:gray",
                linestyle=(0, (1, 3)),
                linewidth=1.0,
                alpha=0.7,
                label="stub / background extent (outside domain)",
            )
        )

    # Port-extension stubs: synthetic waveguide the solver bolts onto each port
    # so it runs out through the domain edge (into the absorbing boundary)
    # rather than ending on a reflecting facet. Engine-added, not part of the
    # device; the extent is 2*buffer, ending on the outer "stub / background
    # extent" box above. Dashed + hatched + low-alpha so it stays subordinate.
    if spec is not None:
        stub_labeled = False
        for p in component.ports:
            stub = p.polygon_extension(buffer=2 * spec.buffer)
            ax.add_patch(
                MplPolygon(
                    stub,
                    closed=True,
                    facecolor="none",
                    edgecolor=plt.get_cmap("RdBu")(0.98),
                    linewidth=1.0,
                    linestyle="--",
                    hatch="///",
                    alpha=0.5,
                    label=None if stub_labeled else "port extension (out through domain edge)",
                )
            )
            stub_labeled = True

    # ports: arrow along direction + name
    arrow = max(b.x_span, b.y_span) * 0.06
    for p in component.ports:
        dx, dy = {0: (arrow, 0), 90: (0, arrow), 180: (-arrow, 0), 270: (0, -arrow)}[
            int(p.direction)
        ]
        ax.annotate(
            "",
            xy=(p.x + dx, p.y + dy),
            xytext=(p.x, p.y),
            arrowprops={"arrowstyle": "-|>", "color": plt.get_cmap("RdBu")(0.98), "lw": 2},
        )
        # port width tick, perpendicular to the direction
        if p.direction in (0, 180):
            ax.plot(
                [p.x, p.x],
                [p.y - p.width / 2, p.y + p.width / 2],
                color=plt.get_cmap("RdBu")(0.98),
                lw=2,
            )
        else:
            ax.plot(
                [p.x - p.width / 2, p.x + p.width / 2],
                [p.y, p.y],
                color=plt.get_cmap("RdBu")(0.98),
                lw=2,
            )
        ax.annotate(
            p.name,
            (p.x, p.y),
            textcoords="offset points",
            xytext=(6, 6),
            color=plt.get_cmap("RdBu")(0.98),
            fontsize=9,
            fontweight="bold",
        )

    # Frame the union of everything actually drawn so nothing clips regardless
    # of the buffer/device-size ratio: devrec bounds, device polygons, the
    # stub/background-extent box (bounds + 2*buffer, the outermost artist when a
    # spec is given), and the port arrows/width ticks (reusing `arrow` above).
    xs: list[float] = [b.x_min, b.x_max]
    ys: list[float] = [b.y_min, b.y_max]
    for s in component.structures:
        if s.role == "device":
            for vx, vy in s.polygon:
                xs.append(float(vx))
                ys.append(float(vy))
    if spec is not None:
        xs += [b.x_min - 2 * spec.buffer, b.x_max + 2 * spec.buffer]
        ys += [b.y_min - 2 * spec.buffer, b.y_max + 2 * spec.buffer]
    for p in component.ports:
        xs += [p.x - arrow, p.x + arrow, p.x - p.width / 2, p.x + p.width / 2]
        ys += [p.y - arrow, p.y + arrow, p.y - p.width / 2, p.y + p.width / 2]
    mx = (max(xs) - min(xs)) * 0.05 or 1e-9
    my = (max(ys) - min(ys)) * 0.05 or 1e-9
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"{component.name}: geometry, ports, simulation region")
    ax.legend(loc="upper right", fontsize="small")
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_tech_stack(
    tech: Technology,
    wavelength_um: float = 1.55,
    ax: Any = None,
    savefig: str | None = None,
) -> tuple[Any, Any]:
    """Vertical cross-section of the technology's layer stack.

    Draws the substrate, each device layer, and the superstrate as z-bands,
    labelled with their refractive index at ``wavelength_um``. This is the
    "what is my material stack?" view — the vertical companion to the top-down
    :func:`plot_component`. Offline and engine-agnostic.

    Args:
        tech: a :class:`gds_fdtd.technology.Technology`.
        wavelength_um: wavelength for the refractive-index labels.
        ax: existing matplotlib axes (optional).
        savefig: path to write the figure to (optional).

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from .grid import resolve_index

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))
    else:
        fig = ax.figure

    def _n(material: Any) -> float | None:
        try:
            return float(np.real(resolve_index(material, wavelength_um)))
        except Exception:
            return None

    # each band: (label, z_lo, z_hi, refractive index n, is_device)
    bands = []
    zb, zs = tech.substrate.z_base, tech.substrate.z_span
    bands.append(
        ("substrate", min(zb, zb + zs), max(zb, zb + zs), _n(tech.substrate.material), False)
    )
    for d in tech.device:
        bands.append(
            (
                f"device {d.layer[0]}/{d.layer[1]}",
                min(d.z_base, d.z_base + d.z_span),
                max(d.z_base, d.z_base + d.z_span),
                _n(d.material),
                True,
            )
        )
    zb, zs = tech.superstrate.z_base, tech.superstrate.z_span
    bands.append(
        ("superstrate", min(zb, zb + zs), max(zb, zb + zs), _n(tech.superstrate.material), False)
    )

    # zoom to the device region so the thin (often ~0.2 µm) device layers are
    # visible; substrate/superstrate fill the margins and continue beyond it.
    dev_z = [z for _, lo, hi, _, dev in bands if dev for z in (lo, hi)]
    dev_lo, dev_hi = (min(dev_z), max(dev_z)) if dev_z else (0.0, 0.22)
    pad = max(0.8, 1.5 * (dev_hi - dev_lo))
    view_lo, view_hi = dev_lo - pad, dev_hi + pad
    view_h = view_hi - view_lo

    ns = [n for *_, n, _ in bands if n is not None]
    nmin, nmax = (min(ns), max(ns)) if ns else (1.0, 1.0)
    cmap = plt.get_cmap("RdBu_r")

    def _color(n: float | None) -> Any:
        if n is None:
            return "0.8"
        frac = 0.5 if nmax == nmin else 0.12 + 0.76 * (n - nmin) / (nmax - nmin)
        return cmap(frac)

    # draw backgrounds first, device layers on top so thin cores stay visible
    for label, z_lo, z_hi, n, is_device in sorted(bands, key=lambda b: b[4]):
        ax.add_patch(
            Rectangle(
                (0, z_lo),
                1,
                z_hi - z_lo,
                facecolor=_color(n),
                edgecolor="k",
                linewidth=0.7,
                zorder=2 if is_device else 1,
            )
        )
        n_txt = f"n={n:.3f}" if n is not None else "n=?"
        vis_lo, vis_hi = max(z_lo, view_lo), min(z_hi, view_hi)
        zc = (vis_lo + vis_hi) / 2
        if is_device or (vis_hi - vis_lo) < 0.12 * view_h:
            # thin/device layer: leader line to a label on the right
            ax.annotate(
                f"{label}\n{n_txt}",
                xy=(1.0, zc),
                xytext=(1.28, zc),
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": "0.3", "lw": 0.9},
                zorder=5,
            )
        else:
            dark = n is not None and n >= (nmin + nmax) / 2
            ax.text(
                0.5,
                zc,
                f"{label}\n{n_txt}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if dark else "k",
                zorder=5,
            )

    ax.set_xlim(0, 1.7)
    ax.set_ylim(view_lo, view_hi)
    ax.set_xticks([])
    ax.set_ylabel("z [µm]")
    ax.set_title(f"{tech.name}: layer stack @ {wavelength_um} µm")
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_permittivity(
    component: Component,
    dx: float = 0.02,
    wavelength_um: float = 1.55,
    axis: str = "y",
    position: float | None = None,
    buffer: float = 1.0,
    ax: Any = None,
    savefig: str | None = None,
) -> tuple[Any, Any]:
    """Cross-section of the rasterized refractive-index (√ε) the solver sees.

    Rasterizes the component with :func:`gds_fdtd.grid.rasterize` and slices
    it along ``axis`` at ``position`` (defaulting to the geometry center for
    x/y and the first device-layer mid-plane for z). ``axis="y"`` gives the
    waveguide side view (x–z), ``axis="z"`` the top view (x–y). Offline; needs
    materials that resolve locally (constant ``nk`` or an ``rii`` reference).

    Args:
        component: a gds_fdtd Component.
        dx: rasterization cell size in µm (isotropic).
        wavelength_um: wavelength for index resolution.
        axis: slice-normal axis, one of "x", "y", "z".
        position: coordinate along ``axis`` to slice at (µm); defaults per above.
        buffer: xy padding around the component bounds (µm).
        ax: existing matplotlib axes (optional).
        savefig: path to write the figure to (optional).

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    from .grid import rasterize

    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")

    # extend_ports + full-domain cladding => the slice matches what the solvers
    # actually build (guide extended through the domain edge, clad everywhere),
    # not just the bare device footprint in vacuum.
    grid = rasterize(component, dx, wavelength=wavelength_um, buffer=buffer, extend_ports=True)
    n = np.sqrt(grid.eps.real)  # refractive index at each cell center
    x, y, z = grid.x, grid.y, grid.z

    if axis == "y":  # side view: x (horizontal) vs z (vertical)
        pos = y[len(y) // 2] if position is None else position
        j = int(np.argmin(np.abs(y - pos)))
        img, extent, xlabel, ylabel = n[:, j, :].T, (x[0], x[-1], z[0], z[-1]), "x [µm]", "z [µm]"
        where = f"y = {y[j]:.3f} µm"
    elif axis == "x":  # side view: y vs z
        pos = x[len(x) // 2] if position is None else position
        i = int(np.argmin(np.abs(x - pos)))
        img, extent, xlabel, ylabel = n[i, :, :].T, (y[0], y[-1], z[0], z[-1]), "y [µm]", "z [µm]"
        where = f"x = {x[i]:.3f} µm"
    else:  # top view: x vs y
        if position is None:
            devs = [s for s in component.structures if s.role == "device"]
            position = (devs[0].z_base + devs[0].z_span / 2) if devs else z[len(z) // 2]
        k = int(np.argmin(np.abs(z - position)))
        img, extent, xlabel, ylabel = n[:, :, k].T, (x[0], x[-1], y[0], y[-1]), "x [µm]", "y [µm]"
        where = f"z = {z[k]:.3f} µm"

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
    else:
        fig = ax.figure
    im = ax.imshow(img, origin="lower", extent=extent, aspect="equal", cmap=DEFAULT_CMAP)
    fig.colorbar(im, ax=ax, label=f"refractive index n @ {wavelength_um} µm")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{component.name}: permittivity ({axis}-slice, {where})")
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax


def smatrix_summary(
    sm: SMatrix, wavelength_um: float | None = None, atol: float = 1e-2
) -> dict[str, Any]:
    """Human-readable figures of merit for an SMatrix at one wavelength.

    Returns insertion/return loss per driven port plus the physical sanity
    checks (reciprocity, passivity, worst-case power imbalance). Handy for a
    one-line readout next to :func:`plot_smatrix`.

    Args:
        sm: an SMatrix.
        wavelength_um: wavelength to evaluate at; defaults to the band center.
        atol: absolute tolerance for the reciprocity/passivity checks. The
            default 1e-2 suits real FDTD output (which carries ~1e-3 numerical
            asymmetry); tighten it for analytic/reference matrices.

    Returns:
        dict with keys ``wavelength_um``, ``reciprocal``, ``passive``,
        ``max_power_imbalance``, and ``paths`` (list of per-path dB dicts).
    """
    wl = sm.wavelength_um
    idx = len(wl) // 2 if wavelength_um is None else int(np.argmin(np.abs(wl - wavelength_um)))

    paths = []
    for i in range(sm.n_ports):
        for o in range(sm.n_ports):
            for mi in range(sm.n_modes):
                for mo in range(sm.n_modes):
                    s = sm.s[idx, o, i, mo, mi]
                    if np.isnan(s):
                        continue
                    with np.errstate(divide="ignore"):
                        db = float(10 * np.log10(abs(s) ** 2))
                    if db <= -60:  # numerical noise floor
                        continue
                    paths.append(
                        {
                            "in": sm.port_names[i],
                            "out": sm.port_names[o],
                            "mode_in": mi + 1,
                            "mode_out": mo + 1,
                            "kind": "reflection" if i == o else "transmission",
                            "db": round(db, 3),
                        }
                    )
    pb = sm.power_balance()
    return {
        "wavelength_um": round(float(wl[idx]), 4),
        "reciprocal": bool(sm.is_reciprocal(atol=atol)),
        "passive": bool(sm.is_passive(atol=atol)),
        "max_power_imbalance": round(float(np.nanmax(np.abs(pb))), 4),
        "paths": sorted(paths, key=lambda p: (p["in"], -p["db"])),
    }


def plot_mode(
    mode: Mode, field: str = "E", ax: Any = None, savefig: str | None = None
) -> tuple[Any, Any]:
    """Plot a waveguide mode's transverse field on its cross-section.

    ``field="E"`` shows the total field magnitude ``|E| = √(|Ex|²+|Ey|²+|Ez|²)``;
    pass a single component (``"Ex"``, ``"Ey"``, ``"Ez"``, ``"Hx"`` …) for one.
    The mode's effective index is annotated — the number that tells you how the
    mode propagates. Pairs with :class:`gds_fdtd.modes.Tidy3DModeSolver` (a free,
    offline solve).

    Args:
        mode: a :class:`gds_fdtd.modes.Mode`.
        field: "E" (default, total |E|) or a single field-component key.
        ax: existing matplotlib axes (optional).
        savefig: path to write the figure to (optional).

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    if field == "E":
        data = np.sqrt(sum(np.abs(mode.fields[k]) ** 2 for k in ("Ex", "Ey", "Ez")))
        label = "|E|"
    else:
        if field not in mode.fields:
            raise ValueError(f"unknown field {field!r}; have {sorted(mode.fields)}")
        data = np.abs(mode.fields[field])
        label = f"|{field}|"

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.figure
    extent = (mode.u[0], mode.u[-1], mode.v[0], mode.v[-1])
    im = ax.imshow(data.T, origin="lower", extent=extent, aspect="equal", cmap=DEFAULT_CMAP)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("u [µm]")
    ax.set_ylabel("v [µm]")
    ax.set_title(f"mode {label} @ {mode.wavelength_um} µm  (n_eff = {mode.n_eff.real:.4f})")
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax
