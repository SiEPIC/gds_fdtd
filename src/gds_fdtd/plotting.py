"""
gds_fdtd simulation toolbox.

Plotting helpers for the canonical SMatrix (WP2.4d). matplotlib is imported
lazily so headless / minimal installs never pay for it.
"""

from __future__ import annotations

import numpy as np


def plot_smatrix(sm, kind: str = "db", ax=None, paths=None):
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
        paths = [
            (sm.port_names[o], sm.port_names[i], mo + 1, mi + 1)
            for o in range(sm.n_ports)
            for i in range(sm.n_ports)
            for mo in range(sm.n_modes)
            for mi in range(sm.n_modes)
            if not np.all(np.isnan(sm.s[:, o, i, mo, mi]))
        ]

    wavl = sm.wavelength_um
    for out, in_, mode_out, mode_in in paths:
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
        ax.plot(wavl, y, label=label)

    ax.set_xlabel("Wavelength [µm]")
    ax.set_title(f"{sm.name} S-parameters")
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_component(component, spec=None, ax=None, savefig=None):
    """Standard geometry view: device polygons, ports, bounds, simulation region.

    This is the first step of the standardized example flow — always offline,
    engine-agnostic.

    Args:
        component: a gds_fdtd Component.
        spec: optional SimulationSpec; when given, the FDTD region
            (bounds + buffer) is drawn as well.
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
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    layer_colors: dict[tuple, str] = {}
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

    # simulation region = bounds + buffer (matches Solver.domain())
    if spec is not None:
        buf = spec.buffer
        ax.add_patch(
            Rectangle(
                (b.x_min - buf, b.y_min - buf),
                b.x_span + 2 * buf,
                b.y_span + 2 * buf,
                fill=False,
                edgecolor="tab:red",
                linestyle=":",
                linewidth=1.4,
                label=f"FDTD region (buffer {buf} µm)",
            )
        )

    # ports: arrow along direction + name
    arrow = max(b.x_span, b.y_span) * 0.06
    for p in component.ports:
        dx, dy = {0: (arrow, 0), 90: (0, arrow), 180: (-arrow, 0), 270: (0, -arrow)}[p.direction]
        ax.annotate(
            "",
            xy=(p.x + dx, p.y + dy),
            xytext=(p.x, p.y),
            arrowprops={"arrowstyle": "-|>", "color": "tab:green", "lw": 2},
        )
        # port width tick, perpendicular to the direction
        if p.direction in (0, 180):
            ax.plot([p.x, p.x], [p.y - p.width / 2, p.y + p.width / 2], "tab:green", lw=2)
        else:
            ax.plot([p.x - p.width / 2, p.x + p.width / 2], [p.y, p.y], "tab:green", lw=2)
        ax.annotate(
            p.name,
            (p.x, p.y),
            textcoords="offset points",
            xytext=(6, 6),
            color="tab:green",
            fontsize=9,
            fontweight="bold",
        )

    pad = max(b.x_span, b.y_span) * 0.15 + (spec.buffer if spec else 0)
    ax.set_xlim(b.x_min - pad, b.x_max + pad)
    ax.set_ylim(b.y_min - pad, b.y_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"{component.name}: geometry, ports, simulation region")
    ax.legend(loc="upper right", fontsize="small")
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax
