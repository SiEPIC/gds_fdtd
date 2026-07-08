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
