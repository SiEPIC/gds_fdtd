"""
gds_fdtd simulation toolbox.

Waveguide mode solving — the second Tier-B enabler. Kernel-level
engines need mode profiles at port cross-sections to synthesize sources and
decompose recorded fields; ``ModeSolver`` is the backend-agnostic protocol
(raw permittivity cross-section in, ``Mode`` list out) and
``Tidy3DModeSolver`` the first backend: tidy3d's LOCAL mode-solver plugin —
pure local math, no cloud tasks, no credits (verified: the 500x220 nm Si
strip acceptance case gives n_eff(TE0) = 2.451 at 1.55 um).

The cross-section convention matches ``grid.rasterize`` output: ``eps[u, v]``
with u, v the two transverse axes, cell-centered, spacings ``du``/``dv`` um.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

FIELD_KEYS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@dataclass
class Mode:
    """One guided mode on a transverse cross-section grid."""

    n_eff: complex
    fields: dict[str, np.ndarray]  # Ex..Hz, each (Nu, Nv) complex
    u: np.ndarray  # transverse coordinates of the field grid [um]
    v: np.ndarray
    wavelength_um: float

    def __post_init__(self) -> None:
        missing = [k for k in FIELD_KEYS if k not in self.fields]
        if missing:
            raise ValueError(f"mode is missing field components: {missing}")


@runtime_checkable
class ModeSolver(Protocol):
    """Backend-agnostic mode solver: eps cross-section in, modes out."""

    def solve(
        self,
        eps: np.ndarray,
        du: float,
        dv: float,
        wavelength_um: float,
        n_modes: int = 1,
    ) -> list[Mode]: ...


class Tidy3DModeSolver:
    """tidy3d's local mode-solver plugin (free, offline, no cloud tasks).

    The raw eps cross-section becomes a ``td.CustomMedium`` broadcast along
    the propagation axis; the plugin solves on a plane through it. tidy3d
    must be installed (``pip install gds_fdtd[tidy3d]``) but nothing leaves
    the machine.
    """

    def solve(
        self,
        eps: np.ndarray,
        du: float,
        dv: float,
        wavelength_um: float,
        n_modes: int = 1,
    ) -> list[Mode]:
        try:
            import tidy3d as td
            from tidy3d.plugins.mode import ModeSolver as _TdModeSolver
        except ImportError as e:
            raise ImportError(
                "Tidy3DModeSolver needs tidy3d (local plugin only, no cloud): "
                "pip install gds_fdtd[tidy3d]"
            ) from e

        eps = np.asarray(eps)
        if eps.ndim != 2:
            raise ValueError(f"eps must be 2-D (Nu, Nv); got shape {eps.shape}")
        if min(du, dv) <= 0:
            raise ValueError(f"spacings must be positive; got {(du, dv)}")

        nu, nv = eps.shape
        u = (np.arange(nu) + 0.5) * du - nu * du / 2
        v = (np.arange(nv) + 0.5) * dv - nv * dv / 2
        x = np.array([-1.0, 1.0])  # propagation axis; medium is uniform along it

        data = td.SpatialDataArray(
            np.broadcast_to(eps.real[None, :, :], (2, nu, nv)).copy(),
            coords={"x": x, "y": u, "z": v},
        )
        medium = td.CustomMedium(permittivity=data)
        f0 = td.C_0 / wavelength_um
        span_u, span_v = nu * du, nv * dv
        sim = td.Simulation(
            size=(0.2, span_u, span_v),
            grid_spec=td.GridSpec.uniform(dl=min(du, dv)),
            structures=[
                td.Structure(geometry=td.Box(size=(td.inf, td.inf, td.inf)), medium=medium)
            ],
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
        solver = _TdModeSolver(
            simulation=sim,
            plane=td.Box(center=(0, 0, 0), size=(0, span_u, span_v)),
            mode_spec=td.ModeSpec(num_modes=n_modes),
            freqs=[f0],
        )
        result = solver.solve()

        modes: list[Mode] = []
        n_eff = np.array(result.n_complex).reshape(-1, n_modes)[0]
        grid_u = np.asarray(result.Ex.coords["y"].values)
        grid_v = np.asarray(result.Ex.coords["z"].values)
        for m in range(n_modes):
            fields = {
                key: np.asarray(getattr(result, key).isel(f=0, mode_index=m).values).squeeze()
                for key in FIELD_KEYS
            }
            modes.append(
                Mode(
                    n_eff=complex(n_eff[m]),
                    fields=fields,
                    u=grid_u,
                    v=grid_v,
                    wavelength_um=wavelength_um,
                )
            )
        logger.info(
            "solved %d mode(s) at %g um: n_eff = %s",
            n_modes,
            wavelength_um,
            [round(m.n_eff.real, 4) for m in modes],
        )
        return modes


def waveguide_mode(
    width_um: float,
    height_um: float,
    n_core: float,
    n_clad: float,
    wavelength_um: float,
    *,
    n_modes: int = 1,
    resolution: float = 0.02,
    margin: float = 1.0,
    solver: ModeSolver | None = None,
) -> list[Mode]:
    """Guided modes of a rectangular strip waveguide, offline.

    Builds an ``n_core`` core of ``width_um`` × ``height_um`` centred in an
    ``n_clad`` cladding window (the core plus ``margin`` µm on every side),
    then solves with ``solver`` (default :class:`Tidy3DModeSolver`, a free
    local solve — needs tidy3d installed). Returns the ``n_modes`` lowest
    modes, fundamental first.

    Example: ``waveguide_mode(0.5, 0.22, 3.48, 1.44, 1.55)`` — the canonical
    SOI strip, whose TE0 has n_eff ≈ 2.45 at 1.55 µm.
    """
    if min(width_um, height_um, resolution, margin) <= 0:
        raise ValueError("width_um/height_um/resolution/margin must be positive")
    du = dv = resolution
    nu = max(int(round((width_um + 2 * margin) / du)), 1)
    nv = max(int(round((height_um + 2 * margin) / dv)), 1)
    u = (np.arange(nu) + 0.5) * du - nu * du / 2
    v = (np.arange(nv) + 0.5) * dv - nv * dv / 2
    eps = np.full((nu, nv), float(n_clad) ** 2)
    core = (np.abs(u)[:, None] <= width_um / 2) & (np.abs(v)[None, :] <= height_um / 2)
    eps[core] = float(n_core) ** 2
    active = solver if solver is not None else Tidy3DModeSolver()
    return active.solve(eps, du, dv, wavelength_um, n_modes=n_modes)
