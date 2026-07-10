"""
gds_fdtd simulation toolbox.

Mode-overlap extraction — the last Tier-B enabler. Kernel engines
(fdtdz, fdtdx) record raw DFT fields at port planes; this module turns them
into the complex modal amplitudes that become SMatrix entries. Everything is
solver-independent numpy: fields in, complex out.

Convention (the standard bidirectional mode-orthogonality projection):

    a_m = [ ∫(E × H_m*)·n̂ dA + ∫(E_m* × H)·n̂ dA ] / (4 N_m)
    N_m = ½ Re ∫ (E_m × H_m*)·n̂ dA

so a field equal to the mode gives a_m = 1 exactly, the backward-traveling
mode gives 0, and |a_m|² is the fraction of the mode's normalized power
carried forward. ``direction="-"`` projects onto the backward mode instead
(transverse H flips sign).
"""

from __future__ import annotations

import logging

import numpy as np

from .modes import Mode

logger = logging.getLogger(__name__)

# transverse component pairs (u, v) for a plane normal to each axis, ordered
# so that u x v = +normal (right-handed)
_TRANSVERSE = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}


def _cross_flux(eu, ev, hu, hv, du: float, dv: float) -> complex:
    """∫ (E × H)·n̂ dA on the cell-centered grid."""
    return complex(np.sum(eu * hv - ev * hu) * du * dv)


def _transverse(fields: dict[str, np.ndarray], normal: str):
    try:
        cu, cv = _TRANSVERSE[normal]
    except KeyError:
        raise ValueError(f"normal must be one of {sorted(_TRANSVERSE)}; got {normal!r}") from None
    return fields[f"E{cu}"], fields[f"E{cv}"], fields[f"H{cu}"], fields[f"H{cv}"]


def mode_power(mode: Mode, du: float, dv: float, normal: str = "x") -> float:
    """N_m = half the real Poynting flux of the mode through its plane."""
    eu, ev, hu, hv = _transverse(mode.fields, normal)
    return 0.5 * _cross_flux(eu, ev, np.conj(hu), np.conj(hv), du, dv).real


def mode_amplitude(
    fields: dict[str, np.ndarray],
    mode: Mode,
    du: float,
    dv: float,
    normal: str = "x",
    direction: str = "+",
) -> complex:
    """Complex amplitude of ``mode`` in a recorded field on the same grid.

    ``fields`` maps component names ("Ex".."Hz") to (Nu, Nv) arrays sampled
    on the SAME grid as the mode profile (resample beforehand if not).
    ``direction`` selects the forward ("+", along +normal) or backward
    ("-") traveling mode.
    """
    if direction not in ("+", "-"):
        raise ValueError(f"direction must be '+' or '-'; got {direction!r}")
    eu, ev, hu, hv = _transverse(fields, normal)
    m_eu, m_ev, m_hu, m_hv = _transverse(mode.fields, normal)
    if eu.shape != m_eu.shape:
        raise ValueError(
            f"field grid {eu.shape} does not match mode grid {m_eu.shape}; "
            "resample the fields onto the mode grid first"
        )
    n_m = mode_power(mode, du, dv, normal)
    if n_m <= 0:
        raise ValueError(
            f"mode carries no forward power through the plane (N_m = {n_m:g}); "
            "check the mode orientation/normal"
        )
    c1 = _cross_flux(eu, ev, np.conj(m_hu), np.conj(m_hv), du, dv)
    c2 = _cross_flux(np.conj(m_eu), np.conj(m_ev), hu, hv, du, dv)
    sign = 1.0 if direction == "+" else -1.0
    return (c1 + sign * c2) / (4.0 * n_m)


def decompose(
    fields: dict[str, np.ndarray],
    modes: list[Mode],
    du: float,
    dv: float,
    normal: str = "x",
    direction: str = "+",
) -> np.ndarray:
    """Amplitudes of every mode in a recorded field; shape (len(modes),)."""
    return np.array(
        [mode_amplitude(fields, m, du, dv, normal=normal, direction=direction) for m in modes]
    )
