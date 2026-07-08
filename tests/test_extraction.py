"""WP5.2c: mode-overlap extraction — exact identities on synthetic modes.

The bidirectional projection has exact algebraic properties that pin the
convention: self-overlap = 1, backward-traveling field -> forward amplitude
0 / backward amplitude 1, orthogonal (odd/even) modes -> 0, linearity.
"""

from __future__ import annotations

import numpy as np
import pytest

from gds_fdtd.extraction import decompose, mode_amplitude, mode_power
from gds_fdtd.modes import FIELD_KEYS, Mode

DU = DV = 0.02


def _gaussian_mode(order=0, n_eff=2.4):
    """Synthetic TE-like mode on a plane normal to x: Ey, Hz Gaussian
    (times a Hermite factor for order 1), other components zero."""
    u = (np.arange(100) + 0.5) * DU - 1.0
    v = (np.arange(80) + 0.5) * DV - 0.8
    uu, vv = np.meshgrid(u, v, indexing="ij")
    envelope = np.exp(-(uu**2) / 0.18 - (vv**2) / 0.08)
    if order == 1:
        envelope = envelope * uu  # odd in u -> orthogonal to order 0
    fields = {k: np.zeros_like(envelope, dtype=complex) for k in FIELD_KEYS}
    fields["Ey"] = envelope.astype(complex)
    fields["Hz"] = n_eff * envelope.astype(complex)  # H ~ n_eff * E (arb. units)
    return Mode(n_eff=n_eff, fields=fields, u=u, v=v, wavelength_um=1.55)


def test_self_overlap_is_one():
    mode = _gaussian_mode()
    a = mode_amplitude(mode.fields, mode, DU, DV, normal="x")
    assert a == pytest.approx(1.0 + 0j, abs=1e-12)


def test_backward_field_projects_to_backward():
    mode = _gaussian_mode()
    backward = dict(mode.fields)
    backward["Hz"] = -mode.fields["Hz"]  # backward-traveling: transverse H flips
    assert mode_amplitude(backward, mode, DU, DV, direction="+") == pytest.approx(0, abs=1e-12)
    assert mode_amplitude(backward, mode, DU, DV, direction="-") == pytest.approx(1, abs=1e-12)


def test_orthogonal_modes_do_not_mix():
    te0, te1 = _gaussian_mode(order=0), _gaussian_mode(order=1)
    assert mode_amplitude(te0.fields, te1, DU, DV) == pytest.approx(0, abs=1e-12)


def test_linearity_and_decompose():
    te0, te1 = _gaussian_mode(order=0), _gaussian_mode(order=1)
    s21 = 0.7 * np.exp(1j * 0.3)
    field = {k: s21 * te0.fields[k] + 0.2j * te1.fields[k] for k in FIELD_KEYS}
    amps = decompose(field, [te0, te1], DU, DV)
    assert amps[0] == pytest.approx(s21, abs=1e-12)
    assert amps[1] == pytest.approx(0.2j, abs=1e-12)


def test_mode_power_positive_and_errors():
    mode = _gaussian_mode()
    assert mode_power(mode, DU, DV) > 0
    with pytest.raises(ValueError, match="normal"):
        mode_amplitude(mode.fields, mode, DU, DV, normal="q")
    with pytest.raises(ValueError, match="direction"):
        mode_amplitude(mode.fields, mode, DU, DV, direction="fwd")
    small = {k: v[:10, :10] for k, v in mode.fields.items()}
    with pytest.raises(ValueError, match="does not match"):
        mode_amplitude(small, mode, DU, DV)


def test_tidy3d_mode_self_overlap():
    """The identity must also hold for a REAL solved mode (all 6 components)."""
    pytest.importorskip("tidy3d")
    from gds_fdtd.modes import Tidy3DModeSolver

    du = 0.02
    u = np.arange(-1.5, 1.5, du) + du / 2
    v = np.arange(-1.5, 1.5, du) + du / 2
    eps = np.full((u.size, v.size), 1.444**2)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    eps[(np.abs(uu) <= 0.25) & (np.abs(vv) <= 0.11)] = 3.476**2
    mode = Tidy3DModeSolver().solve(eps, du, du, 1.55, n_modes=1)[0]
    a = mode_amplitude(mode.fields, mode, du, du, normal="x")
    assert a == pytest.approx(1.0 + 0j, abs=1e-9)
