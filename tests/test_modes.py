"""WP5.2b: mode solver protocol + tidy3d local backend.

The acceptance case from the plan card: a 500x220 nm Si strip in SiO2 at
1.55 um must give n_eff(TE0) ~ 2.4 +/- 0.1 (literature). Runs wherever
tidy3d is installed (all-extras CI leg / owner env); the solve is pure
local math — no cloud tasks, no credits.
"""

from __future__ import annotations

import numpy as np
import pytest

from gds_fdtd.modes import FIELD_KEYS, Mode, ModeSolver, Tidy3DModeSolver


def _strip_eps(du=0.02, n_core=3.476, n_clad=1.444):
    u = np.arange(-1.5, 1.5, du) + du / 2
    v = np.arange(-1.5, 1.5, du) + du / 2
    eps = np.full((u.size, v.size), n_clad**2)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    eps[(np.abs(uu) <= 0.25) & (np.abs(vv) <= 0.11)] = n_core**2
    return eps


def test_mode_requires_all_components():
    with pytest.raises(ValueError, match="missing field components"):
        Mode(
            n_eff=2.4,
            fields={"Ex": np.zeros((2, 2))},
            u=np.arange(2),
            v=np.arange(2),
            wavelength_um=1.55,
        )


def test_backend_satisfies_protocol():
    assert isinstance(Tidy3DModeSolver(), ModeSolver)


def test_si_strip_neff_acceptance():
    pytest.importorskip("tidy3d")
    modes = Tidy3DModeSolver().solve(_strip_eps(), du=0.02, dv=0.02, wavelength_um=1.55, n_modes=2)
    assert len(modes) == 2
    te0 = modes[0]
    # plan-card acceptance: literature n_eff(TE0) ~ 2.4 +/- 0.1
    assert te0.n_eff.real == pytest.approx(2.4, abs=0.1)
    assert modes[1].n_eff.real < te0.n_eff.real  # higher modes are slower
    for key in FIELD_KEYS:
        assert te0.fields[key].shape == (te0.u.size, te0.v.size)
    # TE0 is Ey-dominant in this (u, v) = (y, z) orientation
    assert np.abs(te0.fields["Ey"]).max() > np.abs(te0.fields["Ez"]).max()


def test_solver_input_validation():
    pytest.importorskip("tidy3d")
    with pytest.raises(ValueError, match="2-D"):
        Tidy3DModeSolver().solve(np.zeros(5), du=0.02, dv=0.02, wavelength_um=1.55)
    with pytest.raises(ValueError, match="positive"):
        Tidy3DModeSolver().solve(np.zeros((5, 5)), du=0.0, dv=0.02, wavelength_um=1.55)
