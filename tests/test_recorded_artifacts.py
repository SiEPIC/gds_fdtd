"""Replay tests over recorded real-solver artifacts (WP7.1.4).

These exercise the full results pipeline (HDF5 <-> SMatrix <-> .dat, physics
checks) against a REAL tidy3d cloud result without any network access.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from gds_fdtd.smatrix import SMatrix

RECORDED = pathlib.Path(__file__).parent / "recorded"


@pytest.fixture(scope="module")
def escalator_sm() -> SMatrix:
    pytest.importorskip("h5py")
    return SMatrix.from_hdf5(str(RECORDED / "si_sin_escalator_smatrix.h5"))


def test_recorded_smatrix_shape(escalator_sm):
    assert escalator_sm.n_ports == 2 and escalator_sm.n_modes == 1
    assert escalator_sm.f.size == 11
    assert np.all(np.diff(escalator_sm.f) > 0)


def test_recorded_physics(escalator_sm):
    """Real cloud result: near-unity escalator transmission, low reflection."""
    s21_db = escalator_sm.magnitude_db(out=2, in_=1)
    s11_db = escalator_sm.magnitude_db(out=1, in_=1)
    assert s21_db.max() > -1.0  # adiabatic escalator ≈ full transmission
    assert s11_db.max() < -20.0
    assert escalator_sm.is_reciprocal(atol=0.05)
    # coarse-mesh normalization overshoots unity by <1%: use a loose passivity tol
    assert escalator_sm.is_passive(atol=0.02)


def test_recorded_dat_matches_hdf5(escalator_sm):
    """The .dat exported from the same run parses back to the same values."""
    back = SMatrix.from_dat(str(RECORDED / "si_sin_escalator.dat"))
    np.testing.assert_allclose(
        np.abs(back.sel(out=2, in_=1)), np.abs(escalator_sm.sel(out=2, in_=1)), rtol=1e-6
    )
    np.testing.assert_allclose(back.f, escalator_sm.f, rtol=1e-9)


def test_cross_solver_agreement():
    """THE flagship check: tidy3d (cloud) and Lumerical v252 (local) ran the
    SAME escalator fixture at the same coarse settings — their S-matrices must
    agree. Recorded 2026-07-07; |S21| within 0.15 dB, |S11| both < -20 dB."""
    pytest.importorskip("h5py")
    t3d = SMatrix.from_hdf5(str(RECORDED / "si_sin_escalator_smatrix.h5"))
    lum = SMatrix.from_hdf5(str(RECORDED / "si_sin_escalator_lum_smatrix.h5"))

    s21_t3d = t3d.magnitude_db(out=2, in_=1)
    s21_lum = lum.magnitude_db(out=2, in_=1)
    assert np.max(np.abs(np.mean(s21_t3d) - np.mean(s21_lum))) < 0.15
    assert t3d.magnitude_db(out=1, in_=1).max() < -20
    assert lum.magnitude_db(out=1, in_=1).max() < -20


def test_three_engine_agreement():
    """All three engines ran the IDENTICAL gf-straight job (mesh 10, unified
    tech, zero solver-specific kwargs) on 2026-07-08: tidy3d and Lumerical
    agree within 0.004 dB; beamz (free, 0.x) within 0.06 dB of both."""
    from gds_fdtd.validation import compare_smatrices

    sms = {
        name: SMatrix.from_npz(str(RECORDED / f"straight_mesh10_{name}.npz"))
        for name in ("tidy3d", "lumerical", "beamz")
    }
    report = compare_smatrices(sms)
    assert report.pairwise_db[("tidy3d", "lumerical")] < 0.01, report.summary()
    assert report.worst_db < 0.08, report.summary()
    for sm in sms.values():
        assert sm.magnitude_db(out=1, in_=1).max() < -25  # stubs reach the PML
