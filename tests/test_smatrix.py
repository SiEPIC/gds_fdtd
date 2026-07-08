"""Tests for gds_fdtd.smatrix (WP2.4a)."""

from __future__ import annotations

import numpy as np
import pytest

from gds_fdtd.smatrix import SMatrix

F = np.linspace(1.87e14, 2.0e14, 5)


def _entries_2port(reciprocal=True, lossy=True):
    thru = (0.8 if lossy else 1.0) * np.exp(1j * np.linspace(0, np.pi, F.size))
    back = thru if reciprocal else thru * 0.5
    return [
        ("opt1", "opt2", 1, 1, F, thru),
        ("opt2", "opt1", 1, 1, F, back),
        ("opt1", "opt1", 1, 1, F, np.full(F.size, 0.1 + 0j)),
        ("opt2", "opt2", 1, 1, F, np.full(F.size, 0.1 + 0j)),
    ]


def test_from_entries_shape_and_sel():
    sm = SMatrix.from_entries(_entries_2port(), name="dut")
    assert sm.n_ports == 2 and sm.n_modes == 1 and sm.f.size == F.size
    np.testing.assert_allclose(np.abs(sm.sel(out="opt2", in_="opt1")), 0.8)
    # integer port id convenience (trailing digits)
    np.testing.assert_allclose(sm.sel(out=2, in_=1), sm.sel(out="opt2", in_="opt1"))
    assert sm.magnitude_db(2, 1)[0] == pytest.approx(10 * np.log10(0.64))


def test_partial_matrix_is_nan():
    sm = SMatrix.from_entries([("opt1", "opt2", 1, 1, F, np.ones(F.size))])
    assert np.all(np.isnan(sm.sel(out="opt1", in_="opt2")))  # unmeasured reverse path
    assert not np.any(np.isnan(sm.sel(out="opt2", in_="opt1")))


def test_reciprocity_check():
    assert SMatrix.from_entries(_entries_2port(reciprocal=True)).is_reciprocal()
    assert not SMatrix.from_entries(_entries_2port(reciprocal=False)).is_reciprocal()


def test_passivity_check():
    assert SMatrix.from_entries(_entries_2port(lossy=True)).is_passive()
    gain = [("opt1", "opt2", 1, 1, F, np.full(F.size, 1.2 + 0j))]
    assert not SMatrix.from_entries(gain).is_passive()


def test_multimode_entries():
    e = _entries_2port() + [("opt1", "opt2", 2, 2, F, np.full(F.size, 0.3 + 0j))]
    sm = SMatrix.from_entries(e)
    assert sm.n_modes == 2
    np.testing.assert_allclose(np.abs(sm.sel("opt2", "opt1", mode_out=2, mode_in=2)), 0.3)
    assert np.all(np.isnan(sm.sel("opt2", "opt1", mode_out=1, mode_in=2)))


def test_mismatched_frequency_grids_rejected():
    with pytest.raises(ValueError, match="frequency grid"):
        SMatrix.from_entries(
            [("opt1", "opt2", 1, 1, F, np.ones(F.size)), ("opt2", "opt1", 1, 1, F * 2, np.ones(F.size))]
        )


def test_npz_round_trip(tmp_path):
    sm = SMatrix.from_entries(_entries_2port(), name="dut")
    p = sm.to_npz(str(tmp_path / "s.npz"))
    back = SMatrix.from_npz(p)
    assert back.port_names == sm.port_names and back.name == "dut"
    np.testing.assert_allclose(back.f, sm.f)
    np.testing.assert_array_equal(np.isnan(back.s), np.isnan(sm.s))
    np.testing.assert_allclose(np.nan_to_num(back.s), np.nan_to_num(sm.s))


def test_hdf5_round_trip(tmp_path):
    pytest.importorskip("h5py")
    sm = SMatrix.from_entries(_entries_2port(), name="dut")
    p = sm.to_hdf5(str(tmp_path / "s.h5"))
    back = SMatrix.from_hdf5(p)
    assert back.port_names == sm.port_names
    np.testing.assert_allclose(np.nan_to_num(back.s), np.nan_to_num(sm.s))


def test_wavelength_grid():
    sm = SMatrix.from_entries(_entries_2port())
    assert sm.wavelength_um[0] == pytest.approx(299792458.0 / 1.87e14 * 1e6)


def test_unknown_port_raises():
    sm = SMatrix.from_entries(_entries_2port())
    with pytest.raises(KeyError, match="unknown port"):
        sm.sel(out="opt9", in_="opt1")
