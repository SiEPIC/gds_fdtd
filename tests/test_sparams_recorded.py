"""Legacy sparameters pipeline exercised on RECORDED real engine output.

tests/recorded/si_sin_escalator.dat is genuine Lumerical-format data written
from a real tidy3d run; parsing it covers the .dat reader, the accessors,
the excitation analysis, and every plot flavor - all offline.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gds_fdtd._sparams import process_dat

RECORDED = pathlib.Path(__file__).parent / "recorded"


@pytest.fixture(scope="module")
def spar():
    return process_dat(str(RECORDED / "si_sin_escalator.dat"), verbose=True)


def test_parse_recorded_dat(spar):
    assert len(spar.data) == 4  # full 2-port, 1-mode matrix
    wavl = spar.wavelength  # property
    assert wavl.min() == pytest.approx(1.5, abs=0.01)
    assert wavl.max() == pytest.approx(1.6, abs=0.01)


def test_S_accessor_hit_and_miss(spar):
    thru = spar.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)
    assert thru is not None
    assert 10 * np.log10(max(np.abs(thru.s_mag)) ** 2) > -1.0  # escalator ~ 0 dB
    assert spar.S(in_port=9, out_port=9) is None  # miss logs a warning


def test_sparam_entry_properties_and_plot(spar):
    entry = spar.S(in_port=1, out_port=2)
    assert entry.in_port_num == 1 and entry.out_port_num == 2
    assert entry.wavl.shape == np.asarray(entry.f).shape
    assert "1" in entry.idn
    fig, _ = entry.plot()
    plt.close(fig)


@pytest.mark.parametrize("kind", ["log", "phase", "linear", "not_a_kind"])
def test_sparameters_plot_flavors(spar, kind):
    result = spar.plot(plot_type=kind)  # invalid kind falls back to log
    plt.close("all")
    del result


def test_excitation_analysis_roundtrip(spar):
    analysis = spar.analyze_excitations(verbose=True)
    assert len(analysis["non_zero_entries"]) + len(analysis["zero_entries"]) == 4
    expected = spar.get_expected_excitations([1], [1])
    assert isinstance(expected, (list, set))
    report = spar.validate_excitations([1], [1], verbose=True)
    assert report is not None


def test_to_smatrix_matches_hdf5(spar):
    pytest.importorskip("h5py")
    from gds_fdtd.smatrix import SMatrix

    sm = spar.to_smatrix()
    ref = SMatrix.from_hdf5(str(RECORDED / "si_sin_escalator_smatrix.h5"))
    np.testing.assert_allclose(
        np.abs(sm.sel(out=2, in_=1)), np.abs(ref.sel(out=2, in_=1)), rtol=1e-6
    )
