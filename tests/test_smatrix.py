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
            [
                ("opt1", "opt2", 1, 1, F, np.ones(F.size)),
                ("opt2", "opt1", 1, 1, F * 2, np.ones(F.size)),
            ]
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


# ---------- WP2.4b: .dat interop + legacy-class bridge ----------


def test_dat_round_trip_via_smatrix(tmp_path):
    sm = SMatrix.from_entries(_entries_2port(), name="dut")
    path = sm.to_dat(str(tmp_path / "dut.dat"))
    back = SMatrix.from_dat(path)
    # names normalize to "port N"; numeric ids and values survive exactly
    np.testing.assert_allclose(back.sel(out=2, in_=1), sm.sel(out=2, in_=1), rtol=1e-9)
    np.testing.assert_allclose(back.sel(out=1, in_=1), sm.sel(out=1, in_=1), rtol=1e-9)
    assert back.n_ports == sm.n_ports and back.n_modes == sm.n_modes


def test_dat_skips_nan_paths(tmp_path):
    sm = SMatrix.from_entries([("opt1", "opt2", 1, 1, F, np.ones(F.size))])
    path = sm.to_dat(str(tmp_path / "partial.dat"))
    back = SMatrix.from_dat(path)
    assert np.all(np.isnan(back.sel(out=1, in_=2)))  # reverse path stayed unmeasured
    np.testing.assert_allclose(np.abs(back.sel(out=2, in_=1)), 1.0)


def test_sparameters_to_smatrix_equivalence(tmp_path):
    """The legacy container and the canonical one agree entry-by-entry."""
    from tests.test_sparams import _make_multimode_sparams

    spar = _make_multimode_sparams()
    sm = spar.to_smatrix()
    for d in spar.data:
        got = sm.sel(
            out=d.out_port_num, in_=d.in_port_num, mode_out=d.out_mode_num, mode_in=d.in_mode_num
        )
        expected = np.asarray(d.s_mag) * np.exp(1j * np.asarray(d.s_phase))
        np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---------- WP2.4c: Touchstone export, validated with scikit-rf ----------


def test_touchstone_2port_read_back_with_skrf(tmp_path):
    skrf = pytest.importorskip("skrf")
    sm = SMatrix.from_entries(_entries_2port(), name="dut")
    path = str(tmp_path / "dut.s2p")
    sm.to_touchstone(path)
    nw = skrf.Network(path)
    np.testing.assert_allclose(nw.f, sm.f)
    np.testing.assert_allclose(nw.s[:, 1, 0], sm.sel(out="opt2", in_="opt1"), rtol=1e-9)
    np.testing.assert_allclose(nw.s[:, 0, 1], sm.sel(out="opt1", in_="opt2"), rtol=1e-9)
    np.testing.assert_allclose(nw.s[:, 0, 0], sm.sel(out="opt1", in_="opt1"), rtol=1e-9)


def test_touchstone_multimode_flattening_convention(tmp_path):
    """2 ports x 2 modes -> .s4p; Touchstone port k = (p, m), port-major."""
    skrf = pytest.importorskip("skrf")
    rng = np.random.default_rng(7)
    entries = []
    for pi, po in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        for mi, mo in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            entries.append(
                (f"opt{pi}", f"opt{po}", mi, mo, F, rng.normal(size=F.size) * 0.1 + 0.2j)
            )
    sm = SMatrix.from_entries(entries, name="mm")
    path = str(tmp_path / "mm.s4p")
    sm.to_touchstone(path)
    nw = skrf.Network(path)
    # touchstone port index k = (port_idx)*M + mode_idx  (0-based)
    np.testing.assert_allclose(
        nw.s[:, 2, 1],  # (opt2,m1) <- (opt1,m2)
        sm.sel(out="opt2", in_="opt1", mode_out=1, mode_in=2),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        nw.s[:, 3, 0],  # (opt2,m2) <- (opt1,m1)
        sm.sel(out="opt2", in_="opt1", mode_out=2, mode_in=1),
        rtol=1e-9,
    )


def test_touchstone_rejects_nan_and_wrong_extension(tmp_path):
    partial = SMatrix.from_entries([("opt1", "opt2", 1, 1, F, np.ones(F.size))])
    with pytest.raises(ValueError, match="NaN"):
        partial.to_touchstone(str(tmp_path / "x.s2p"))
    full = SMatrix.from_entries(_entries_2port())
    with pytest.raises(ValueError, match=r"\.s2p"):
        full.to_touchstone(str(tmp_path / "x.s3p"))


# ---------- WP2.4d: plotting ----------


def test_plot_smatrix_kinds():
    import matplotlib

    matplotlib.use("Agg")
    from gds_fdtd.plotting import plot_smatrix

    sm = SMatrix.from_entries(_entries_2port(), name="dut")
    for kind in ("db", "linear", "phase"):
        fig, ax = plot_smatrix(sm, kind=kind)
        assert len(ax.lines) == 4  # all measured paths plotted
        assert ax.get_title().startswith("dut")
    with pytest.raises(ValueError, match="kind"):
        plot_smatrix(sm, kind="bogus")


def test_plot_component_geometry():
    """Standard geometry view: polygons per layer, ports w/ arrows, regions."""
    import pathlib

    import matplotlib

    matplotlib.use("Agg")
    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.plotting import plot_component
    from gds_fdtd.simprocessor import load_component_from_tech
    from gds_fdtd.spec import SimulationSpec

    tests_dir = pathlib.Path(__file__).parent
    tech = parse_yaml_tech(str(tests_dir / "tech_lumerical.yaml"))
    cell, layout = load_cell(str(tests_dir / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)

    fig, ax = plot_component(comp, spec=SimulationSpec())
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert any("layer 1/0" in label for label in labels)
    assert any("layer 1/5" in label for label in labels)  # both device layers drawn
    assert any("bounds" in label for label in labels)
    assert any("FDTD region" in label for label in labels)
    texts = {t.get_text() for t in ax.texts}
    assert {"opt1", "opt2"} <= texts  # port names annotated
    del layout
