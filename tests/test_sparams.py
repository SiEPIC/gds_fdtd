"""Tests for gds_fdtd.sparams (started in WP1.3; grows with WP1.6)."""

from __future__ import annotations

import pytest

from gds_fdtd.core import port
from gds_fdtd.sparams import _number_of, s, sparameters


def _entry(in_port, out_port, in_mode=1, out_mode=1):
    return s(
        f=[1.9e14, 2.0e14],
        s_mag=[0.5, 0.6],
        s_phase=[0.1, 0.2],
        in_port=in_port,
        out_port=out_port,
        in_modeid=in_mode,
        out_modeid=out_mode,
    )


# ---------- WP1.3 (B6): trailing-digit parsing, no concatenation ambiguity ----


@pytest.mark.parametrize(
    "value,expected",
    [("opt1", 1), ("opt10", 10), ("port 12", 12), ("port2", 2), (3, 3), ("mode 1", 1)],
)
def test_number_of(value, expected):
    assert _number_of(value) == expected


def test_number_of_rejects_digitless():
    with pytest.raises(ValueError):
        _number_of("optA")


def test_port_idx_rejects_digitless():
    p = port("optA", [0, 0, 0], 1.0, 0)
    with pytest.raises(ValueError):
        _ = p.idx


def test_s_lookup_distinguishes_two_digit_ports():
    """S(2,1) must not match the (21, ...) or (1, 12) entries — the old string
    concatenation ("21_11") could not tell these apart."""
    spar = sparameters("dut")
    e_2_1 = _entry(in_port="opt1", out_port="opt2")
    e_21_x = _entry(in_port="opt21", out_port="opt1")
    e_1_12 = _entry(in_port="opt12", out_port="opt1")
    for e in (e_1_12, e_21_x, e_2_1):
        spar.data.append(e)

    assert spar.S(in_port=1, out_port=2) is e_2_1
    assert spar.S(in_port=21, out_port=1) is e_21_x
    assert spar.S(in_port=12, out_port=1) is e_1_12


def test_s_lookup_accepts_names_and_ints():
    spar = sparameters("dut")
    e = _entry(in_port="opt1", out_port="opt2", in_mode=2, out_mode=1)
    spar.data.append(e)
    assert spar.S(in_port="opt1", out_port="opt2", in_modeid=2, out_modeid=1) is e
    assert spar.S(in_port=1, out_port=2, in_modeid=2, out_modeid=1) is e


# ---------- WP1.6 (B13, B14): .dat round trip ----------


def _make_multimode_sparams() -> sparameters:
    """2 ports x 2 modes, deliberately NOT in dense port-order, partial matrix."""
    spar = sparameters("dut")
    spar.add_port("port 1", "LEFT")
    spar.add_port("port 2", "RIGHT")
    freqs = [1.87e14, 1.93e14, 2.00e14]
    k = 0
    for in_port, out_port in [(1, 2), (2, 1), (1, 1)]:  # partial, unordered
        for in_mode, out_mode in [(1, 1), (2, 1)]:
            k += 1
            spar.add_data(
                in_port=f"port {in_port}",
                out_port=f"port {out_port}",
                mode_label=1,
                in_modeid=in_mode,
                out_modeid=out_mode,
                data_type="transmission",
                group_delay=0.0,
                f=list(freqs),
                s_mag=[0.1 * k, 0.2 * k, 0.3 * k],
                s_phase=[0.01 * k, -0.02 * k, 0.03 * k],
            )
    return spar


def test_dat_round_trip(tmp_path):
    from gds_fdtd.sparams import process_dat, write_dat

    spar = _make_multimode_sparams()
    path = tmp_path / "dut.dat"
    write_dat(spar, str(path))
    back = process_dat(str(path), verbose=False)

    assert len(back.data) == len(spar.data)
    for orig, rt in zip(spar.data, back.data, strict=True):
        # identity, not just presence: order and labeling survive the trip
        assert rt.in_port_num == orig.in_port_num
        assert rt.out_port_num == orig.out_port_num
        assert rt.in_mode_num == orig.in_mode_num
        assert rt.out_mode_num == orig.out_mode_num
        assert rt.f == pytest.approx(orig.f, rel=1e-9)
        assert rt.s_mag == pytest.approx(orig.s_mag, rel=1e-9)
        assert rt.s_phase == pytest.approx(orig.s_phase, rel=1e-9)


def test_write_dat_rejects_inconsistent_lengths(tmp_path):
    from gds_fdtd.sparams import write_dat

    spar = sparameters("bad")
    e = _entry(in_port="opt1", out_port="opt2")
    e.s_mag = e.s_mag[:-1]
    spar.data.append(e)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        write_dat(spar, str(tmp_path / "bad.dat"))
