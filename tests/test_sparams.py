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
