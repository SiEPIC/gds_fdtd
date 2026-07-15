"""Property-based tests (hypothesis) for the numeric core (WS2).

Random S-matrices, geometries, and technologies must satisfy the package's
structural invariants: lossless I/O round-trips, reciprocity/passivity
detection, port-extension geometry, and v1<->v2 technology equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gds_fdtd.geometry import Port
from gds_fdtd.smatrix import SMatrix
from gds_fdtd.technology import Technology

# ---------------------------------------------------------------- strategies


@st.composite
def smatrices(draw: st.DrawFn) -> SMatrix:
    n_ports = draw(st.integers(min_value=1, max_value=4))
    n_modes = draw(st.integers(min_value=1, max_value=2))
    n_f = draw(st.integers(min_value=2, max_value=8))
    f = np.linspace(1.8e14, 2.0e14, n_f)
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    mag = rng.uniform(0.0, 1.0, size=(n_f, n_ports, n_ports, n_modes, n_modes))
    phase = rng.uniform(-np.pi, np.pi, size=mag.shape)
    s = mag * np.exp(1j * phase)
    names = [f"opt{i + 1}" for i in range(n_ports)]
    return SMatrix(f=f, s=s, port_names=names, name="prop")


# ---------------------------------------------------------------- round-trips


@given(sm=smatrices())
@settings(max_examples=25, deadline=None)
def test_npz_roundtrip_lossless(tmp_path_factory: pytest.TempPathFactory, sm: SMatrix) -> None:
    path = str(tmp_path_factory.mktemp("npz") / "sm.npz")
    sm.to_npz(path)
    back = SMatrix.from_npz(path)
    assert back.port_names == sm.port_names
    np.testing.assert_allclose(back.f, sm.f)
    np.testing.assert_allclose(back.s, sm.s, rtol=0, atol=0)


@given(sm=smatrices())
@settings(max_examples=15, deadline=None)
def test_dat_roundtrip_close(tmp_path_factory: pytest.TempPathFactory, sm: SMatrix) -> None:
    """INTERCONNECT .dat encodes magnitude+phase as text: near-lossless."""
    path = str(tmp_path_factory.mktemp("dat") / "sm.dat")
    sm.to_dat(path)
    back = SMatrix.from_dat(path)
    # .dat normalizes names to "port N"; identity is by trailing-digit id
    assert back.n_ports == sm.n_ports and back.n_modes == sm.n_modes
    for pid_out in range(1, sm.n_ports + 1):
        for pid_in in range(1, sm.n_ports + 1):
            for mo in range(1, sm.n_modes + 1):
                for mi in range(1, sm.n_modes + 1):
                    np.testing.assert_allclose(
                        back.sel(pid_out, pid_in, mo, mi),
                        sm.sel(pid_out, pid_in, mo, mi),
                        rtol=1e-9,
                        atol=1e-12,
                    )


@given(sm=smatrices())
@settings(max_examples=10, deadline=None)
def test_hdf5_roundtrip_lossless(tmp_path_factory: pytest.TempPathFactory, sm: SMatrix) -> None:
    pytest.importorskip("h5py")
    path = str(tmp_path_factory.mktemp("h5") / "sm.h5")
    sm.to_hdf5(path)
    back = SMatrix.from_hdf5(path)
    np.testing.assert_allclose(back.s, sm.s)


@given(sm=smatrices())
@settings(max_examples=10, deadline=None)
def test_touchstone_flattens_port_modes(
    tmp_path_factory: pytest.TempPathFactory, sm: SMatrix
) -> None:
    n = sm.n_ports * sm.n_modes
    path = str(tmp_path_factory.mktemp("snp") / f"sm.s{n}p")
    out = sm.to_touchstone(path)
    lines = [
        ln for ln in open(out).read().splitlines() if ln.strip() and not ln.startswith(("!", "#"))
    ]
    # Touchstone v1: F rows, each row group carrying n*n complex entries
    values = sum(len(ln.split()) for ln in lines)
    assert values == sm.f.size * (1 + 2 * n * n)


# ------------------------------------------------------------ physics checks


@given(sm=smatrices())
@settings(max_examples=25, deadline=None)
def test_symmetrized_matrix_is_reciprocal(sm: SMatrix) -> None:
    F, P, _, M, _ = sm.s.shape
    flat = sm.s.transpose(0, 1, 3, 2, 4).reshape(F, P * M, P * M)
    sym = 0.5 * (flat + np.transpose(flat, (0, 2, 1)))
    s_sym = sym.reshape(F, P, M, P, M).transpose(0, 1, 3, 2, 4)
    rec = SMatrix(f=sm.f, s=s_sym, port_names=sm.port_names, name="sym")
    assert rec.is_reciprocal(atol=1e-9)


@given(sm=smatrices(), gain=st.floats(min_value=3.0, max_value=10.0))
@settings(max_examples=25, deadline=None)
def test_scaled_up_matrix_is_not_passive(sm: SMatrix, gain: float) -> None:
    hot = SMatrix(f=sm.f, s=sm.s * gain + gain, port_names=sm.port_names, name="hot")
    assert not hot.is_passive(atol=1e-6)


@given(sm=smatrices())
@settings(max_examples=25, deadline=None)
def test_power_balance_matches_manual_sum(sm: SMatrix) -> None:
    pb = sm.power_balance()
    F, P, _, M, _ = sm.s.shape
    flat = np.abs(sm.s.transpose(0, 1, 3, 2, 4).reshape(F, P * M, P * M)) ** 2
    manual = flat.sum(axis=1)  # sum over outputs per (f, input)
    np.testing.assert_allclose(pb, manual, rtol=1e-12)


# ------------------------------------------------------------------ geometry


@given(
    direction=st.sampled_from([0, 90, 180, 270]),
    width=st.floats(min_value=0.2, max_value=3.0),
    buffer=st.floats(min_value=0.1, max_value=5.0),
    cx=st.floats(min_value=-50, max_value=50),
    cy=st.floats(min_value=-50, max_value=50),
)
@settings(max_examples=50, deadline=None)
def test_port_extension_extends_outward_by_buffer(
    direction: int, width: float, buffer: float, cx: float, cy: float
) -> None:
    p = Port(name="opt1", center=[cx, cy, 0.11], width=width, direction=direction)
    poly = np.asarray(p.polygon_extension(buffer=buffer))
    assert poly.shape == (4, 2)
    xs, ys = poly[:, 0], poly[:, 1]
    if direction == 0:  # facing +x: stub extends beyond center in +x
        assert xs.max() == pytest.approx(cx + buffer)
        assert ys.max() - ys.min() == pytest.approx(width)
    elif direction == 180:
        assert xs.min() == pytest.approx(cx - buffer)
        assert ys.max() - ys.min() == pytest.approx(width)
    elif direction == 90:
        assert ys.max() == pytest.approx(cy + buffer)
        assert xs.max() - xs.min() == pytest.approx(width)
    else:
        assert ys.min() == pytest.approx(cy - buffer)
        assert xs.max() - xs.min() == pytest.approx(width)


# ---------------------------------------------------------------- technology


@given(
    nk_si=st.floats(min_value=1.5, max_value=4.0),
    nk_clad=st.floats(min_value=1.0, max_value=1.6),
    z_span=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=25, deadline=None)
def test_technology_v2_expands_to_v1_dict(nk_si: float, nk_clad: float, z_span: float) -> None:
    v2 = {
        "name": "prop",
        "schema_version": 2,
        "materials": {
            "Si": {"nk": nk_si},
            "SiO2": {"nk": nk_clad},
        },
        "substrate": {"z_base": 0.0, "z_span": -2, "material": "SiO2"},
        "superstrate": {"z_base": 0.0, "z_span": 3, "material": "SiO2"},
        "pinrec": [{"layer": [1, 10]}],
        "devrec": [{"layer": [68, 0]}],
        "device": [
            {"layer": [1, 0], "z_base": 0.0, "z_span": z_span, "material": "Si"},
        ],
    }
    tech = Technology.model_validate(v2)
    d = tech.to_solver_dict()
    assert d["device"][0]["material"]["nk"] == pytest.approx(nk_si)
    assert d["substrate"][0]["material"]["nk"] == pytest.approx(nk_clad)
    assert d["device"][0]["z_span"] == pytest.approx(z_span)
    assert tuple(d["device"][0]["layer"]) == (1, 0)
