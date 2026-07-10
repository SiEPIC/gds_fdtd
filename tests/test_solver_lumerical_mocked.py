"""Modern LumericalSolver adapter, exercised offline.

``build()`` generates the ``.lsf`` setup script as text with **no lumapi**;
``run()`` opens a lumapi session (the recording mock in
``tests/mocks/lumapi.py``) and replays the script, producing an ``SMatrix``
from a recorded real ``.dat``. Replaces the pre-0.6 tests of the removed
``fdtd_solver_lumerical`` class; the F6/F7 findings are covered against the
modern adapter's ``run()`` path.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology

from .mocks.lumapi import install

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture()
def mock_lumapi(monkeypatch):
    return install(monkeypatch)


@pytest.fixture()
def escalator():
    tech = Technology.from_yaml(str(TESTS_DIR / "tech_unified.yaml"))
    cell, layout = load_cell(
        str(TESTS_DIR.parent / "examples" / "devices.gds"),
        top_cell="si_sin_escalator_te1550",
    )
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp, tech
    del layout


def _solver(escalator, tmp_path, **spec_kw):
    comp, tech = escalator
    spec = SimulationSpec(wavelength_points=5, mesh=4, z_min=-1.0, z_max=1.11, **spec_kw)
    return get_solver("lumerical")(comp, tech, spec, workdir=tmp_path)


def test_build_generates_lsf_offline(escalator, tmp_path):
    """build() is pure offline .lsf generation - no lumapi, no license."""
    solver = _solver(escalator, tmp_path)
    assert solver.validate() == []
    lsf = solver.build().native
    assert isinstance(lsf, str)
    assert "addlayerbuilder" in lsf
    assert lsf.count("addlayer(") >= 3
    assert "Si (Silicon) - Palik" in lsf
    assert "Si3N4 (Silicon Nitride) - Luke" in lsf
    assert 'setnamed("FDTD"' in lsf
    assert "sparams" in lsf


def test_build_is_deterministic(escalator, tmp_path):
    a = _solver(escalator, tmp_path / "a").build().native
    b = _solver(escalator, tmp_path / "b").build().native
    assert a == b


def test_run_via_mock_produces_smatrix(mock_lumapi, escalator, tmp_path):
    """run() replays the .lsf through the mock and parses the recorded
    escalator .dat into an SMatrix."""
    solver = _solver(escalator, tmp_path)
    sm = solver.run()
    assert sm.n_ports == 2
    thru = float(sm.magnitude_db(out=2, in_=1).max())
    assert np.isfinite(thru) and thru > -1.0
    names = [c[0] for c in mock_lumapi.INSTANCES[-1].calls]
    assert "eval" in names and "runsweep" in names and "exportsweep" in names


def test_run_device_type_v2025_then_2024_fallback(mock_lumapi, escalator, tmp_path):
    """F7: run() tries the 2025 setresource signature first, falls back to 2024."""
    mock_lumapi.RAISE_ON_4ARG_SETRESOURCE = True
    _solver(escalator, tmp_path).run()
    setres = [c[1] for c in mock_lumapi.INSTANCES[-1].calls if c[0] == "setresource"]
    assert any(len(a) == 4 for a in setres)
    assert any(len(a) == 3 for a in setres)


def test_run_gpu_flag_selects_device(mock_lumapi, escalator, tmp_path):
    solver = _solver(escalator, tmp_path)
    solver.gpu = True
    solver.run()
    setres = [c[1] for c in mock_lumapi.INSTANCES[-1].calls if c[0] == "setresource"]
    assert any("GPU" in a for a in setres)
