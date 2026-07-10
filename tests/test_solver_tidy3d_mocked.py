"""Legacy tidy3d adapter setup path, offline via the permissive tidy3d fake.

Scene CONSTRUCTION is deterministic local logic (structures, boundaries,
ports, monitors, runtime math); the fake records it without tidy3d
installed. Real-engine behavior stays covered by the all-extras CI leg and
the recorded live validations.
"""

from __future__ import annotations

import pathlib

import pytest

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech

from .mocks.tidy3d_fake import FakeNode, install

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture()
def fake_td(monkeypatch):
    return install(monkeypatch)


@pytest.fixture()
def escalator(fake_td):
    # load AFTER the fake is installed so materials resolve through it
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_unified.yaml"))
    cell, layout = load_cell(
        str(TESTS_DIR.parent / "examples" / "devices.gds"),
        top_cell="si_sin_escalator_te1550",
    )
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp, tech
    del layout


def _make(escalator, tmp_path, **kwargs):
    from gds_fdtd.solvers._tidy3d_engine import _TidyEngine as fdtd_solver_tidy3d

    comp, tech = escalator
    defaults = {
        "component": comp,
        "tech": tech,
        "wavelength_points": 5,
        "mesh": 4,
        "z_min": -1.0,
        "z_max": 1.11,
        "working_dir": str(tmp_path),
    }
    defaults.update(kwargs)
    return fdtd_solver_tidy3d(**defaults)


def test_setup_constructs_scene_offline(fake_td, escalator, tmp_path):
    solver = _make(escalator, tmp_path)
    assert solver.base_simulation is not None
    assert isinstance(solver.component_modeler, FakeNode)
    # one Port per component port reaches the modeler
    ports = solver.component_modeler._kwargs.get("ports")
    assert ports is not None and len(ports) == 2


def test_setup_exports_gds(fake_td, escalator, tmp_path):
    solver = _make(escalator, tmp_path)
    assert pathlib.Path(solver._gds_filepath).exists()


def test_boundaries_and_symmetry_flow_through(fake_td, escalator, tmp_path):
    solver = _make(escalator, tmp_path, boundary=["PML", "Metal", "Periodic"], symmetry=[0, 0, 1])
    sim_kwargs = solver.base_simulation._kwargs
    assert tuple(sim_kwargs.get("symmetry", ())) == (0, 0, 1)


def test_run_time_uses_group_index_helper(fake_td, escalator, tmp_path):
    a = _make(escalator, tmp_path, run_time_factor=3)
    b = _make(escalator, tmp_path, run_time_factor=6)
    assert b.base_simulation._kwargs["run_time"] == pytest.approx(
        2 * a.base_simulation._kwargs["run_time"]
    )


def test_field_monitor_optional(fake_td, escalator, tmp_path):
    with_mon = _make(escalator, tmp_path, field_monitors=["z"])
    without = _make(escalator, tmp_path, field_monitors=[])
    n_with = len(with_mon.base_simulation._kwargs.get("monitors", ()))
    n_without = len(without.base_simulation._kwargs.get("monitors", ()))
    assert n_with == n_without + 1


def test_modern_adapter_offline_lifecycle(fake_td, escalator, tmp_path):
    """The MODERN Tidy3DSolver validate/build/estimate are offline by
    contract - the fake proves no tidy3d install is needed for them."""
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec

    comp, tech = escalator
    solver = get_solver("tidy3d")(
        comp,
        tech,
        SimulationSpec(wavelength_points=5, mesh=4, z_min=-1, z_max=1.11),
        workdir=tmp_path,
    )
    assert solver.validate() == []
    artifacts = solver.build()
    assert artifacts.summary["n_ports"] == 2
    est = solver.estimate()  # offline by contract; counts come from the fake
    assert est is not None
    assert "tidy3d" in solver.describe()


def test_modern_adapter_plot_fields_requires_run(fake_td, escalator, tmp_path):
    from gds_fdtd.errors import SolverError
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec

    comp, tech = escalator
    solver = get_solver("tidy3d")(comp, tech, SimulationSpec(), workdir=tmp_path)
    with pytest.raises((SolverError, RuntimeError), match="run"):
        solver.plot_fields(axis="z")


def test_public_alias_emits_deprecation_warning(fake_td, escalator, tmp_path):
    """The deprecated fdtd_solver_tidy3d public alias warns; the internal
    _TidyEngine (used by the adapter) does not."""
    import gds_fdtd.solver_tidy3d as legacy_mod

    comp, tech = escalator
    with pytest.warns(DeprecationWarning, match="deprecated since gds_fdtd 0.5"):
        legacy_mod.fdtd_solver_tidy3d(
            component=comp,
            tech=tech,
            wavelength_points=5,
            mesh=4,
            z_min=-1.0,
            z_max=1.11,
            working_dir=str(tmp_path),
        )
