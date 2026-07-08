"""Offline tests for gds_fdtd.solver_tidy3d (WP1.5: B9, B10, B11).

These construct the full solver setup — including the td.Simulation and the
ComponentModeler — WITHOUT any network access or cloud credits. Skipped when
tidy3d is not installed (the all-extras CI leg runs them).
"""

from __future__ import annotations

import pathlib

import pytest

td = pytest.importorskip("tidy3d")

from gds_fdtd.core import parse_yaml_tech  # noqa: E402
from gds_fdtd.lyprocessor import load_cell  # noqa: E402
from gds_fdtd.simprocessor import load_component_from_tech  # noqa: E402
from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d  # noqa: E402

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def solver(tmp_path_factory):
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_tidy3d.yaml"))
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    s = fdtd_solver_tidy3d(
        component=comp,
        tech=tech,
        boundary=["PML", "Metal", "Periodic"],  # deliberately mixed (B10)
        z_min=-1.0,
        z_max=1.11,
        working_dir=str(tmp_path_factory.mktemp("t3d")),
    )
    yield s
    del layout


def test_boundary_spec_honored(solver):
    """B10: the user's boundary list must reach the td.Simulation."""
    bs = solver.base_simulation.boundary_spec
    assert isinstance(bs.x.plus, td.PML)
    assert isinstance(bs.y.plus, td.PECBoundary)
    assert isinstance(bs.z.plus, td.Periodic)


def test_unknown_boundary_raises():
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_tidy3d.yaml"))
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    with pytest.raises(ValueError, match="Unsupported boundary"):
        fdtd_solver_tidy3d(
            component=comp, tech=tech, boundary=["PML", "PML", "Bogus"], working_dir="/tmp"
        )
    del layout


def test_run_time_includes_group_index(solver):
    """B11: run_time must match the shared base-class calculation."""
    expected = solver._calculate_simulation_time(max(solver.span) * 1e-6)
    assert solver.base_simulation.run_time == pytest.approx(expected, rel=1e-12)


def test_field_monitor_centered_on_component(solver):
    """B9: monitors must track the component center, not the origin."""
    mon = solver.base_simulation.monitors[0]
    assert mon.center[0] == pytest.approx(solver.component.bounds.x_center)
    assert mon.center[1] == pytest.approx(solver.component.bounds.y_center)
    # escalator's devrec spans x in [0, 18] -> center is decisively off-origin
    assert abs(solver.component.bounds.x_center) > 1


def test_background_structures_centered_on_component(solver):
    """B9: substrate/superstrate rectangles track the component center."""
    import numpy as np

    b = solver.component.bounds
    checked = 0
    for structure in solver.base_simulation.structures:
        if solver._is_background(structure.name):
            verts = np.asarray(structure.geometry.vertices)
            assert verts[:, 0].mean() == pytest.approx(b.x_center, abs=1e-9)
            assert verts[:, 1].mean() == pytest.approx(b.y_center, abs=1e-9)
            checked += 1
    assert checked == 2  # substrate + superstrate


def test_setup_is_offline(solver):
    """Constructing the solver must produce a serializable simulation without
    any cloud interaction (the artifact CI tests depend on this)."""
    js = solver.base_simulation.json()
    assert len(js) > 100
    assert len(solver.smatrix_ports) == len(solver.component.ports)
