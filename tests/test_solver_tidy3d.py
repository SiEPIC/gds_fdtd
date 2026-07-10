"""Offline tests for the Tidy3D engine (gds_fdtd.solvers._tidy3d_engine).

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
from gds_fdtd.solvers._tidy3d_engine import _TidyEngine  # noqa: E402

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def solver(tmp_path_factory):
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_tidy3d.yaml"))
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    s = _TidyEngine(
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
        _TidyEngine(
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
        if solver._is_background(structure):  # WP2.3: takes the object (role or name fallback)
            verts = np.asarray(structure.geometry.vertices)
            assert verts[:, 0].mean() == pytest.approx(b.x_center, abs=1e-9)
            assert verts[:, 1].mean() == pytest.approx(b.y_center, abs=1e-9)
            checked += 1
    assert checked == 2  # substrate + superstrate


def test_setup_is_offline(solver):
    """Constructing the solver must produce a serializable simulation without
    any cloud interaction (the artifact CI tests depend on this)."""
    js = (
        solver.base_simulation.model_dump_json()
        if hasattr(solver.base_simulation, "model_dump_json")
        else solver.base_simulation.json()
    )
    assert len(js) > 100
    assert len(solver.smatrix_ports) == len(solver.component.ports)


def test_web_submodule_resolves_in_fresh_interpreter():
    """Regression for finding F10: tidy3d.web is a LAZY submodule — `import
    tidy3d as td` does not provide td.web. The validation scripts masked this
    by importing tidy3d.web themselves; a user running an example hit
    AttributeError. Run the check in a FRESH interpreter so nothing in this
    test session can mask it, and statically forbid the td.web idiom.
    """
    import inspect
    import subprocess
    import sys

    import gds_fdtd.solvers._tidy3d_engine as engine
    import gds_fdtd.solvers.tidy3d as adapter

    # static: the lazy-import-unsafe idiom must not reappear
    for module in (engine, adapter):
        assert "td.web" not in inspect.getsource(module).replace("tidy3d.web", ""), (
            f"{module.__name__} uses td.web — tidy3d.web must be imported explicitly"
        )

    # dynamic, unmasked: fresh interpreter imports the solver module only, then
    # verifies the exact import run() performs is resolvable
    code = (
        "import gds_fdtd.solvers._tidy3d_engine\n"
        "import tidy3d.web as web\n"
        "assert callable(web.run)\n"
        "print('web-ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert "web-ok" in out.stdout
