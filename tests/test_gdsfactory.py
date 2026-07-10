"""Tests for the gdsfactory (>=9) converter (WP4.2; kills B2/B3/B4).

Skipped when gdsfactory is not installed; the all-extras CI leg runs them.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

gf = pytest.importorskip("gdsfactory")

from gds_fdtd.layout.gdsfactory import from_gdsfactory  # noqa: E402
from gds_fdtd.technology import Technology  # noqa: E402

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="module", autouse=True)
def _activate_pdk():
    # gf 9.44+ requires an active PDK to build components (caller's job)
    gf.gpdk.PDK.activate()


@pytest.fixture(scope="module")
def tech():
    return Technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))


def test_straight_ports_um_and_directions(tech):
    """B3/B4: ports carry their OWN names, um coordinates, snapped directions."""
    comp = from_gdsfactory(gf.components.straight(length=10), tech)
    assert len(comp.ports) == 2
    by_name = {p.name: p for p in comp.ports}
    assert set(by_name) == {"o1", "o2"}  # not the component name (B3)
    assert by_name["o1"].center[:2] == [0.0, 0.0]
    assert by_name["o2"].center[:2] == [10.0, 0.0]  # um, not nm/dbu (B4)
    assert by_name["o1"].width == pytest.approx(0.5)
    assert by_name["o1"].direction == 180 and by_name["o2"].direction == 0
    # z from the (1,0) device layer: 0 + 0.22/2
    assert by_name["o1"].center[2] == pytest.approx(0.11)
    assert by_name["o1"].idx == 1 and by_name["o2"].idx == 2


def test_straight_polygons_per_layer(tech):
    """B2: every polygon converts (no hardcoded index), on the right layer."""
    comp = from_gdsfactory(gf.components.straight(length=10), tech)
    device = [s for s in comp.structures if s.role == "device"]
    assert len(device) >= 1
    for s in device:
        assert s.layer == [1, 0]
        arr = np.asarray(s.polygon)
        assert arr[:, 0].min() == pytest.approx(0.0) and arr[:, 0].max() == pytest.approx(10.0)
        assert s.z_span == pytest.approx(0.22) and s.sidewall_angle == 85
    roles = {s.role for s in comp.structures}
    assert roles == {"device", "substrate", "superstrate"}


def test_bounds_ports_on_edge_margin_elsewhere(tech):
    """Port planes get NO margin (devrec convention — stubs must reach the
    PML); port-free sides get the 1.9 um evanescent margin."""
    comp = from_gdsfactory(gf.components.straight(length=10), tech)
    assert comp.bounds.x_min == pytest.approx(0.0)  # port o1 plane
    assert comp.bounds.x_max == pytest.approx(10.0)  # port o2 plane
    assert comp.bounds.y_min == pytest.approx(-0.25 - 1.9)
    assert comp.bounds.y_max == pytest.approx(0.25 + 1.9)
    assert comp.bounds.z_center == pytest.approx(0.11)


def test_bend_circular_smoke(tech):
    comp = from_gdsfactory(gf.components.bend_circular(), tech)
    assert len(comp.ports) == 2
    dirs = sorted(p.direction for p in comp.ports)
    assert dirs == [90, 180] or dirs == [180, 270]
    assert all(p.height == pytest.approx(0.22) for p in comp.ports)


def test_component_without_tech_layers_raises(tech):
    import copy

    bad_tech = copy.deepcopy(tech)
    for d in bad_tech["device"]:
        d["layer"] = [42, 7]
    with pytest.raises(ValueError, match="no polygons on any technology device layer"):
        from_gdsfactory(gf.components.straight(length=10), bad_tech)


def test_gf_component_flows_into_solver_build(tech, tmp_path):
    """End-to-end offline: gdsfactory -> Component -> LumericalSolver.build()."""
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec

    comp = from_gdsfactory(gf.components.straight(length=10), tech)
    solver = get_solver("lumerical")(
        comp,
        technology=tech,
        spec=SimulationSpec(z_min=-1.0, z_max=1.11),
        workdir=str(tmp_path),
    )
    assert solver.validate() == []
    art = solver.build()
    assert 'set("name", "o1");' in art.native
    assert '"layer number", "1:0"' in art.native
    assert art.files["gds"].exists()


def test_legacy_simprocessor_delegate(tech):
    from gds_fdtd.simprocessor import from_gdsfactory as legacy

    comp = legacy(gf.components.straight(length=10), tech)
    assert {p.name for p in comp.ports} == {"o1", "o2"}
