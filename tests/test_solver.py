"""Tests for gds_fdtd.solver base class (WP1.2: bugs B5/B17)."""

from __future__ import annotations

import pathlib

import pytest

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solver import fdtd_port, fdtd_solver

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture
def escalator_component():
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_lumerical.yaml"))
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp
    del layout


def _make_solver(comp, tmp_path, **kwargs):
    # fdtd_solver's "abstract" methods don't use ABCMeta, so the base class is
    # directly instantiable — which conveniently exercises the full base __init__.
    return fdtd_solver(component=comp, tech=None, working_dir=str(tmp_path), **kwargs)


# ---------- B5: port_input default ----------


def test_default_port_input_activates_all_ports(escalator_component, tmp_path):
    solver = _make_solver(escalator_component, tmp_path)
    names = [p.name for p in escalator_component.ports]
    assert [p.name for p in solver.port_input] == names
    assert solver._get_active_ports() == names


def test_single_port_input_normalized_to_list(escalator_component, tmp_path):
    p0 = escalator_component.ports[0]
    solver = _make_solver(escalator_component, tmp_path, port_input=p0)
    assert solver._get_active_ports() == [p0.name]


def test_port_input_list_passthrough(escalator_component, tmp_path):
    ports = [escalator_component.ports[0]]
    solver = _make_solver(escalator_component, tmp_path, port_input=ports)
    assert solver._get_active_ports() == [ports[0].name]


def test_invalid_port_input_raises(escalator_component, tmp_path):
    with pytest.raises(ValueError, match="Invalid port object"):
        _make_solver(escalator_component, tmp_path, port_input=[42])


# ---------- B17: no shared mutable defaults ----------


def test_solver_instances_share_no_mutable_state(escalator_component, tmp_path):
    s1 = _make_solver(escalator_component, tmp_path / "a")
    s2 = _make_solver(escalator_component, tmp_path / "b")
    s1.boundary.append("Metal")
    s1.modes.append(99)
    s1.symmetry[0] = 1
    s1.field_monitors.append("x")
    assert s2.boundary == ["PML", "PML", "PML"]
    assert s2.modes == [1]
    assert s2.symmetry == [0, 0, 0]
    assert s2.field_monitors == ["z"]


def test_fdtd_port_instances_share_no_mutable_state():
    p1 = fdtd_port(name="opt1")
    p2 = fdtd_port(name="opt2")
    p1.position[0] = 123.0
    p1.modes.append(7)
    p1.span[1] = 99.0
    assert p2.position == [0.0, 0.0, 0.0]
    assert p2.modes == [0]
    assert p2.span == [None, 2.5, 1.5]


# ---------- WP1.7 (B16): library must not touch the root logger ----------


def test_solver_construction_leaves_root_logger_alone(escalator_component, tmp_path):
    import logging

    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    level_before = root.level
    handlers_before = list(root.handlers)
    try:
        _make_solver(escalator_component, tmp_path)
        assert root.handlers == handlers_before, "solver construction modified root handlers"
        assert root.level == level_before, "solver construction modified root level"
        pkg = logging.getLogger("gds_fdtd")
        assert pkg.propagate is False
        assert any(isinstance(h, logging.FileHandler) for h in pkg.handlers)
    finally:
        root.removeHandler(sentinel)
        # close package handlers so tmp log files release cleanly on Windows
        pkg = logging.getLogger("gds_fdtd")
        for h in pkg.handlers[:]:
            pkg.removeHandler(h)
            h.close()
