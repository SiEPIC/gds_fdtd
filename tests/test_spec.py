"""Tests for gds_fdtd.spec.SimulationSpec (WP3.1a)."""

from __future__ import annotations

import pytest

from gds_fdtd.spec import SimulationSpec


def test_defaults_match_legacy_solver_defaults():
    s = SimulationSpec()
    assert (s.wavelength_start, s.wavelength_end, s.wavelength_points) == (1.5, 1.6, 100)
    assert s.mesh == 10 and s.boundary == ("PML", "PML", "PML")
    assert s.symmetry == (0, 0, 0) and (s.z_min, s.z_max) == (-1.0, 1.0)
    assert (s.width_ports, s.depth_ports, s.buffer) == (2.0, 1.5, 1.0)
    assert s.modes == (1,) and s.mode_freq_pts == 3 and s.run_time_factor == 3.0
    assert s.field_monitors == ("z",)
    assert s.wavelength_center_um == pytest.approx(1.55)
    assert s.z_span == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"wavelength_start": 1.6, "wavelength_end": 1.5}, "wavelength_start"),
        ({"wavelength_points": 1}, "wavelength_points"),
        ({"z_min": 2.0, "z_max": 1.0}, "z_min"),
        ({"width_ports": 0}, "width_ports"),
        ({"buffer": -1}, "buffer"),
        ({"modes": []}, "modes"),
        ({"modes": [0]}, "modes"),
        ({"mesh": 0}, "mesh"),
        ({"symmetry": (0, 2, 0)}, "symmetry"),
        ({"boundary": ("PML", "PML", "Bogus")}, "Unsupported boundary"),
        ({"field_monitors": ("q",)}, "field_monitors"),
    ],
)
def test_validators(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SimulationSpec(**kwargs)


def test_boundary_case_normalization():
    s = SimulationSpec(boundary=("pml", "METAL", "Periodic"))
    assert s.boundary == ("PML", "Metal", "Periodic")


def test_extra_keys_rejected():
    with pytest.raises(ValueError):
        SimulationSpec(mesh_size=5)


def test_engine_base_exposes_spec(tmp_path):
    """The Tidy3D engine base builds a spec and mirrors it into legacy attrs."""
    import pathlib

    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.simprocessor import load_component_from_tech
    from tests.test_solver import _make_solver
    from tests.test_solver import escalator_component as _  # noqa: F401  (fixture reuse)

    tests_dir = pathlib.Path(__file__).parent
    tech = parse_yaml_tech(str(tests_dir / "tech_lumerical.yaml"))
    cell, layout = load_cell(str(tests_dir / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)

    solver = _make_solver(comp, tmp_path, boundary=["pml", "Metal", "PML"], modes=[1, 2])
    assert solver.spec.boundary == ("PML", "Metal", "PML")
    assert solver.boundary == ["PML", "Metal", "PML"]  # legacy list mirror
    assert solver.modes == [1, 2]
    # invalid params now fail at construction, through the same spec
    with pytest.raises(ValueError, match="wavelength_start"):
        _make_solver(comp, tmp_path, wavelength_start=2.0, wavelength_end=1.0)
    del layout
