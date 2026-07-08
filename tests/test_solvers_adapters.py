"""Offline tests for the ABC solver adapters (WP3.1c/d)."""

from __future__ import annotations

import pathlib

import pytest

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

TESTS_DIR = pathlib.Path(__file__).parent


def _job(tech_file: str):
    tech = parse_yaml_tech(str(TESTS_DIR / tech_file))
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    return comp, tech, layout


# ---------------- LumericalSolver: pure .lsf generation ----------------


@pytest.fixture(scope="module")
def lum_artifacts(tmp_path_factory):
    comp, tech, layout = _job("tech_lumerical.yaml")
    solver = get_solver("lumerical")(
        comp,
        technology=tech,
        spec=SimulationSpec(
            wavelength_points=11,
            mesh=6,
            z_min=-1.0,
            z_max=1.11,
            boundary=("PML", "PML", "Metal"),
            symmetry=(0, -1, 1),
            modes=(1, 2),
        ),
        workdir=str(tmp_path_factory.mktemp("lum_abc")),
    )
    art = solver.build()
    yield solver, art
    del layout


def test_lum_script_geometry_and_layers(lum_artifacts):
    _, art = lum_artifacts
    script = art.native
    assert art.files["gds"].exists() and art.files["lsf"].exists()
    # both device layers present in the escalator must be configured
    assert '"layer number", "1:0"' in script
    assert '"layer number", "1:5"' in script
    assert 'setlayer("substrate", "background material", "SiO2 (Glass) - Palik");' in script
    assert '"sidewall angle", 85' in script and '"sidewall angle", 83' in script


def test_lum_script_boundaries_and_symmetry(lum_artifacts):
    _, art = lum_artifacts
    script = art.native
    assert 'setnamed("FDTD", "z max bc", "Metal");' in script
    assert 'setnamed("FDTD", "y min bc", "Anti-Symmetric");' in script
    assert 'setnamed("FDTD", "z min bc", "Symmetric");' in script
    assert 'setnamed("FDTD", "x min bc", "PML");' in script


def test_lum_script_ports_and_sweep(lum_artifacts):
    _, art = lum_artifacts
    script = art.native
    assert script.count("addport;") == 2
    assert 'set("name", "opt1");' in script and 'set("name", "opt2");' in script
    assert "updateportmodes([1;2]);" in script
    # 2 ports x 2 modes = 4 sweep entries, all active (full matrix)
    assert script.count('addsweepparameter("sparams", entry);') == 4
    assert 'entry.Mode = "mode 2";' in script


def test_lum_build_deterministic(lum_artifacts):
    solver, art = lum_artifacts
    assert solver.build().native == art.native


def test_lum_validate_flags_missing_materials():
    comp, tech, layout = _job("tech_tidy3d.yaml")  # tidy3d materials only
    solver = get_solver("lumerical")(comp, technology=tech)
    problems = solver.validate()
    assert any("lum_db" in p for p in problems)
    del layout


# ---------------- Tidy3DSolver: validation surface ----------------


def test_t3d_validate_flags_missing_tech():
    pytest.importorskip("tidy3d")
    comp, _tech, layout = _job("tech_lumerical.yaml")
    solver = get_solver("tidy3d")(comp, technology=None)
    assert any("technology" in p for p in solver.validate())
    del layout
