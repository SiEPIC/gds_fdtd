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
    """Runs in the base (no-tidy3d) profile: strip lum_db from a lum tech copy
    instead of loading tech_tidy3d.yaml (whose materials import tidy3d)."""
    import copy

    comp, tech, layout = _job("tech_lumerical.yaml")
    bad = copy.deepcopy(tech)
    for d in bad["device"]:
        d["material"].pop("lum_db", None)
    solver = get_solver("lumerical")(comp, technology=bad)
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


# ---------------- BeamzSolver: validation surface (WP5.3) ----------------


def test_beamz_validate_requires_gf_component():
    pytest.importorskip("beamz")
    comp, tech, layout = _job("tech_tidy3d.yaml")
    solver = get_solver("beamz")(comp, technology=tech)
    problems = solver.validate()
    assert any("gdsfactory-sourced" in p for p in problems)
    del layout


def test_beamz_validate_v1_mode_restriction():
    pytest.importorskip("beamz")
    comp, tech, layout = _job("tech_tidy3d.yaml")
    solver = get_solver("beamz")(
        comp, technology=tech, spec=SimulationSpec(modes=(1, 2)), gf_component=object()
    )
    problems = solver.validate()
    assert any("modes=(1,)" in p for p in problems)
    del layout


def test_beamz_resolves_index_from_rii(tmp_path, monkeypatch):
    """The rii material source feeds beamz with zero solver-specific entries."""
    pytest.importorskip("beamz")
    import copy
    import pathlib as _pl

    monkeypatch.setenv("GDS_FDTD_RII_DB", str(_pl.Path(__file__).parent / "rii_db"))
    comp, tech, layout = _job("tech_tidy3d.yaml")
    rii_tech = copy.deepcopy(tech)
    # v1 needs exactly ONE device layer present: keep [1,0] only, rii material
    rii_tech["device"] = [rii_tech["device"][0]]
    rii_tech["device"][0]["material"] = {"rii": {"shelf": "main", "book": "Si", "page": "Li-293"}}
    solver = get_solver("beamz")(comp, technology=rii_tech, gf_component=object())
    n_core, _ = solver._indices()
    assert n_core == pytest.approx(3.4757, abs=0.01)
    del layout


def test_beamz_field_plane_orientation():
    """beamz grids are (z, y, x)-ordered: a synthetic along-x stripe injected
    through the run() reshape convention must render along x (caught a real
    transposed field plot during live validation)."""
    pytest.importorskip("beamz")
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from gds_fdtd.solvers.beamz import BeamzSolver

    ny, nx = 116, 197
    flat = np.zeros((1, ny * nx), dtype=complex)
    flat.reshape(1, ny, nx)[0, ny // 2 - 5 : ny // 2 + 5, :] = 1.0

    s = object.__new__(BeamzSolver)
    ncells = ny * nx
    s._field_z = {
        c: flat.reshape(-1)[:ncells].reshape(ncells // nx, nx) for c in ("Ex", "Ey", "Ez")
    }
    s._field_z_meta = {"width_um": 11.0, "height_um": 6.5, "source": "o1"}
    _, ax = s.plot_fields(axis="z")
    img = np.asarray(ax.images[0].get_array())
    assert img.shape == (ny, nx)
    assert (img.sum(axis=1) > 0).sum() == 10  # stripe occupies 10 y-rows
    assert (img.sum(axis=0) > 0).all()  # and spans every x column


def test_beamz_rejects_y_oriented_ports():
    """F14: beamz modal normalization on y-normal monitors is wrong by tens
    of dB (measured S11 +40 dB on a vertical straight) - validate() must
    fail loudly instead of run() returning garbage."""
    pytest.importorskip("beamz")
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.layout.gdsfactory import from_gdsfactory

    c = gf.Component(name="vstraight_f14")
    ref = c.add_ref(gf.components.straight(length=5))
    ref.rotate(90)
    c.add_ports(ref.ports)
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_unified.yaml"))
    comp = from_gdsfactory(c, tech)
    problems = get_solver("beamz")(comp, technology=tech).validate()
    assert any("F14" in p for p in problems), problems
