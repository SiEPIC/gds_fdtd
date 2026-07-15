"""Targeted offline tests for previously unexecuted paths (WS2).

Everything here is offline and free: the tidy3d S-matrix/field conversion runs
on synthetic xarray data, the Lumerical material emission is pure .lsf text
generation, rii parsing uses tiny in-tmpdir database pages, and the CLI tests
drive exit codes without any engine.
"""

from __future__ import annotations

import pathlib
import textwrap

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

TESTS_DIR = pathlib.Path(__file__).parent


# ------------------------------------------------- tidy3d: DataArray -> SMatrix


def _tidy3d_solver_stub(modes: tuple[int, ...] = (1,)):
    """A Tidy3DSolver with just the attributes the conversion helpers read."""
    pytest.importorskip("tidy3d")
    from gds_fdtd.solvers.tidy3d import Tidy3DSolver
    from gds_fdtd.spec import SimulationSpec

    solver = object.__new__(Tidy3DSolver)
    solver.spec = SimulationSpec(wavelength_points=3, modes=list(modes))

    class _C:
        name = "synthetic"

    solver.component = _C()
    return solver


def test_dataarray_to_smatrix_maps_ports_and_modes():
    xr = pytest.importorskip("xarray")
    freqs = np.linspace(1.9e14, 2.0e14, 3)
    ports = ["opt1", "opt2"]
    modes = [0, 1]
    rng = np.random.default_rng(7)
    values = rng.uniform(0.1, 1.0, (2, 2, 2, 2, 3)) * np.exp(
        1j * rng.uniform(-3, 3, (2, 2, 2, 2, 3))
    )
    da = xr.DataArray(
        values,
        dims=("port_in", "port_out", "mode_index_in", "mode_index_out", "f"),
        coords={
            "port_in": ports,
            "port_out": ports,
            "mode_index_in": modes,
            "mode_index_out": modes,
            "f": freqs,
        },
    )
    solver = _tidy3d_solver_stub(modes=(1, 2))
    sm = solver._dataarray_to_smatrix(da)
    assert sm.n_ports == 2 and sm.n_modes == 2
    # spot-check one path: 0-based tidy3d mode index -> 1-based package id
    expected = da.sel(port_in="opt1", port_out="opt2", mode_index_in=0, mode_index_out=1).values
    np.testing.assert_allclose(sm.sel(out="opt2", in_="opt1", mode_out=2, mode_in=1), expected)


def test_dataarray_to_smatrix_filters_unwanted_modes():
    xr = pytest.importorskip("xarray")
    freqs = np.linspace(1.9e14, 2.0e14, 2)
    da = xr.DataArray(
        np.ones((1, 1, 2, 2, 2), dtype=complex),
        dims=("port_in", "port_out", "mode_index_in", "mode_index_out", "f"),
        coords={
            "port_in": ["opt1"],
            "port_out": ["opt1"],
            "mode_index_in": [0, 1],
            "mode_index_out": [0, 1],
            "f": freqs,
        },
    )
    sm = _tidy3d_solver_stub(modes=(1,))._dataarray_to_smatrix(da)
    assert sm.n_modes == 1  # mode 2 columns were dropped


def test_plot_tidy3d_fields_synthetic():
    xr = pytest.importorskip("xarray")
    from gds_fdtd.solvers.tidy3d import plot_tidy3d_fields

    x = np.linspace(-2, 2, 24)
    y = np.linspace(-1, 1, 16)
    f = np.array([1.9e14, 1.95e14, 2.0e14])
    data = np.exp(-np.abs(y)[None, :, None] * 3) * np.ones((x.size, 1, f.size))

    class _FD:
        Ex = xr.DataArray(data, dims=("x", "y", "f"), coords={"x": x, "y": y, "f": f})
        Ey = Ex * 0.1
        Ez = Ex * 0.0

    class _SimData:
        def __getitem__(self, key):
            assert key == "z_field"
            return _FD()

    class _ModelerData:
        data = {"opt1@0": _SimData()}

    fig, ax = plot_tidy3d_fields(_ModelerData(), axis="z", scale="db")
    assert len(ax.collections) == 1  # pcolormesh on true grid
    assert ax.collections[0].get_clim() == (-40.0, 0.0)


def test_tidy3d_plot_fields_before_run_raises():
    pytest.importorskip("tidy3d")
    from gds_fdtd.errors import SolverError

    solver = _tidy3d_solver_stub()
    with pytest.raises(SolverError, match="run\\(\\) has not completed"):
        solver.plot_fields()


# ----------------------------------------------- lumerical: offline .lsf paths


def _lum_build(tech_dict):
    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.simprocessor import load_component_from_tech
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec

    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech_dict)
    solver = get_solver("lumerical")(
        comp,
        technology=tech_dict,
        spec=SimulationSpec(wavelength_points=3, mesh=6, z_min=-1.0, z_max=1.11),
    )
    art = solver.build()
    del layout
    return art.native


def _escalator_tech_dict(material_override):
    import copy

    from gds_fdtd.technology import Technology

    tech = Technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))
    d = copy.deepcopy(tech.to_solver_dict())
    for layer in d["device"]:
        layer["material"] = dict(material_override)
    return d


def test_lum_script_emits_nk_material_for_constant():
    script = _lum_build(_escalator_tech_dict({"nk": 3.2}))
    assert 'addmaterial("(n,k) Material")' in script
    assert '"Refractive Index", 3.2' in script


def test_lum_script_emits_nk_material_for_rii(monkeypatch):
    monkeypatch.setenv("GDS_FDTD_RII_DB", str(TESTS_DIR / "rii_db"))
    script = _lum_build(
        _escalator_tech_dict({"rii": {"shelf": "main", "book": "Si", "page": "Li-293"}})
    )
    assert 'addmaterial("(n,k) Material")' in script  # rii -> constant at band center
    assert '"Refractive Index", 3.4' in script  # Si near 1.55 um


# ------------------------------------------------------------- rii: parsers


def _write_page(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    page = tmp_path / "main" / "X" / "test.yml"
    page.parent.mkdir(parents=True)
    page.write_text(textwrap.dedent(body))
    return tmp_path


def test_rii_formula_2_sellmeier():
    from gds_fdtd.materials.rii import load_rii_material

    db = TESTS_DIR / "rii_db"
    # tests ship a Sellmeier fixture? build one on the fly instead
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        db = _write_page(
            pathlib.Path(td),
            """
            DATA:
              - type: formula 2
                wavelength_range: 0.4 2.0
                coefficients: 0 1.0 0.01 0.2 100
            """,
        )
        m = load_rii_material("main", "X", "test", db_dir=db)
        n = float(np.asarray(m.n_at(1.55)))
        assert 1.3 < n < 1.6  # sqrt(1 + B1*w2/(w2-C1) + ...)


def test_rii_separate_tabulated_n_and_k_blocks():
    import tempfile

    from gds_fdtd.materials.rii import load_rii_material

    with tempfile.TemporaryDirectory() as td:
        db = _write_page(
            pathlib.Path(td),
            """
            DATA:
              - type: tabulated n
                data: |
                  1.0 2.0
                  2.0 2.2
              - type: tabulated k
                data: |
                  1.0 0.5
                  2.0 0.7
            """,
        )
        m = load_rii_material("main", "X", "test", db_dir=db)
        assert float(np.asarray(m.n_at(1.5))) == pytest.approx(2.1, abs=1e-6)
        assert float(np.asarray(m.k_at(1.5))) == pytest.approx(0.6, abs=1e-6)


def test_rii_unsupported_formula_and_missing_data():
    import tempfile

    from gds_fdtd.errors import TechnologyError
    from gds_fdtd.materials.rii import load_rii_material

    with tempfile.TemporaryDirectory() as td:
        db = _write_page(
            pathlib.Path(td),
            """
            DATA:
              - type: formula 5
                wavelength_range: 0.4 2.0
                coefficients: 1 2 3
            """,
        )
        with pytest.raises(TechnologyError, match="not supported"):
            load_rii_material("main", "X", "test", db_dir=db)

    with tempfile.TemporaryDirectory() as td:
        db = _write_page(pathlib.Path(td), "REFERENCES: none\n")
        with pytest.raises(TechnologyError, match="no DATA section"):
            load_rii_material("main", "X", "test", db_dir=db)


# ------------------------------------------------------------------ CLI paths


def test_cli_convert_npz_to_touchstone_and_back(tmp_path):
    from gds_fdtd.cli import main
    from gds_fdtd.smatrix import SMatrix

    f = np.linspace(1.8e14, 2.0e14, 3)
    s = np.full((3, 2, 2, 1, 1), 0.5 + 0.1j)
    src = tmp_path / "dev.npz"
    SMatrix(f=f, s=s, port_names=["opt1", "opt2"], name="dev").to_npz(str(src))

    assert main(["convert", str(src), "--to", "snp"]) == 0
    assert (tmp_path / "dev.s2p").exists()
    assert main(["convert", str(src), "--to", "dat"]) == 0
    back_dat = tmp_path / "dev.dat"
    assert back_dat.exists()
    assert main(["convert", str(back_dat), "--to", "npz"]) == 0

    # unsupported input extension -> invalid
    bad = tmp_path / "dev.xyz"
    bad.write_text("nope")
    assert main(["convert", str(bad), "--to", "npz"]) == 2


def test_cli_run_unavailable_engine(tmp_path):
    import json

    from gds_fdtd.cli import main

    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(  # json.dumps escapes Windows path backslashes correctly
            {
                "gds_path": str(TESTS_DIR / "si_sin_escalator.gds"),
                "top_cell": "si_sin_escalator",
                "technology_path": str(TESTS_DIR / "tech_lumerical.yaml"),
                "solver": "not_a_real_engine",
                "spec": {"wavelength_points": 3},
            }
        )
    )
    assert main(["run", str(job), "--out", str(tmp_path)]) == 3  # EXIT_UNAVAILABLE


def test_cli_top_level_error_path(tmp_path):
    from gds_fdtd.cli import main

    assert main(["validate", str(tmp_path / "missing.json")]) == 1  # generic error exit


# --------------------------------------- tidy3d engine: material-source media


def _t3d_build_with_material(material_override, monkeypatch=None, **spec_kw):
    pytest.importorskip("tidy3d")
    import copy

    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.simprocessor import load_component_from_tech
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec
    from gds_fdtd.technology import Technology

    tech = Technology.from_yaml(str(TESTS_DIR / "tech_tidy3d.yaml"))
    d = copy.deepcopy(tech.to_solver_dict())
    for layer in d["device"]:
        layer["material"] = dict(material_override)
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=d)
    solver = get_solver("tidy3d")(
        comp,
        technology=d,
        spec=SimulationSpec(wavelength_points=3, mesh=6, z_min=-1.0, z_max=1.11, **spec_kw),
    )
    art = solver.build()
    del layout
    return art


def test_t3d_engine_builds_nk_constant_medium(tmp_path):
    art = _t3d_build_with_material({"nk": 3.2})
    assert art.summary["n_ports"] == 2


def test_t3d_engine_builds_rii_dispersive_medium(monkeypatch):
    monkeypatch.setenv("GDS_FDTD_RII_DB", str(TESTS_DIR / "rii_db"))
    art = _t3d_build_with_material({"rii": {"shelf": "main", "book": "Si", "page": "Li-293"}})
    assert art.summary["n_ports"] == 2


def test_t3d_engine_estimate_offline():
    pytest.importorskip("tidy3d")
    art = _t3d_build_with_material({"nk": 3.0})
    assert art.summary["n_simulations"] >= 2


# ------------------------------------------------- technology validator errors


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"device": [{"layer": [1], "z_base": 0, "z_span": 0.22, "material": "Si"}]}, "layer"),
        ({"device": [{"layer": [1, 0], "z_base": 0, "z_span": 0, "material": "Si"}]}, "z_span"),
        ({"materials": {"Si": 3.4}}, "mapping"),
        ({"schema_version": 9}, "schema"),
    ],
)
def test_technology_validator_errors(mutation, match):
    from gds_fdtd.errors import TechnologyError
    from gds_fdtd.technology import Technology

    base = {
        "name": "bad",
        "schema_version": 2,
        "materials": {"Si": {"nk": 3.476}, "SiO2": {"nk": 1.444}},
        "substrate": {"z_base": 0.0, "z_span": -2, "material": "SiO2"},
        "superstrate": {"z_base": 0.0, "z_span": 3, "material": "SiO2"},
        "pinrec": [{"layer": [1, 10]}],
        "devrec": [{"layer": [68, 0]}],
        "device": [{"layer": [1, 0], "z_base": 0.0, "z_span": 0.22, "material": "Si"}],
    }
    base.update(mutation)
    import pydantic

    with pytest.raises((pydantic.ValidationError, TechnologyError), match=match):
        Technology.model_validate(base)


def test_technology_from_yaml_file_errors(tmp_path):
    from gds_fdtd.errors import TechnologyError
    from gds_fdtd.technology import Technology

    p = tmp_path / "no_top.yaml"
    p.write_text("nottechnology: {}\n")
    with pytest.raises(TechnologyError, match="top-level 'technology' mapping"):
        Technology.from_yaml(p)

    p2 = tmp_path / "invalid.yaml"
    p2.write_text("technology: {name: x, device: []}\n")
    with pytest.raises(TechnologyError, match="Invalid technology file"):
        Technology.from_yaml(p2)


# -------------------------------------------------------- smatrix error paths


def test_smatrix_constructor_and_export_errors(tmp_path):
    from gds_fdtd.errors import SMatrixError
    from gds_fdtd.smatrix import SMatrix

    f = np.linspace(1.8e14, 2.0e14, 3)
    good = SMatrix(
        f=f, s=np.zeros((3, 2, 2, 1, 1), dtype=complex), port_names=["opt1", "opt2"], name="x"
    )

    with pytest.raises(SMatrixError, match="duplicate port names"):
        SMatrix(f=f, s=np.zeros((3, 2, 2, 1, 1), dtype=complex), port_names=["a1", "a1"], name="x")
    with pytest.raises(SMatrixError, match="strictly ascending"):
        SMatrix(
            f=f[::-1].copy(),
            s=np.zeros((3, 2, 2, 1, 1), dtype=complex),
            port_names=["a1", "a2"],
            name="x",
        )
    with pytest.raises(SMatrixError, match="no entries"):
        SMatrix.from_entries([], name="x")

    # touchstone: NaN (partial) matrices are rejected with guidance
    partial = SMatrix(
        f=f,
        s=np.full((3, 2, 2, 1, 1), np.nan, dtype=complex),
        port_names=["opt1", "opt2"],
        name="x",
    )
    with pytest.raises(SMatrixError, match="NaN"):
        partial.to_touchstone(str(tmp_path / "p.s2p"))
    # wrong extension for the flattened port count
    with pytest.raises(SMatrixError, match="must end with"):
        good.to_touchstone(str(tmp_path / "p.s9p"))

    with pytest.raises(KeyError, match="no port with id"):
        good.sel(out=7, in_=1)
    with pytest.raises(KeyError, match="unknown port"):
        good.sel(out="nope", in_="opt1")


# --------------------------------------------------------- execution backends


def test_local_backend_failure_surfaces_as_solver_error(tmp_path):
    from gds_fdtd.errors import SolverError
    from gds_fdtd.execution import JobSpec
    from gds_fdtd.execution.backends import LocalBackend

    job = JobSpec(
        gds_path=str(TESTS_DIR / "si_sin_escalator.gds"),
        top_cell="si_sin_escalator",
        technology_path=str(TESTS_DIR / "tech_lumerical.yaml"),
        solver="not_a_real_engine",
        spec={"wavelength_points": 3},
    )
    backend = LocalBackend()
    handle = backend.submit(job, out_dir=str(tmp_path))
    assert backend.status(handle) == "failed"
    with pytest.raises(SolverError, match="failed"):
        backend.result(handle)


# ------------------------------------------------------------- logging helpers


def test_logging_helpers_emit(caplog):
    import logging

    from gds_fdtd.logging_config import (
        get_logger,
        log_dict,
        log_separator,
        log_simulation_complete,
        log_simulation_start,
    )

    lg = get_logger("unit")
    with caplog.at_level(logging.INFO, logger=lg.name):
        log_separator(lg, "TITLE")
        log_dict(lg, {"a": 1, "b": {"c": 2}}, title="Cfg")
        log_simulation_start(lg, "beamz", "dev")
        log_simulation_complete(lg, "beamz")
    text = caplog.text
    assert "TITLE" in text and "Cfg" in text and "BEAMZ" in text


# ------------------------------------------------------------ caching hashing


def test_job_fingerprint_handles_cycles_and_objects():
    from gds_fdtd.caching import _jsonable

    class Weird:
        pass

    w = Weird()
    w.self_ref = w  # cycle
    out = _jsonable(w)
    assert "Weird" in str(out)

    class NoDict:
        __slots__ = ()

    assert isinstance(_jsonable(NoDict()), str)


# --------------------------------------------------------------- plotting gaps


def test_plot_smatrix_phase_linear_and_dense_legend(tmp_path):
    from gds_fdtd.plotting import plot_smatrix
    from gds_fdtd.smatrix import SMatrix

    f = np.linspace(1.8e14, 2.0e14, 5)
    rng = np.random.default_rng(3)
    s = rng.uniform(0.3, 1.0, (5, 4, 4, 1, 1)) * np.exp(1j * rng.uniform(-3, 3, (5, 4, 4, 1, 1)))
    sm = SMatrix(f=f, s=s, port_names=[f"o{i}" for i in range(1, 5)], name="dense")

    fig, ax = plot_smatrix(sm, kind="phase")
    assert ax.get_ylabel().startswith("Phase")
    fig2, ax2 = plot_smatrix(sm, kind="linear")
    assert "normalized" in ax2.get_ylabel()
    # 16 live paths -> legend moves outside (dense-matrix layout)
    fig3, ax3 = plot_smatrix(sm, kind="db")
    assert len(ax3.get_lines()) == 16

    with pytest.raises(ValueError, match="kind must be one of"):
        plot_smatrix(sm, kind="nope")


# ------------------------------------------------------------------- round 3


def test_plot_helpers_savefig_and_options(tmp_path):
    from gds_fdtd.plotting import plot_field, plot_tech_stack
    from gds_fdtd.technology import Technology

    # plot_field: imshow path without colorbar + savefig
    m = np.random.default_rng(1).random((8, 10))
    fig, ax = plot_field(
        m, extent=(0, 1, 0, 1), colorbar=False, title="t", savefig=str(tmp_path / "f.png")
    )
    assert (tmp_path / "f.png").exists()

    # plot_tech_stack: savefig + a material no source can resolve (skipped band)
    tech = Technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))
    fig2, ax2 = plot_tech_stack(tech, savefig=str(tmp_path / "s.png"))
    assert (tmp_path / "s.png").exists()


def test_smatrix_shape_validation_errors():
    from gds_fdtd.errors import SMatrixError
    from gds_fdtd.smatrix import SMatrix

    f = np.linspace(1.8e14, 2.0e14, 3)
    with pytest.raises(SMatrixError, match="f must be 1-D"):
        SMatrix(
            f=f.reshape(3, 1),
            s=np.zeros((3, 1, 1, 1, 1), dtype=complex),
            port_names=["o1"],
            name="x",
        )
    with pytest.raises(SMatrixError, match="must have shape"):
        SMatrix(f=f, s=np.zeros((3, 2, 1, 1, 1), dtype=complex), port_names=["o1"], name="x")
    with pytest.raises(SMatrixError, match="mode axes must be square"):
        SMatrix(f=f, s=np.zeros((3, 1, 1, 2, 1), dtype=complex), port_names=["o1"], name="x")


def test_more_technology_validator_branches():
    import pydantic

    from gds_fdtd.technology import Technology

    base = {
        "name": "bad",
        "schema_version": 2,
        "materials": {"Si": {"nk": 3.476}, "SiO2": {"nk": 1.444}},
        "substrate": {"z_base": 0.0, "z_span": -2, "material": "SiO2"},
        "superstrate": {"z_base": 0.0, "z_span": 3, "material": "SiO2"},
        "pinrec": [{"layer": [1, 10]}],
        "devrec": [{"layer": [68, 0]}],
        "device": [{"layer": [1, 0], "z_base": 0.0, "z_span": 0.22, "material": "Si"}],
    }

    # lum_db without model
    b = {**base, "device": [dict(base["device"][0], material={"lum_db": {"nope": 1}})]}
    with pytest.raises(pydantic.ValidationError, match="lum_db"):
        Technology.model_validate({**b, "materials": {}, "schema_version": 1})

    # tidy3d_db without nk/model
    b = {**base, "device": [dict(base["device"][0], material={"tidy3d_db": {"x": 1}})]}
    with pytest.raises(pydantic.ValidationError, match="tidy3d_db"):
        Technology.model_validate({**b, "materials": {}, "schema_version": 1})

    # substrate list with more than one entry
    b = dict(base)
    b["substrate"] = [
        {"z_base": 0.0, "z_span": -2, "material": "SiO2"},
        {"z_base": -2.0, "z_span": -2, "material": "SiO2"},
    ]
    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        Technology.model_validate(b)

    # unknown material name referenced by a layer
    b = dict(base)
    b["device"] = [dict(base["device"][0], material="Unobtainium")]
    with pytest.raises(pydantic.ValidationError, match="Unobtainium"):
        Technology.model_validate(b)


def test_setup_logging_writes_file(tmp_path):
    from gds_fdtd.logging_config import setup_logging

    lg = setup_logging(working_dir=str(tmp_path), component_name="unit_test")
    lg.info("hello file")
    logs = list(tmp_path.glob("**/*.log"))
    assert logs and any("hello file" in p.read_text() for p in logs)
    # tear down handlers so later tests aren't affected
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def test_dilate_helpers_and_dimension_arg():
    from gds_fdtd.lyprocessor import dilate, dilate_1d

    box = [[0.0, 0.0], [2.0, 1.0]]
    grown = dilate(box, extension=0.5)
    xs = [v[0] for v in grown]
    ys = [v[1] for v in grown]
    assert min(xs) == -0.5 and max(xs) == 2.5 and min(ys) == -0.5 and max(ys) == 1.5

    seg_x = dilate_1d([[0.0, 0.0], [2.0, 0.0]], extension=1, dim="x")
    assert min(v[0] for v in seg_x) == -1.0 and max(v[0] for v in seg_x) == 3.0
    seg_y = dilate_1d([[0.0, 0.0], [0.0, 2.0]], extension=1, dim="y")
    assert min(v[1] for v in seg_y) == -1.0 and max(v[1] for v in seg_y) == 3.0
    with pytest.raises(ValueError, match="Dimension must be"):
        dilate_1d(box, dim="z")


def test_subprocess_backend_bad_job_fails(tmp_path):
    from gds_fdtd.errors import SolverError
    from gds_fdtd.execution import JobSpec
    from gds_fdtd.execution.backends import SubprocessBackend

    job = JobSpec(
        gds_path=str(TESTS_DIR / "si_sin_escalator.gds"),
        top_cell="si_sin_escalator",
        technology_path=str(TESTS_DIR / "tech_lumerical.yaml"),
        solver="not_a_real_engine",
        spec={"wavelength_points": 3},
    )
    backend = SubprocessBackend()
    handle = backend.submit(job, out_dir=str(tmp_path))
    with pytest.raises((SolverError, Exception)):
        backend.result(handle)


def test_t3d_engine_field_monitor_axes():
    pytest.importorskip("tidy3d")
    art_y = _t3d_build_with_material({"nk": 3.0}, field_monitors=("y",))
    assert art_y.summary["n_ports"] == 2
    art_xz = _t3d_build_with_material({"nk": 3.0}, field_monitors=("x", "z"))
    assert art_xz.summary["n_ports"] == 2


# ------------------------------------------------------- numeric-core errors


def test_max_delta_db_mismatch_errors():
    from gds_fdtd.convergence import max_delta_db
    from gds_fdtd.errors import SMatrixError
    from gds_fdtd.smatrix import SMatrix

    f = np.linspace(1.8e14, 2.0e14, 3)
    ones = np.full((3, 1, 1, 1, 1), 0.5, dtype=complex)
    a = SMatrix(f=f, s=ones, port_names=["opt1"], name="a")

    # unalignable names (no trailing digits on one side)
    weird = SMatrix(f=f, s=ones, port_names=["west"], name="w")
    with pytest.raises(SMatrixError, match="cannot align ports"):
        max_delta_db(a, weird)

    # mode-count mismatch
    two_modes = SMatrix(
        f=f, s=np.full((3, 1, 1, 2, 2), 0.5, dtype=complex), port_names=["opt1"], name="m"
    )
    with pytest.raises(SMatrixError, match="mode counts differ"):
        max_delta_db(a, two_modes)

    # disjoint frequency grids
    far = SMatrix(f=f + 1e15, s=ones, port_names=["opt1"], name="far")
    with pytest.raises(SMatrixError, match="frequency grids do not overlap"):
        max_delta_db(a, far)


def test_sweep_rejects_invalid_job():
    from gds_fdtd.convergence import sweep
    from gds_fdtd.errors import JobValidationError
    from gds_fdtd.geometry import Component, Region
    from gds_fdtd.solvers import get_solver

    empty = Component(
        name="empty",
        structures=[],
        ports=[],
        bounds=Region(vertices=[[0, 0], [1, 0], [1, 1], [0, 1]], z_center=0.11, z_span=2),
    )
    with pytest.raises(JobValidationError, match="job invalid"):
        sweep(get_solver("beamz"), empty, None, None, field="mesh", values=[4])


def test_grid_error_paths():
    from gds_fdtd.geometry import Component, Region
    from gds_fdtd.grid import rasterize, resolve_index

    with pytest.raises(ValueError, match="cannot be resolved offline"):
        resolve_index({"lum_db": {"model": "Si (Silicon) - Palik"}}, 1.55)

    empty = Component(
        name="empty",
        structures=[],
        ports=[],
        bounds=Region(vertices=[[0, 0], [1, 0], [1, 1], [0, 1]], z_center=0.11, z_span=2),
    )
    with pytest.raises(ValueError, match="no structures"):
        rasterize(empty, dx=0.1)


def test_modes_input_validation():
    from gds_fdtd.modes import waveguide_mode

    pytest.importorskip("tidy3d")
    with pytest.raises(ValueError, match="must be positive"):
        waveguide_mode(-0.5, 0.22, 3.47, 1.44, 1.55)


def test_extraction_zero_power_mode_rejected():
    from gds_fdtd.extraction import mode_amplitude
    from gds_fdtd.modes import FIELD_KEYS, Mode

    n = 8
    fields = {k: np.zeros((n, n), dtype=complex) for k in FIELD_KEYS}
    mode = Mode(
        n_eff=2.4 + 0j,
        fields=fields,
        u=np.arange(n) * 0.1,
        v=np.arange(n) * 0.1,
        wavelength_um=1.55,
    )
    with pytest.raises(ValueError, match="no forward power"):
        mode_amplitude(fields, mode, du=0.1, dv=0.1, normal="x")
