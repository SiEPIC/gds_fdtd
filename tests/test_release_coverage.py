"""Release-gate coverage: modern Lumerical run (mocked), backend edges,
caching/logging corners — every test asserts real behavior, no padding."""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pytest

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

from .mocks.lumapi import install

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture()
def escalator():
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_unified.yaml"))
    cell, layout = load_cell(
        str(TESTS_DIR.parent / "examples" / "devices.gds"),
        top_cell="si_sin_escalator_te1550",
    )
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp, tech
    del layout


def test_modern_lumerical_run_via_mock(monkeypatch, escalator, tmp_path):
    """The MODERN adapter's run(): session -> script replay -> sweep ->
    recorded .dat -> canonical SMatrix, all offline via the lumapi mock."""
    install(monkeypatch)
    comp, tech = escalator
    solver = get_solver("lumerical")(
        comp,
        tech,
        SimulationSpec(wavelength_points=5, mesh=4, z_min=-1, z_max=1.11),
        workdir=tmp_path,
    )
    sm = solver.run()
    assert sm.n_ports == 2
    assert float(sm.magnitude_db(out=2, in_=1).max()) > -1.0  # recorded escalator thru


_SLEEPER_PLUGIN = textwrap.dedent(
    """
    import time
    from gds_fdtd.solvers import SetupArtifacts, Solver, SolverCapabilities, register_solver
    from gds_fdtd.solvers.base import ResourceEstimate

    @register_solver
    class SleepSolver(Solver):
        name = "sleeper"
        capabilities = SolverCapabilities(
            tier="full", execution="local", supports_dispersion=False,
            supports_sidewall_angle=False, supports_multimode=False,
            supports_gpu=False, cost_model="free")
        def validate(self): return []
        def build(self): return SetupArtifacts()
        def estimate(self): return ResourceEstimate()
        def run(self):
            time.sleep(60)
    """
)


def _sleeper_job(tmp_path, **budget_kwargs):
    from gds_fdtd.execution import Budget, JobSpec

    (tmp_path / "sleep_plugin.py").write_text(_SLEEPER_PLUGIN)
    return JobSpec(
        gds_path=str(TESTS_DIR / "si_sin_escalator.gds"),
        technology_path=str(TESTS_DIR / "tech_lumerical.yaml"),
        solver="sleeper",
        budget=Budget(**budget_kwargs) if budget_kwargs else None,
    )


def test_subprocess_backend_timeout_kills_child(tmp_path, monkeypatch):
    """budget.max_wall_seconds must kill a hung child and raise TimeoutError."""
    from gds_fdtd.execution import SubprocessBackend

    job = _sleeper_job(tmp_path, max_wall_seconds=2)
    monkeypatch.chdir(tmp_path)
    backend = SubprocessBackend(extra_imports=("sleep_plugin",))
    handle = backend.submit(job, tmp_path / "out")
    with pytest.raises(TimeoutError, match="max_wall_seconds"):
        backend.result(handle)


def test_subprocess_backend_cancel_and_status(tmp_path, monkeypatch):
    from gds_fdtd.execution import SubprocessBackend

    job = _sleeper_job(tmp_path)
    monkeypatch.chdir(tmp_path)
    backend = SubprocessBackend(extra_imports=("sleep_plugin",))
    handle = backend.submit(job, tmp_path / "out")
    assert backend.status(handle) == "running"
    backend.cancel(handle)
    assert backend.status(handle) == "cancelled"


def test_cached_run_accepts_str_dir(tmp_path, escalator):
    from tests.test_convergence import CannedSolver

    comp, _ = escalator
    solver = CannedSolver(comp, None, SimulationSpec(mesh=5))
    sm = solver.run_cached(str(tmp_path))  # str, not Path
    assert sm.n_ports == 2
    again = CannedSolver(comp, None, SimulationSpec(mesh=5)).run_cached(str(tmp_path))
    np.testing.assert_allclose(sm.s, again.s)


def test_json_log_includes_exception(monkeypatch, tmp_path, capsys):
    import json
    import logging

    monkeypatch.setenv("GDS_FDTD_LOG_FORMAT", "json")
    from gds_fdtd.settings import reset_settings

    reset_settings()
    try:
        from gds_fdtd.logging_config import get_logger, setup_logging

        logger = setup_logging(working_dir=str(tmp_path))
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("it failed")
        rec = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "boom" in rec["exc"]
        assert get_logger("gds_fdtd.sub").name == "gds_fdtd.sub"
    finally:
        reset_settings()
        lg = logging.getLogger("gds_fdtd")
        for h in lg.handlers[:]:
            lg.removeHandler(h)
            h.close()
