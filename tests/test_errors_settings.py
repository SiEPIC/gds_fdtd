"""WP7.6: error hierarchy contracts + settings + JSON logging."""

from __future__ import annotations

import json
import logging

import pytest

from gds_fdtd.errors import (
    BudgetExceededError,
    GdsFdtdError,
    JobValidationError,
    LayoutError,
    SolverError,
    SolverUnavailableError,
    TechnologyError,
)
from gds_fdtd.settings import GdsFdtdSettings, reset_settings, settings


def test_hierarchy_is_backward_compatible():
    """Every subclass must still satisfy the builtin type it replaced."""
    assert issubclass(JobValidationError, ValueError)
    assert issubclass(TechnologyError, ValueError)
    assert issubclass(LayoutError, ValueError)
    assert issubclass(SolverError, RuntimeError)
    assert issubclass(SolverUnavailableError, RuntimeError)
    assert issubclass(BudgetExceededError, PermissionError)
    for cls in (
        JobValidationError,
        TechnologyError,
        LayoutError,
        SolverError,
        SolverUnavailableError,
        BudgetExceededError,
    ):
        assert issubclass(cls, GdsFdtdError)


def test_solver_raises_hierarchy_types(tmp_path):
    """cannot-build and budget paths raise the new types (old excepts still work)."""
    import pathlib

    from gds_fdtd.execution import Budget, JobSpec, run_job

    tests_dir = pathlib.Path(__file__).parent
    job = JobSpec(
        gds_path=str(tests_dir / "si_sin_escalator.gds"),
        technology_path=str(tests_dir / "tech_lumerical.yaml"),
        solver="lumerical",
        budget=Budget(max_flexcredits=0),
    )
    with pytest.raises(BudgetExceededError):
        run_job(job, tmp_path)
    with pytest.raises(PermissionError):  # backward-compatible catch
        run_job(job, tmp_path)


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GDS_FDTD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GDS_FDTD_LOG_FORMAT", "json")
    reset_settings()
    try:
        s = settings()
        assert s.cache_dir == tmp_path / "cache"
        assert s.log_format == "json"
        assert s.telemetry is False  # policy default: OFF
    finally:
        reset_settings()


def test_settings_defaults():
    s = GdsFdtdSettings(_env_file=None)
    assert s.log_format == "text" and s.log_level == "INFO"
    assert s.default_budget_fc is None


def test_json_log_format(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GDS_FDTD_LOG_FORMAT", "json")
    reset_settings()
    try:
        from gds_fdtd.logging_config import setup_logging

        logger = setup_logging(working_dir=str(tmp_path))
        logger.info("hello structured world")
        out = capsys.readouterr().out.strip().splitlines()
        rec = json.loads(out[-1])
        assert rec["message"] == "hello structured world"
        assert rec["level"] == "INFO" and rec["logger"] == "gds_fdtd"
    finally:
        reset_settings()
        for h in logging.getLogger("gds_fdtd").handlers[:]:
            logging.getLogger("gds_fdtd").removeHandler(h)
            h.close()


def test_run_cached_uses_settings_default(monkeypatch, tmp_path):
    monkeypatch.setenv("GDS_FDTD_CACHE_DIR", str(tmp_path / "auto_cache"))
    reset_settings()
    try:
        from gds_fdtd.core import parse_yaml_tech
        from gds_fdtd.lyprocessor import load_cell
        from gds_fdtd.simprocessor import load_component_from_tech
        from tests.test_convergence import TESTS_DIR, CannedSolver  # noqa: F401

        tech = parse_yaml_tech(str(TESTS_DIR / "tech_lumerical.yaml"))
        cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
        comp = load_component_from_tech(cell=cell, tech=tech)
        solver = CannedSolver(comp, None)
        solver.run_cached()  # no cache_dir argument
        assert any((tmp_path / "auto_cache").glob("canned_*.npz"))
    finally:
        reset_settings()
