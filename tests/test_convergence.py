"""WP5.5: convergence sweeps, simulation caching, cross-solver validation.

All offline: CannedSolver returns a mesh-dependent analytic S-matrix that
converges as 1/mesh^2, so sweep behaviour, cache short-circuiting and the
comparison metric are all exactly predictable.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from gds_fdtd.caching import cached_run, job_hash
from gds_fdtd.convergence import ConvergenceReport, max_delta_db, sweep
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.smatrix import SMatrix
from gds_fdtd.solvers import SetupArtifacts, Solver, SolverCapabilities
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology
from gds_fdtd.validation import compare_smatrices, validate_across

TESTS_DIR = pathlib.Path(__file__).parent


class CannedSolver(Solver):
    """Thru-line whose |S21| = 1 - 1/mesh^2: converges monotonically."""

    name = "canned"
    capabilities = SolverCapabilities(
        tier="full",
        execution="local",
        supports_dispersion=True,
        supports_sidewall_angle=True,
        supports_multimode=True,
        supports_gpu=False,
        cost_model="free",
    )
    run_calls = 0

    def validate(self):
        return []

    def build(self):
        return SetupArtifacts()

    def estimate(self):
        from gds_fdtd.solvers import ResourceEstimate

        return ResourceEstimate()

    def run(self) -> SMatrix:
        type(self).run_calls += 1
        f = self.frequencies_hz()
        amp = 1.0 - 1.0 / self.spec.mesh**2
        names = [p.name for p in self.component.ports][:2]
        thru = np.full(f.size, amp + 0j)
        entries = [
            (names[0], names[1], 1, 1, f, thru),
            (names[1], names[0], 1, 1, f, thru),
        ]
        return SMatrix.from_entries(entries, name=self.component.name, port_names=names)


class CannedSolverB(CannedSolver):
    name = "canned_b"
    run_calls = 0


@pytest.fixture(scope="module")
def component():
    tech = Technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp
    del layout


@pytest.fixture(autouse=True)
def _reset_run_counters():
    CannedSolver.run_calls = 0
    CannedSolverB.run_calls = 0


# ---------------------------------------------------------------------------
# convergence sweep
# ---------------------------------------------------------------------------


def test_sweep_deltas_and_recommendation(component):
    report = sweep(CannedSolver, component, field="mesh", values=[2, 4, 8, 16])
    assert isinstance(report, ConvergenceReport)
    assert len(report.smatrices) == 4 and len(report.deltas_db) == 3
    # 1/mesh^2 convergence: deltas strictly shrink
    assert report.deltas_db[0] > report.deltas_db[1] > report.deltas_db[2]
    # analytic check of the first delta: 20*log10((1-1/16)/(1-1/4))
    expected = abs(20 * np.log10((1 - 1 / 16) / (1 - 1 / 4)))
    assert report.deltas_db[0] == pytest.approx(expected, rel=1e-9)
    # a tight tolerance recommends the last step; an absurd one recommends early
    assert report.recommend(tol_db=report.deltas_db[2] * 1.01) == 16
    assert report.recommend(tol_db=10.0) == 4
    assert report.recommend(tol_db=1e-12) is None
    assert "recommended" in report.summary() or "NOT converged" in report.summary()


def test_sweep_rejects_unknown_field(component):
    with pytest.raises(ValueError, match="no field"):
        sweep(CannedSolver, component, field="not_a_field", values=[1, 2])


def test_report_plot_smoke(component, tmp_path):
    report = sweep(CannedSolver, component, field="mesh", values=[2, 4, 8])
    out = tmp_path / "conv.png"
    fig, ax = report.plot(savefig=str(out))
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------


def test_cached_run_short_circuits(component, tmp_path):
    solver = CannedSolver(component, None, SimulationSpec(mesh=6))
    sm1 = cached_run(solver, tmp_path)
    sm2 = cached_run(CannedSolver(component, None, SimulationSpec(mesh=6)), tmp_path)
    assert CannedSolver.run_calls == 1  # second call was a cache hit
    np.testing.assert_allclose(sm1.s, sm2.s)
    np.testing.assert_allclose(sm1.f, sm2.f)
    assert sm1.port_names == sm2.port_names


def test_job_hash_sensitivity(component):
    base = job_hash(CannedSolver(component, None, SimulationSpec(mesh=6)))
    assert base == job_hash(CannedSolver(component, None, SimulationSpec(mesh=6)))
    assert base != job_hash(CannedSolver(component, None, SimulationSpec(mesh=7)))
    assert base != job_hash(CannedSolverB(component, None, SimulationSpec(mesh=6)))


def test_job_hash_is_process_stable(component):
    """No memory addresses may leak into the fingerprint (repr ' at 0x')."""
    import json

    from gds_fdtd.caching import job_fingerprint

    tech = Technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))
    fp = json.dumps(job_fingerprint(CannedSolver(component, tech, SimulationSpec())))
    assert " at 0x" not in fp


def test_sweep_with_cache_reruns_free(component, tmp_path):
    values = [2, 4, 8]
    sweep(CannedSolver, component, field="mesh", values=values, cache_dir=tmp_path)
    assert CannedSolver.run_calls == len(values)
    report = sweep(CannedSolver, component, field="mesh", values=values, cache_dir=tmp_path)
    assert CannedSolver.run_calls == len(values)  # all hits, zero new runs
    assert report.recommend(tol_db=10.0) == 4


# ---------------------------------------------------------------------------
# cross-solver validation
# ---------------------------------------------------------------------------


def test_validate_across_agreement(component, tmp_path):
    report = validate_across(
        [CannedSolver, CannedSolverB],
        component,
        spec=SimulationSpec(mesh=8),
        cache_dir=tmp_path,
    )
    assert set(report.smatrices) == {"canned", "canned_b"}
    assert report.worst_db == pytest.approx(0.0, abs=1e-12)  # identical physics
    assert report.passed(tol_db=0.01)
    assert "worst pair" in report.summary()


def test_compare_smatrices_known_offset(component):
    sm_a = CannedSolver(component, None, SimulationSpec(mesh=4)).run()
    sm_b = CannedSolver(component, None, SimulationSpec(mesh=8)).run()
    report = compare_smatrices({"coarse": sm_a, "fine": sm_b})
    expected = abs(20 * np.log10((1 - 1 / 16) / (1 - 1 / 64)))
    assert report.worst_db == pytest.approx(expected, rel=1e-9)
    with pytest.raises(ValueError, match="at least two"):
        compare_smatrices({"only": sm_a})


def test_cross_report_plot_smoke(component, tmp_path):
    report = validate_across([CannedSolver, CannedSolverB], component, spec=SimulationSpec(mesh=8))
    out = tmp_path / "agreement.png"
    fig, _ = report.plot(out=2, in_=1, savefig=str(out))
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_max_delta_db_aligns_ports_by_digit():
    """Recorded artifacts name ports '1'/'2' vs 'opt1'/'opt2' — must align."""
    f = np.linspace(1.8e14, 2.0e14, 5)
    thru = np.full(5, 0.9 + 0j)
    a = SMatrix.from_entries(
        [("1", "2", 1, 1, f, thru), ("2", "1", 1, 1, f, thru)], port_names=["1", "2"]
    )
    b = SMatrix.from_entries(
        [("opt1", "opt2", 1, 1, f, thru), ("opt2", "opt1", 1, 1, f, thru)],
        port_names=["opt1", "opt2"],
    )
    assert max_delta_db(a, b) == pytest.approx(0.0, abs=1e-12)


def test_max_delta_db_dead_column_is_finite():
    """A path that is signal in one engine but zero in the other (a failed mode
    injection) must give a BOUNDED disagreement, never +inf."""
    f = np.linspace(1.8e14, 2.0e14, 5)
    thru = np.full(5, 0.9 + 0j)
    dead = np.zeros(5, dtype=complex)  # engine b under-injected this path
    a = SMatrix.from_entries(
        [("1", "2", 1, 1, f, thru), ("2", "1", 1, 1, f, thru)], port_names=["1", "2"]
    )
    b = SMatrix.from_entries(
        [("1", "2", 1, 1, f, thru), ("2", "1", 1, 1, f, dead)], port_names=["1", "2"]
    )
    d = max_delta_db(a, b, floor_db=-30.0)
    assert np.isfinite(d)
    # |20log10(0.9) - (-30)| ~= 29.1 dB, bounded by |signal - floor|
    assert 25.0 < d < 31.0


def test_recorded_cross_solver_via_api():
    """The tidy3d-vs-Lumerical escalator agreement, through the WP5.5 API."""
    pytest.importorskip("h5py")
    rec = TESTS_DIR / "recorded"
    report = compare_smatrices(
        {
            "tidy3d": SMatrix.from_hdf5(str(rec / "si_sin_escalator_smatrix.h5")),
            "lumerical": SMatrix.from_hdf5(str(rec / "si_sin_escalator_lum_smatrix.h5")),
        }
    )
    assert report.passed(tol_db=1.0), report.summary()
