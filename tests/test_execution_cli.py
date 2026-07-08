"""WP7.3: JobSpec round-trip law, execution backends, CLI exit codes.

SubprocessBackend tests prove the serialization boundary for real: the job
crosses a process boundary as JSON and a plugin solver registers inside the
child via --import.
"""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pytest

from gds_fdtd.cli import EXIT_BUDGET, EXIT_INVALID, EXIT_OK, EXIT_UNAVAILABLE, main
from gds_fdtd.execution import Budget, JobSpec, LocalBackend, SubprocessBackend, run_job
from gds_fdtd.smatrix import SMatrix

TESTS_DIR = pathlib.Path(__file__).parent

_FAKE_IMPL = """
import numpy as np

from gds_fdtd.smatrix import SMatrix
from gds_fdtd.solvers import (
    ResourceEstimate,
    SetupArtifacts,
    Solver,
    SolverCapabilities,
    register_solver,
)


@register_solver
class CliFakeSolver(Solver):
    name = "clifake"
    capabilities = SolverCapabilities(
        tier="full",
        execution="local",
        supports_dispersion=True,
        supports_sidewall_angle=True,
        supports_multimode=True,
        supports_gpu=False,
        cost_model="free",
    )

    def validate(self):
        return [] if self.component.ports else ["component has no ports"]

    def build(self):
        center, span = self.domain()
        return SetupArtifacts(
            native={"ports": self.injection_plan(), "center": center, "span": span}
        )

    def estimate(self):
        return ResourceEstimate(n_simulations=len(self.component.ports))

    def run(self) -> SMatrix:
        f = self.frequencies_hz()
        names = [p.name for p in self.component.ports][:2]
        thru = np.full(f.size, 0.9 + 0j)
        return SMatrix.from_entries(
            [(names[0], names[1], 1, 1, f, thru), (names[1], names[0], 1, 1, f, thru)],
            name=self.component.name,
            port_names=names,
        )
"""

exec(_FAKE_IMPL)  # registers "clifake" in THIS process for local tests


def _job(**overrides) -> JobSpec:
    kw = dict(
        gds_path=str(TESTS_DIR / "si_sin_escalator.gds"),
        technology_path=str(TESTS_DIR / "tech_lumerical.yaml"),
        solver="clifake",
    )
    kw.update(overrides)
    return JobSpec(**kw)


# ---------------------------------------------------------------------------
# round-trip law
# ---------------------------------------------------------------------------


def test_jobspec_roundtrip_builds_identical_artifacts():
    job = _job()
    clone = JobSpec.model_validate_json(job.model_dump_json())
    assert clone == job
    a = job.make_solver().build().native
    b = clone.make_solver().build().native
    assert a == b  # identical ports/domain from the reconstructed job


def test_jobspec_file_roundtrip(tmp_path):
    job = _job(budget=Budget(max_flexcredits=0))
    path = job.to_file(tmp_path / "job.json")
    assert JobSpec.from_file(path) == job


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


def test_local_backend_end_to_end(tmp_path):
    backend = LocalBackend()
    handle = backend.submit(_job(), tmp_path / "out")
    assert backend.status(handle) == "done"
    result = backend.result(handle)
    assert result.solver == "clifake" and len(result.job_hash) == 64
    sm = SMatrix.from_npz(str(handle.out_dir / result.smatrix_path))
    assert sm.magnitude_db(out=2, in_=1).max() == pytest.approx(20 * np.log10(0.9), abs=1e-9)


def test_run_job_budget_refusal(tmp_path):
    """A zero budget must refuse any non-free engine before spending."""
    job = _job(solver="lumerical", budget=Budget(max_flexcredits=0))
    with pytest.raises(PermissionError, match="budget forbids"):
        run_job(job, tmp_path / "out")


def test_subprocess_backend_end_to_end(tmp_path):
    """The flagship boundary test: JSON job -> child process -> SMatrix."""
    (tmp_path / "fake_plugin.py").write_text(textwrap.dedent(_FAKE_IMPL))
    backend = SubprocessBackend(extra_imports=("fake_plugin",))
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)  # child resolves --import fake_plugin from cwd
    try:
        handle = backend.submit(_job(), tmp_path / "out")
        result = backend.result(handle)
    finally:
        os.chdir(cwd)
    assert result.solver == "clifake"
    local = LocalBackend().result(LocalBackend().submit(_job(), tmp_path / "out_local"))
    assert result.job_hash == local.job_hash  # same job, either backend
    sm_sub = SMatrix.from_npz(str(tmp_path / "out" / result.smatrix_path))
    sm_loc = SMatrix.from_npz(str(tmp_path / "out_local" / local.smatrix_path))
    np.testing.assert_allclose(sm_sub.s, sm_loc.s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_validate_ok(tmp_path, capsys):
    path = _job().to_file(tmp_path / "job.json")
    assert main(["validate", str(path)]) == EXIT_OK
    assert "valid: True" in capsys.readouterr().out


def test_cli_unknown_solver_exit_3(tmp_path):
    path = _job(solver="no_such_engine").to_file(tmp_path / "job.json")
    assert main(["validate", str(path)]) == EXIT_UNAVAILABLE


def test_cli_run_budget_exit_4(tmp_path):
    job = _job(solver="lumerical", budget=Budget(max_flexcredits=0))
    path = job.to_file(tmp_path / "job.json")
    assert main(["run", str(path), "--out", str(tmp_path / "out")]) == EXIT_BUDGET


def test_cli_build_and_estimate(tmp_path, capsys):
    path = _job().to_file(tmp_path / "job.json")
    assert main(["--json", "estimate", str(path)]) == EXIT_OK
    assert '"n_simulations": 2' in capsys.readouterr().out
    assert main(["build", str(path), "--out", str(tmp_path / "b")]) == EXIT_OK


def test_cli_convert_dat_to_touchstone_and_npz(tmp_path, capsys):
    import shutil

    src = tmp_path / "esc.dat"
    shutil.copy(TESTS_DIR / "recorded" / "si_sin_escalator.dat", src)
    assert main(["convert", str(src), "--to", "npz"]) == EXIT_OK
    assert (tmp_path / "esc.npz").exists()
    assert main(["convert", str(src), "--to", "snp"]) == EXIT_OK
    assert (tmp_path / "esc.s2p").exists()
    assert main(["convert", str(tmp_path / "esc.npz"), "--to", "dat"]) == EXIT_OK
    back = SMatrix.from_dat(str(tmp_path / "esc.dat"))
    assert back.n_ports == 2


def test_cli_convert_rejects_unknown_format(tmp_path):
    (tmp_path / "x.csv").write_text("nope")
    assert main(["convert", str(tmp_path / "x.csv"), "--to", "npz"]) == EXIT_INVALID


def test_registered_backend_satisfies_protocol():
    from gds_fdtd.execution import ExecutionBackend

    assert isinstance(LocalBackend(), ExecutionBackend)
    assert isinstance(SubprocessBackend(), ExecutionBackend)
