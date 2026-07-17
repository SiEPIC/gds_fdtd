"""
gds_fdtd simulation toolbox.

Execution backends: where a JobSpec runs. ``LocalBackend`` executes
in-process; ``SubprocessBackend`` spawns ``gds-fdtd run job.json --out dir``
— proving the serialization boundary and giving crash isolation and
parallelism (a sweep can submit N jobs and collect as they finish).

Handles are in-memory only (no job database), which is out of scope here.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

from ..errors import SolverError
from .jobspec import RESULT_FILENAME, JobResult, JobSpec, run_job

logger = logging.getLogger(__name__)

JobStatus = Literal["running", "done", "failed", "cancelled"]


@dataclass
class JobHandle:
    """Opaque ticket for a submitted job."""

    id: str
    out_dir: Path
    _state: dict[str, Any] = field(default_factory=dict, repr=False)


@runtime_checkable
class ExecutionBackend(Protocol):
    def submit(self, job: JobSpec, out_dir: str | Path) -> JobHandle: ...

    def status(self, handle: JobHandle) -> JobStatus: ...

    def result(self, handle: JobHandle) -> JobResult: ...

    def cancel(self, handle: JobHandle) -> None: ...


class LocalBackend:
    """Runs the job synchronously in this process; submit() blocks."""

    def submit(self, job: JobSpec, out_dir: str | Path) -> JobHandle:
        handle = JobHandle(id=str(uuid.uuid4()), out_dir=Path(out_dir))
        try:
            handle._state["result"] = run_job(job, out_dir)
            handle._state["status"] = "done"
        except Exception as e:
            handle._state["status"] = "failed"
            handle._state["error"] = e
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        return cast(JobStatus, handle._state["status"])

    def result(self, handle: JobHandle) -> JobResult:
        if handle._state["status"] == "failed":
            raise SolverError(f"job {handle.id} failed") from handle._state["error"]
        return cast(JobResult, handle._state["result"])

    def cancel(self, handle: JobHandle) -> None:
        pass  # synchronous: nothing in flight after submit returns


class SubprocessBackend:
    """Runs each job as ``python -m gds_fdtd.cli run ...`` (crash-isolated).

    ``extra_imports`` are passed as ``--import`` so out-of-package solver
    plugins register inside the child process. Budget.max_wall_seconds is
    enforced here as a hard timeout on wait().
    """

    def __init__(self, extra_imports: tuple[str, ...] = ()):
        self.extra_imports = tuple(extra_imports)

    def submit(self, job: JobSpec, out_dir: str | Path) -> JobHandle:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        job_path = out / "job.json"
        job.to_file(job_path)
        cmd = [sys.executable, "-m", "gds_fdtd.cli"]
        for mod in self.extra_imports:
            cmd += ["--import", mod]
        cmd += ["run", str(job_path), "--out", str(out)]
        log = open(out / "job.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        logger.info("submitted job as pid %d -> %s", proc.pid, out)
        handle = JobHandle(id=str(proc.pid), out_dir=out)
        handle._state.update(
            proc=proc, log=log, timeout=(job.budget.max_wall_seconds if job.budget else None)
        )
        return handle

    def _finalize(self, handle: JobHandle) -> None:
        log = handle._state.pop("log", None)
        if log is not None:
            log.close()

    def status(self, handle: JobHandle) -> JobStatus:
        if handle._state.get("cancelled"):
            return "cancelled"
        proc: subprocess.Popen[bytes] = handle._state["proc"]
        rc = proc.poll()
        if rc is None:
            return "running"
        self._finalize(handle)
        return "done" if rc == 0 else "failed"

    def result(self, handle: JobHandle) -> JobResult:
        proc: subprocess.Popen[bytes] = handle._state["proc"]
        timeout = handle._state.get("timeout")
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self._finalize(handle)
            raise TimeoutError(
                f"job {handle.id} exceeded budget.max_wall_seconds={timeout}"
            ) from None
        self._finalize(handle)
        if rc != 0:
            log_tail = (handle.out_dir / "job.log").read_text()[-2000:]
            raise SolverError(f"job {handle.id} exited {rc}; log tail:\n{log_tail}")
        return JobResult.from_file(handle.out_dir / RESULT_FILENAME)

    def cancel(self, handle: JobHandle) -> None:
        proc: subprocess.Popen[bytes] = handle._state["proc"]
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        handle._state["cancelled"] = True
        self._finalize(handle)
