"""
gds_fdtd simulation toolbox.

Serializable simulation jobs (WP7.3). A JobSpec is everything needed to run
one simulation, as JSON: layout SOURCE references (GDS path + top cell +
technology YAML), the SimulationSpec, and the solver registry name.

Design note (deviation D10): the job references the layout source instead of
embedding the loaded Component — loaded components carry engine-resolved
material objects (a deliberate Phase-2 choice) that don't belong in a
serialized payload. Reconstructing through the same loading path makes the
round-trip law exact: a JobSpec deserialized from its own JSON builds
identical artifacts.

Credentials policy: secrets (TIDY3D_API_KEY, license servers, ...) are NEVER
part of a JobSpec — solvers read them from the environment at run() time, so
a job file can be shipped to any machine safely.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ..spec import SimulationSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..solvers.base import Solver

logger = logging.getLogger(__name__)


class Budget(BaseModel):
    """Spending limits for one job.

    ``max_wall_seconds`` is enforced by SubprocessBackend (hard timeout).
    ``max_flexcredits`` is advisory until estimate() reports numeric cost:
    run_job refuses non-free solvers when it is set to 0.
    """

    model_config = ConfigDict(extra="forbid")

    max_flexcredits: float | None = Field(None, ge=0)
    max_wall_seconds: float | None = Field(None, gt=0)


class JobSpec(BaseModel):
    """One simulation as a portable, serializable description."""

    model_config = ConfigDict(extra="forbid")

    gds_path: str
    top_cell: str | None = None
    technology_path: str
    spec: SimulationSpec = SimulationSpec()
    solver: str
    solver_options: dict[str, Any] = Field(default_factory=dict)
    budget: Budget | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> JobSpec:
        return cls.model_validate_json(Path(path).read_text())

    def to_file(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.model_dump_json(indent=2))
        return p

    # ---------------- reconstruction ----------------

    def load_component(self) -> tuple[Any, Any]:
        """Load (component, technology) through the standard loading path."""
        from ..core import parse_yaml_tech
        from ..lyprocessor import load_cell
        from ..simprocessor import load_component_from_tech

        tech = parse_yaml_tech(str(self.technology_path))
        cell, layout = load_cell(str(self.gds_path), top_cell=self.top_cell)
        component = load_component_from_tech(cell=cell, tech=tech)  # type: ignore[no-untyped-call]
        component._layout_keepalive = layout  # klayout object must outlive the cell
        return component, tech

    def make_solver(self, workdir: str | Path | None = None) -> Solver:
        from ..solvers import get_solver

        component, tech = self.load_component()
        cls = get_solver(self.solver)
        return cls(component, tech, self.spec, workdir=workdir, **self.solver_options)


class JobResult(BaseModel):
    """What a completed job hands back (paths are relative to the out dir)."""

    smatrix_path: str
    job_hash: str
    solver: str
    solver_version: str
    wall_seconds: float
    log_path: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> JobResult:
        return cls.model_validate_json(Path(path).read_text())


RESULT_FILENAME = "result.json"


def run_job(job: JobSpec, out_dir: str | Path) -> JobResult:
    """Execute a JobSpec: validate, budget-check, run, persist SMatrix + result.

    Raises ValueError on validation failure and PermissionError on budget
    refusal (the CLI maps these to exit codes 2 and 4).
    """
    from ..caching import job_hash

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    solver = job.make_solver(workdir=out)

    problems = solver.validate()
    if problems:
        raise ValueError(f"job invalid: {problems}")

    if (
        job.budget is not None
        and job.budget.max_flexcredits == 0
        and solver.capabilities.cost_model != "free"
    ):
        raise PermissionError(
            f"budget forbids spending (max_flexcredits=0) but solver "
            f"'{solver.name}' has cost model '{solver.capabilities.cost_model}'"
        )

    t0 = time.monotonic()
    sm = solver.run()
    wall = time.monotonic() - t0

    sm_path = out / "smatrix.npz"
    sm.to_npz(str(sm_path))
    result = JobResult(
        smatrix_path=sm_path.name,
        job_hash=job_hash(solver),
        solver=solver.name,
        solver_version=solver.engine_version(),
        wall_seconds=wall,
    )
    (out / RESULT_FILENAME).write_text(result.model_dump_json(indent=2))
    logger.info("job done in %.1f s -> %s", wall, sm_path)
    return result


def _json_default(obj: object) -> str:  # pragma: no cover - debugging helper
    return repr(obj)


def describe_job(job: JobSpec) -> str:
    """Human summary without loading anything heavy."""
    return json.dumps(job.model_dump(), indent=2, default=_json_default)
