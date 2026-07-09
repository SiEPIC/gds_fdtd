"""
gds_fdtd simulation toolbox.

Simulation-result caching. A job is identified by a sha256
over a canonical JSON fingerprint of everything that determines its result:
component geometry (structures, ports, bounds), technology, SimulationSpec,
solver name and engine version. ``cached_run`` short-circuits repeat runs to
a stored ``SMatrix`` — the expensive ``run()`` executes at most once per
distinct job.

Results are stored as ``<solver>_<hash16>.npz`` via ``SMatrix.to_npz``
(npz rather than HDF5 so the cache has zero optional dependencies).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .geometry import Component
    from .smatrix import SMatrix
    from .solvers.base import Solver

logger = logging.getLogger(__name__)


def _jsonable(obj: Any, _seen: set[int] | None = None) -> Any:
    """Canonical, deterministic JSON-compatible view of an arbitrary object.

    Determinism matters more than fidelity here: the output feeds a hash, so
    it must be identical across processes (no ``repr`` memory addresses).
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):  # includes np.float64 (numpy 2 reprs differ)
        return float(obj)
    seen = _seen if _seen is not None else set()
    if id(obj) in seen:
        return f"<cycle:{type(obj).__qualname__}>"
    seen = seen | {id(obj)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x, seen) for x in obj]
    if isinstance(obj, dict):
        return {
            str(k): _jsonable(v, seen) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if hasattr(obj, "model_dump"):  # pydantic (Technology, SimulationSpec, MaterialSpec)
        return _jsonable(obj.model_dump(), seen)
    if hasattr(obj, "tolist"):  # numpy arrays/scalars
        return _jsonable(obj.tolist(), seen)
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict(), seen)
    r = repr(obj)
    if " at 0x" not in r:
        return r
    # default object repr embeds a memory address; hash the state instead
    state = getattr(obj, "__dict__", None)
    if state:
        return {
            "__class__": type(obj).__qualname__,
            **{str(k): _jsonable(v, seen) for k, v in sorted(state.items())},
        }
    return type(obj).__qualname__


def component_fingerprint(component: Component) -> dict:
    """Everything about the geometry that affects a simulation result."""
    return {
        "name": component.name,
        "structures": [
            {
                "polygon": _jsonable(s.polygon),
                "z_base": _jsonable(s.z_base),
                "z_span": _jsonable(s.z_span),
                "material": _jsonable(s.material),
                "sidewall_angle": _jsonable(s.sidewall_angle),
                "layer": _jsonable(s.layer),
                "role": _jsonable(getattr(s, "role", None)),
            }
            for s in component.structures
        ],
        "ports": [
            {
                "name": p.name,
                "center": _jsonable(p.center),
                "width": _jsonable(p.width),
                "direction": _jsonable(p.direction),
                "height": _jsonable(p.height),
                "material": _jsonable(p.material),
            }
            for p in sorted(component.ports, key=lambda p: p.name)
        ],
    }


def job_fingerprint(solver: Solver) -> dict:
    """The full canonical description of one simulation job."""
    return {
        "component": component_fingerprint(solver.component),
        "technology": _jsonable(solver.technology),
        "spec": _jsonable(solver.spec),
        "solver": solver.name,
        "engine_version": solver.engine_version(),
    }


def job_hash(solver: Solver) -> str:
    """sha256 hex digest of the job fingerprint (the cache key)."""
    payload = json.dumps(job_fingerprint(solver), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_path(solver: Solver, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{solver.name}_{job_hash(solver)[:16]}.npz"


def cached_run(solver: Solver, cache_dir: str | Path) -> SMatrix:
    """Return the cached SMatrix for this exact job, running only on a miss."""
    from .smatrix import SMatrix

    path = cache_path(solver, cache_dir)
    if path.exists():
        logger.info("cache hit: %s", path)
        return SMatrix.from_npz(str(path))
    sm = solver.run()
    path.parent.mkdir(parents=True, exist_ok=True)
    sm.to_npz(str(path))
    logger.info("cache store: %s", path)
    return sm
