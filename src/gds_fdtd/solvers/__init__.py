"""Solver adapters implementing the Phase-3 contract (gds_fdtd.solvers.base).

The tidy3d/lumerical/beamz adapters implement this contract; the
legacy gds_fdtd.solver_tidy3d / solver_lumerical modules keep working until
then.
"""

import contextlib

from .base import (
    ResourceEstimate,
    SetupArtifacts,
    Solver,
    SolverCapabilities,
    available_solvers,
    get_solver,
    register_solver,
)

__all__ = [
    "ResourceEstimate",
    "SetupArtifacts",
    "Solver",
    "SolverCapabilities",
    "available_solvers",
    "get_solver",
    "register_solver",
]

# Adapters register themselves on import; missing optional engines are fine.
with contextlib.suppress(ImportError):
    from .tidy3d import Tidy3DSolver  # noqa: F401
with contextlib.suppress(ImportError):
    from .lumerical import LumericalSolver  # noqa: F401
with contextlib.suppress(ImportError):
    from .beamz import BeamzSolver  # noqa: F401
