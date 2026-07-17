"""Solver adapters implementing the Solver contract (gds_fdtd.solvers.base).

The tidy3d / lumerical / beamz adapters are the supported API, reached via
``get_solver(name)``. The pre-0.5 ``fdtd_solver_*`` classes were removed in
0.6; the tidy3d scene-building engine lives in the internal ``_tidy3d_engine``
module.
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
    from .tidy3d import Tidy3DSolver  # noqa: F401  (import registers the solver)
with contextlib.suppress(ImportError):
    from .lumerical import LumericalSolver  # noqa: F401  (import registers the solver)
with contextlib.suppress(ImportError):
    from .beamz import BeamzSolver  # noqa: F401  (import registers the solver)
