"""Solver adapters implementing the Phase-3 contract (gds_fdtd.solvers.base).

The tidy3d/lumerical adapters are ported onto this contract in WP3.1c/d; the
legacy gds_fdtd.solver_tidy3d / solver_lumerical modules keep working until
then.
"""

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
