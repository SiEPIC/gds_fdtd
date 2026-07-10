"""Deprecated public alias for the Tidy3D solver.

Use the solver registry instead::

    from gds_fdtd.solvers import get_solver
    solver = get_solver("tidy3d")(component, tech, spec)

``fdtd_solver_tidy3d`` remains importable through the 0.5.x series and will be
removed in 1.0. Its implementation lives in
``gds_fdtd.solvers._tidy3d_engine``.
"""

from gds_fdtd.solvers._tidy3d_engine import _TidyEngine as fdtd_solver_tidy3d

__all__ = ["fdtd_solver_tidy3d"]
