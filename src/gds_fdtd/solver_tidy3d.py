"""Deprecated public alias for the Tidy3D solver.

Use the solver registry instead::

    from gds_fdtd.solvers import get_solver
    solver = get_solver("tidy3d")(component, tech, spec)

``fdtd_solver_tidy3d`` remains importable through the 0.5.x series and will be
removed in 1.0. Its implementation lives in
``gds_fdtd.solvers._tidy3d_engine``.
"""

import warnings

from gds_fdtd.solvers._tidy3d_engine import _TidyEngine

__all__ = ["fdtd_solver_tidy3d"]

_DEPRECATION = (
    "fdtd_solver_tidy3d is deprecated since gds_fdtd 0.5 and will be removed in "
    "1.0; use gds_fdtd.solvers.get_solver('tidy3d')(component, technology, spec)."
)


class fdtd_solver_tidy3d(_TidyEngine):
    """Deprecated Tidy3D solver alias.

    .. deprecated:: 0.5
        Use :func:`gds_fdtd.solvers.get_solver` with ``"tidy3d"``; removed in 1.0.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
