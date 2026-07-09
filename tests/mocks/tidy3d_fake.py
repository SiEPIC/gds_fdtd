"""Permissive fake of the ``tidy3d`` package for OFFLINE setup-path tests.

Every ``td.X(...)`` returns a recording node that accepts any kwargs and
answers any attribute — enough for the legacy adapter's scene CONSTRUCTION
to run without tidy3d installed. Only the constants the adapter's own math
consumes (``C_0``, ``inf``) are real numbers. The real engine paths stay
covered by the all-extras CI leg and the live validations.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class FakeNode:
    """Accepts any construction, records it, answers any attribute."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._kwargs.get(name, FakeNode())

    def __call__(self, *args: Any, **kwargs: Any) -> FakeNode:
        return FakeNode(*args, **kwargs)

    def __len__(self) -> int:  # e.g. len(modeler.sim_dict) in build summaries
        return 0

    def __int__(self) -> int:  # e.g. int(np.prod(grid.num_cells)) in estimates
        return 0

    def __float__(self) -> float:
        return 0.0

    def __iter__(self):
        return iter(())

    def __repr__(self) -> str:  # keep job hashes address-free
        return f"FakeNode({sorted(self._kwargs)})"


class _NodeFactory(types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        # an INSTANCE: callable like a class (td.Structure(...)) and
        # attribute-permissive like a namespace (td.GridSpec.auto(...))
        return FakeNode()


def install(monkeypatch) -> types.ModuleType:
    """Install fake tidy3d (+plugins.smatrix, +web) into sys.modules."""
    td = _NodeFactory("tidy3d")
    td.C_0 = 299792458000000.0  # um * Hz, as tidy3d defines it
    td.inf = float("inf")
    td.Medium = FakeNode
    td.material_library = {
        "cSi": {"Li1993_293K": FakeNode(name="cSi")},
        "Si3N4": {"Luke2015PMLStable": FakeNode(name="Si3N4")},
    }

    plugins = types.ModuleType("tidy3d.plugins")
    smatrix = types.ModuleType("tidy3d.plugins.smatrix")
    smatrix.ModalComponentModeler = FakeNode
    smatrix.Port = FakeNode
    plugins.smatrix = smatrix
    web = types.ModuleType("tidy3d.web")

    def _no_network(*a, **k):
        raise AssertionError("offline test tried to reach tidy3d.web")

    web.run = _no_network
    web.upload = _no_network

    for name, mod in {
        "tidy3d": td,
        "tidy3d.plugins": plugins,
        "tidy3d.plugins.smatrix": smatrix,
        "tidy3d.web": web,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for mod_name in ("gds_fdtd.solver_tidy3d", "gds_fdtd.solvers.tidy3d"):
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
    return td
