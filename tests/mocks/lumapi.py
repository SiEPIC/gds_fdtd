"""Recording mock of the ``lumapi`` module (WP7.1.3).

Injected as ``sys.modules["lumapi"]`` by the ``mock_lumapi`` fixture so the
legacy Lumerical adapter's ENTIRE setup path runs in CI with no license:
every ``eval``/``set``/``setnamed``/... call is recorded as a transcript
that tests can assert against (the offline equivalent of a .lsf golden).
"""

from __future__ import annotations

from typing import Any


class MockFDTD:
    """Stands in for ``lumapi.FDTD``: records calls, answers known queries."""

    #: structured return values for the getters the setup/run path consumes
    RESULTS: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.script: list[str] = []
        self._remove_budget = 3  # real lumapi raises once no sweep params remain

    # every unknown attribute is a recordable Lumerical function
    def __getattr__(self, name: str):
        def method(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            if name == "eval" and args:
                self.script.append(str(args[0]))
            if name == "exportsweep" and len(args) >= 2:
                # write a REAL recorded engine .dat so run() -> process_dat
                # exercises the genuine results pipeline offline
                import pathlib
                import shutil

                recorded = pathlib.Path(__file__).parent.parent / "recorded"
                shutil.copy(recorded / "si_sin_escalator.dat", args[1])
            if name == "removesweepparameter":
                # the adapter drains existing entries in a while-True loop that
                # exits on the exception the REAL API raises when none remain
                self._remove_budget -= 1
                if self._remove_budget < 0:
                    raise RuntimeError("no sweep parameters left (mock)")
            if name == "getv":
                var = args[0] if args else None
                if ("getv", var) in self.RESULTS:
                    return self.RESULTS[("getv", var)]
                if var == "gds_layers":
                    # layer list of the exported escalator GDS, as the real
                    # getlayerlist reports it (newline-separated string)
                    return "1:0\n4:0\n1:10\n68:0"
                # runsystemcheck on Lumerical 2025 returns {} (finding F6)
                return {}
            if name in ("getresult", "getsweepresult"):
                key = (name, *args)
                if key in self.RESULTS:
                    return self.RESULTS[key]
                return {}
            return None

        return method

    # convenience for assertions ------------------------------------
    @property
    def transcript(self) -> str:
        return "\n".join(self.script)

    def calls_named(self, name: str) -> list[tuple[str, tuple, dict]]:
        return [c for c in self.calls if c[0] == name]


def install(monkeypatch) -> type[MockFDTD]:
    """Put this module in sys.modules as ``lumapi`` (scoped via monkeypatch)."""
    import sys
    import types

    mod = types.ModuleType("lumapi")
    mod.FDTD = MockFDTD
    monkeypatch.setitem(sys.modules, "lumapi", mod)
    # the adapter binds `from lumapi import FDTD` at ITS import time; drop any
    # previously-imported copy so it rebinds against the mock
    monkeypatch.delitem(sys.modules, "gds_fdtd.solver_lumerical", raising=False)
    return MockFDTD
