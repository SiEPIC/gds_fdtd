"""Recording mock of the ``lumapi`` module.

Injected as ``sys.modules["lumapi"]`` by the ``mock_lumapi`` fixture so the
Lumerical adapter's ``run()`` path executes in CI with no license: it opens
a session, replays the generated ``.lsf`` script, runs the sweep and exports
a recorded real ``.dat``. Every call is recorded so tests can assert the
session behavior offline.
"""

from __future__ import annotations

from typing import Any


class MockFDTD:
    """Stands in for ``lumapi.FDTD``: records calls, answers known queries."""

    #: structured return values for the getters the setup/run path consumes
    RESULTS: dict[str, Any] = {}
    #: every instance created this session (run() builds its FDTD internally)
    INSTANCES: list[MockFDTD] = []
    #: F7: when True, the 4-arg (2025) setresource signature raises, forcing
    #: the adapter's 2024 fallback
    RAISE_ON_4ARG_SETRESOURCE: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.script: list[str] = []
        self._remove_budget = 3  # real lumapi raises once no sweep params remain
        MockFDTD.INSTANCES.append(self)

    # every unknown attribute is a recordable Lumerical function
    def __getattr__(self, name: str):
        def method(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            if name == "eval" and args:
                self.script.append(str(args[0]))
            if name == "setresource" and len(args) == 4 and MockFDTD.RAISE_ON_4ARG_SETRESOURCE:
                raise RuntimeError("unknown option 'device type' (simulated Lumerical 2024)")
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

    MockFDTD.INSTANCES = []
    MockFDTD.RAISE_ON_4ARG_SETRESOURCE = False
    mod = types.ModuleType("lumapi")
    mod.FDTD = MockFDTD
    monkeypatch.setitem(sys.modules, "lumapi", mod)
    return MockFDTD
