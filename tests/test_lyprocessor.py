"""
Tests for gds_fdtd.lyprocessor.

NOTE (WP0.2/WP0.4): this file previously (a) stubbed the entire klayout/prefab/
simprocessor modules into sys.modules AT IMPORT TIME — global state that poisoned
every test collected after it (the golden tests received a fake simprocessor) —
(b) contained commented-out tests, and (c) contained a "coverage padding" test
that exec-compiled `pass` statements at every line of the module under test,
faking 100% coverage. All three are gone. Module fakes are now scoped with
monkeypatch inside the tests that need them. Real KLayout-backed tests for
load_cell/load_region/load_structure/load_ports arrive with the Phase 1 bug-fix
WPs (see MODERNIZATION_PLAN.md).
"""

from __future__ import annotations

import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gds_fdtd import lyprocessor as lp


# =============================================================================
# Pure-python helpers (dilate, dilate_1d)
# =============================================================================
def test_dilate_rectangle():
    assert lp.dilate([[0, 0], [2, 0], [2, 1], [0, 1]], 1) == [[-1, -1], [3, -1], [3, 2], [-1, 2]]


@pytest.mark.parametrize(
    "v,e,d,expect",
    [
        ([[0, 0], [4, 0]], 1, "x", [[-1, 0], [5, 0]]),
        ([[0, 0], [0, 4]], 2, "y", [[0, -2], [0, 6]]),
        ([[0, 0], [4, 2]], 1, "xy", [[0, -1], [5, 3]]),
    ],
)
def test_dilate_1d_ok(v, e, d, expect):
    assert lp.dilate_1d(v, e, d) == expect


def test_dilate_1d_bad_dim():
    with pytest.raises(ValueError):
        lp.dilate_1d([[0, 0], [1, 1]], dim="z")


# =============================================================================
# apply_prefab (prefab module faked LOCALLY via monkeypatch — no global state)
# =============================================================================
def test_apply_prefab_runs(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    class _PrefDev:
        def predict(self, **_):
            class _Pred:
                def binarize(self):
                    class _Bin:
                        def to_gds(self, **_):
                            pass

                    return _Bin()

            return _Pred()

    prefab = ModuleType("prefab")
    prefab.read = SimpleNamespace(from_gds=lambda **_: _PrefDev())
    prefab.models = {"ANT_NanoSOI_ANF1_d9": object()}
    monkeypatch.setitem(sys.modules, "prefab", prefab)

    f = tmp_path / "dummy.gds"
    f.write_bytes(b"")
    lp.apply_prefab(str(f), "TOP")  # should not raise
