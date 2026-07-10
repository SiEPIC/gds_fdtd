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
        # ascending point order
        ([[0, 0], [4, 0]], 1, "x", [[-1, 0], [5, 0]]),
        ([[0, 0], [0, 4]], 2, "y", [[0, -2], [0, 6]]),
        # NOTE: expectation updated in WP1.1 — the old "xy" implementation failed
        # to extend x1 (expected [[0, -1], [5, 3]]); both corners now extend.
        ([[0, 0], [4, 2]], 1, "xy", [[-1, -1], [5, 3]]),
        # descending point order — the old code returned [] here (bug B1:
        # `[x, y] * sign` multiplied the LIST by -1)
        ([[0, 4], [0, 0]], 2, "y", [[0, 6], [0, -2]]),
        ([[4, 0], [0, 0]], 1, "x", [[5, 0], [-1, 0]]),
        ([[4, 2], [0, 0]], 1, "xy", [[5, 3], [-1, -1]]),
        # negative extension behaves like its absolute value (documented)
        ([[0, 0], [0, 4]], -2, "y", [[0, -2], [0, 6]]),
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
    out = lp.apply_prefab(str(f), str(tmp_path / "out.gds"), "TOP")  # should not raise
    assert out == str(tmp_path / "out.gds")

    # WP1.8: apply_prefab must never overwrite its input
    with pytest.raises(ValueError, match="never overwrites"):
        lp.apply_prefab(str(f), str(f), "TOP")


# =============================================================================
# load_region (WP1.4, bug B8) — real KLayout on the shipped test GDS
# =============================================================================
@pytest.fixture
def escalator_cell():
    fname = str(pathlib.Path(__file__).parent / "si_sin_escalator.gds")
    # keep the layout alive for the duration of the test: the cell's layout()
    # is destroyed when the pya.Layout object is garbage collected
    cell, layout = lp.load_cell(fname)
    yield cell
    del layout


def test_load_region_happy_path(escalator_cell):
    region = lp.load_region(escalator_cell, layer=[68, 0], z_center=0.11, z_span=4)
    assert len(region.vertices) >= 4
    assert region.x_span > 0 and region.y_span > 0


def test_load_region_missing_devrec_layer_raises(escalator_cell):
    with pytest.raises(ValueError, match="No DevRec"):
        lp.load_region(escalator_cell, layer=[123, 45])


# =============================================================================
# load_device (WP1.8, bug B15) — returns the component, never touches the input
# =============================================================================
def test_load_device_returns_component_and_preserves_input(tmp_path):
    from gds_fdtd.technology import Technology

    src = pathlib.Path(__file__).parent / "si_sin_escalator.gds"
    gds = tmp_path / "device.gds"
    gds.write_bytes(src.read_bytes())
    before = gds.read_bytes()

    tech = Technology.from_yaml(str(pathlib.Path(__file__).parent / "tech_lumerical.yaml"))
    out_dir = tmp_path / "out"
    comp = lp.load_device(str(gds), tech=tech, output_dir=str(out_dir))

    assert comp is not None and len(comp.ports) == 2  # previously returned None
    assert gds.read_bytes() == before, "input GDS was modified"
    assert (out_dir / "si_sin_escalator_with_extensions.gds").exists()
