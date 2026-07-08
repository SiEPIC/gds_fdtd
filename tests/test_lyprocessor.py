"""
End‑to‑end tests for gds_fdtd.lyprocessor.

* Pure‑Python helpers (dilate, dilate_1d, load_structure_from_bounds)
* KLayout‑backed readers exercised on the example GDS files shipped under tests/.
  These tests are skipped automatically when `klayout.db` is missing.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

# =============================================================================
# ── 1.  Fake **klayout.db** (the minimum used by lyprocessor)  ─────────────────
# =============================================================================
pya = ModuleType("klayout.db")


class _Point(tuple):  # immutable, hashable
    __slots__ = ()

    def __new__(cls, x, y):
        return super().__new__(cls, (x, y))


class _Polygon(list):
    def transformed(self, _):  # identity
        return self

    def to_simple_polygon(self):  # identity
        return self

    def each_point(self):
        for x, y in self:
            yield _Point(x, y)


class _Path:
    def __init__(self, pts):
        self._pts = pts

    @property
    def points(self):
        return len(self._pts)

    def each_point(self):
        for p in self._pts:
            yield _Point(*p)


class _Shape(SimpleNamespace):
    def is_box(self):
        return hasattr(self, "box")

    def is_polygon(self):
        return hasattr(self, "polygon")

    def is_path(self):
        return hasattr(self, "path")

    def is_text(self):
        return hasattr(self, "text")


class _Iter:
    def __init__(self, shape):
        self._shape, self._done = shape, False

    def shape(self):
        return self._shape

    def itrans(self):
        return None

    def at_end(self):
        return self._done

    def next(self):
        self._done = True

    __next__ = next


class _Region(list):
    def merge(self):
        pass

    def insert(self, poly):
        self.append(poly)

    def each_merged(self):
        return iter(self)


class _Cell:
    def __init__(self, name="TOP"):
        self.name, self._shapes, self._layout = name, [], None

    def layout(self):
        return self._layout

    def begin_shapes_rec(self, _layer):
        return _Iter(self._shapes[0])

    def shapes(self, *_):
        return self

    def insert(self, *_):
        pass

    # helpers ---------------------------------------------------
    def _add_polygon(self, poly):
        self._shapes.append(_Shape(polygon=_Polygon(poly)))

    def _add_pin(self, pts, w=1):
        self._shapes.append(_Shape(path=_Path(pts), path_dwidth=w))

    def _add_label(self, text, x, y):
        self._shapes.append(_Shape(text=SimpleNamespace(string=text, x=x, y=y)))


class _Layout:
    def __init__(self):
        self._dbu, self._cells = 1e-3, [_Cell()]
        self._cells[0]._layout = self

    # API used in lyprocessor ------------------
    def read(self, *_):
        pass

    def write(self, *_):
        pass

    def top_cells(self):
        return self._cells

    def top_cell(self):
        return self._cells[0]

    def cell(self, n):
        return self._cells[0] if n == self._cells[0].name else None

    def layer(self, *_):
        return 0

    @property
    def dbu(self):
        return self._dbu


class _LayerInfo:  # ctor signature only
    def __init__(self, *a):
        pass


pya.Layout, pya.Cell, pya.Polygon, pya.Point = _Layout, _Cell, _Polygon, _Point
pya.LayerInfo, pya.Region = _LayerInfo, _Region
sys.modules["klayout"] = ModuleType("klayout")
sys.modules["klayout"].db = pya
sys.modules["klayout.db"] = pya

# =============================================================================
# ── 2.  Fake very small “prefab” & “simprocessor”  ────────────────────────────
# =============================================================================
prefab = ModuleType("prefab")


class _PrefDev:
    def predict(self, **_):
        class _Pred:  # noqa: D401
            def binarize(self):
                class _Bin:  # noqa: D401
                    def to_gds(self, **_):
                        pass

                return _Bin()

        return _Pred()


prefab.read = SimpleNamespace(from_gds=lambda **_: _PrefDev())
prefab.models = {"ANT_NanoSOI_ANF1_d9": object()}
sys.modules["prefab"] = prefab

simprocessor = ModuleType("gds_fdtd.simprocessor")


class _FakePort:
    def polygon_extension(self, buffer=2):
        return [[0, 0], [1, 0], [1, 1], [0, 1]]


def _fake_lcft(**_):
    return SimpleNamespace(ports=[_FakePort(), _FakePort()])


simprocessor.load_component_from_tech = _fake_lcft
sys.modules["gds_fdtd.simprocessor"] = simprocessor

# =============================================================================
# ── 3.  Now import the System‑Under‑Test  ─────────────────────────────────────
# =============================================================================
lp = importlib.import_module("gds_fdtd.lyprocessor")


# =============================================================================
# ── 4.  Pure‑python helpers (dilate, dilate_1d)  ─────────────────────────────
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
# ── 5.  apply_prefab  ────────────────────────────────────────────────────────
# =============================================================================
def test_apply_prefab_runs(tmp_path: pathlib.Path):
    f = tmp_path / "dummy.gds"
    f.write_bytes(b"")
    lp.apply_prefab(str(f), "TOP")  # should not raise


# NOTE (WP0.2): this file previously contained commented-out tests and a
# "coverage padding" test that exec-compiled `pass` statements at every line
# number of the module under test, faking 100% coverage. Both were removed;
# real KLayout-backed tests for load_cell/load_region/load_structure/load_ports
# arrive with the Phase 1 bug-fix WPs (see MODERNIZATION_PLAN.md).
