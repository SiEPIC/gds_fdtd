"""Material-resolution branch logic in simprocessor, via a minimal fake tidy3d.

The logic under test is OURS (nk float/[n,k] handling, model lookup,
fallbacks) — the fake supplies just td.Medium and td.material_library.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeMedium:
    def __init__(self, permittivity=None):
        self.permittivity = permittivity

    def __eq__(self, other):
        return isinstance(other, _FakeMedium) and self.permittivity == other.permittivity


@pytest.fixture()
def fake_tidy3d(monkeypatch):
    td = types.ModuleType("tidy3d")
    td.Medium = _FakeMedium
    td.material_library = {
        "cSi": {"Li1993_293K": "CSI_MEDIUM"},
        "Si3N4": {"Luke2015PMLStable": "SIN_MEDIUM", "Philipp1973": "SIN_ALT"},
    }
    monkeypatch.setitem(sys.modules, "tidy3d", td)
    return td


def _load(spec, fake):
    from gds_fdtd.simprocessor import _load_tidy3d_material

    return _load_tidy3d_material(spec)


def test_nk_float_becomes_medium(fake_tidy3d):
    m = _load({"nk": 1.5}, fake_tidy3d)
    assert m.permittivity == 1.5**2


def test_nk_complex_pair(fake_tidy3d):
    m = _load({"nk": [3.0, 0.1]}, fake_tidy3d)
    assert m.permittivity == (3.0 + 0.1j) ** 2


def test_model_pair_lookup(fake_tidy3d):
    assert _load({"model": ["Si3N4", "Luke2015PMLStable"]}, fake_tidy3d) == "SIN_MEDIUM"


def test_model_pair_unknown_falls_back_to_silicon(fake_tidy3d, capsys):
    assert _load({"model": ["Unobtainium", "X"]}, fake_tidy3d) == "CSI_MEDIUM"
    assert "not found" in capsys.readouterr().out


def test_model_string_takes_first_variant(fake_tidy3d):
    assert _load({"model": "Si3N4"}, fake_tidy3d) == "SIN_MEDIUM"


def test_model_string_unknown_falls_back(fake_tidy3d, capsys):
    assert _load({"model": "Nothingite"}, fake_tidy3d) == "CSI_MEDIUM"
    assert "not found" in capsys.readouterr().out


def test_unparseable_spec_falls_back(fake_tidy3d, capsys):
    assert _load({"weird": True}, fake_tidy3d) == "CSI_MEDIUM"
    assert "Could not parse" in capsys.readouterr().out
