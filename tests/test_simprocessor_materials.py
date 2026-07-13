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
        if isinstance(permittivity, complex):  # mirror tidy3d: permittivity must be REAL
            raise ValueError("permittivity must be a real number")
        self.permittivity = permittivity

    @classmethod
    def from_nk(cls, n, k, freq):
        m = cls(permittivity=float(n) ** 2)
        m.nk = (n, k, freq)
        return m

    def __eq__(self, other):
        return isinstance(other, _FakeMedium) and self.permittivity == other.permittivity


@pytest.fixture()
def fake_tidy3d(monkeypatch):
    td = types.ModuleType("tidy3d")
    td.Medium = _FakeMedium
    td.C_0 = 299792458000000.0  # um * Hz, as in tidy3d
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
    # a lossy [n, k] constant must go through the n/k constructor — tidy3d's
    # Medium.permittivity is strictly REAL (complex raised ValidationError;
    # found live by the air-clad PSR run)
    m = _load({"nk": [3.0, 0.1]}, fake_tidy3d)
    assert m.permittivity == 3.0**2
    assert m.nk[0] == 3.0 and m.nk[1] == 0.1


def test_nk_pair_lossless_stays_plain_medium(fake_tidy3d):
    m = _load({"nk": [1.0, 0.0]}, fake_tidy3d)  # e.g. an air superstrate
    assert m.permittivity == 1.0
    assert not hasattr(m, "nk")


def test_model_pair_lookup(fake_tidy3d):
    assert _load({"model": ["Si3N4", "Luke2015PMLStable"]}, fake_tidy3d) == "SIN_MEDIUM"


def test_model_pair_unknown_falls_back_to_silicon(fake_tidy3d, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="gds_fdtd.simprocessor"):
        assert _load({"model": ["Unobtainium", "X"]}, fake_tidy3d) == "CSI_MEDIUM"
    assert "not found" in caplog.text


def test_model_string_takes_first_variant(fake_tidy3d):
    assert _load({"model": "Si3N4"}, fake_tidy3d) == "SIN_MEDIUM"


def test_model_string_unknown_falls_back(fake_tidy3d, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="gds_fdtd.simprocessor"):
        assert _load({"model": "Nothingite"}, fake_tidy3d) == "CSI_MEDIUM"
    assert "not found" in caplog.text


def test_unparseable_spec_falls_back(fake_tidy3d, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="gds_fdtd.simprocessor"):
        assert _load({"weird": True}, fake_tidy3d) == "CSI_MEDIUM"
    assert "Could not parse" in caplog.text
