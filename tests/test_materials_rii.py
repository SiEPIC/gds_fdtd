"""refractiveindex.info -> engine material conversions."""

from __future__ import annotations

import pathlib

import pytest

from gds_fdtd.materials.rii import load_rii_material

RII_DB = pathlib.Path(__file__).parent / "rii_db"


def _si():
    return load_rii_material("main", "Si", "Li-293", db_dir=str(RII_DB))


def test_rii_nk_dispersive():
    m = _si()
    # the tabulated Si fixture disperses across the telecom band
    assert m.n_at(1.5) != m.n_at(1.6)
    assert complex(m.nk_at(1.55)).real == pytest.approx(3.476, abs=0.01)


def test_rii_to_tidy3d_medium_roundtrip():
    pytest.importorskip("tidy3d")
    import tidy3d as td

    med = _si().to_tidy3d_medium(max_num_poles=3)
    # the fitted dispersive medium reproduces the source index at band center
    n_fit = float(med.nk_model(td.C_0 / 1.55)[0])
    assert n_fit == pytest.approx(3.476, abs=0.05)


def test_rii_to_tidy3d_medium_without_tidy3d(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name.startswith("tidy3d"):
            raise ImportError("tidy3d blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError, match="needs tidy3d"):
        _si().to_tidy3d_medium()
