"""Per-engine material-source selection (gds_fdtd.materials.select)."""

from __future__ import annotations

import pytest

from gds_fdtd.errors import MaterialSourceError
from gds_fdtd.materials.select import (
    available_sources,
    check_materials,
    select_source,
    source_index,
)

FULL = {
    "tidy3d_db": {"model": ["cSi", "Green2008"]},
    "lum_db": {"model": "Si (Silicon) - Palik"},
    "rii": {"shelf": "main", "book": "Si", "page": "Salzberg"},
    "nk": 3.476,
}


def test_default_precedence_eda_rii_nk():
    # tidy3d & lumerical have their own DB -> eda wins
    assert select_source(FULL, "tidy3d") == "eda"
    assert select_source(FULL, "lumerical") == "eda"
    # beamz has no vendor DB -> rii beats nk
    assert select_source(FULL, "beamz") == "rii"


def test_beamz_never_selects_eda():
    m = {"tidy3d_db": {"nk": 3.4}, "nk": 3.476}
    assert available_sources(m, "beamz") == ["nk"]
    assert select_source(m, "beamz") == "nk"


def test_precedence_falls_through_when_absent():
    assert select_source({"rii": {"shelf": "s"}, "nk": 3.4}, "tidy3d") == "rii"  # no eda
    assert select_source({"nk": 3.4}, "tidy3d") == "nk"  # only nk


def test_explicit_source_overrides():
    for engine in ("tidy3d", "lumerical", "beamz"):
        assert select_source({**FULL, "source": "rii"}, engine) == "rii"
    assert select_source({**FULL, "source": "nk"}, "tidy3d") == "nk"


def test_explicit_source_unavailable_errors():
    with pytest.raises(MaterialSourceError, match="requested but is not defined"):
        select_source({"nk": 3.4, "source": "rii"}, "beamz", name="Si")
    # eda requested on beamz (which has no eda) -> unavailable
    with pytest.raises(MaterialSourceError):
        select_source({"tidy3d_db": {"nk": 3.4}, "source": "eda"}, "beamz")


def test_no_source_for_engine_errors():
    # a Lumerical-only material asked to run on tidy3d
    with pytest.raises(MaterialSourceError, match="no optical-constant source"):
        select_source({"lum_db": {"model": "Si - Palik"}}, "tidy3d", name="Si")


def test_check_materials_reports_per_material():
    tech = {
        "substrate": [{"material": {"nk": 1.44}}],
        "superstrate": [{"material": {"nk": 1.44}}],
        "device": [
            {"layer": [1, 0], "material": {"lum_db": {"model": "Si"}}},  # no tidy3d source
        ],
    }
    problems = check_materials(tech, "tidy3d")
    assert len(problems) == 1 and "device layer 0" in problems[0]
    assert check_materials(tech, "lumerical") == []  # lumerical is happy


def test_source_index_nk_and_rii():
    assert source_index({"nk": 3.5}, "nk", 1.55) == pytest.approx(3.5)
    assert source_index({"nk": [3.5, 0.1]}, "nk", 1.55) == pytest.approx(3.5 + 0.1j)
    import pathlib

    db = pathlib.Path(__file__).parent / "rii_db"
    n = source_index(
        {"rii": {"shelf": "main", "book": "Si", "page": "Li-293"}}, "rii", 1.55, rii_db_dir=str(db)
    )
    assert n.real == pytest.approx(3.476, abs=0.01)
