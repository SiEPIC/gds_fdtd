"""Tests for gds_fdtd.simprocessor (started in WP1.4)."""

from __future__ import annotations

import pytest

from gds_fdtd.simprocessor import get_material


def test_get_material_lum_db_happy():
    mat = get_material({"material": {"lum_db": {"model": "Si (Silicon) - Palik"}}})
    assert mat["lum"] == "Si (Silicon) - Palik"
    assert mat["tidy3d"] is None


def test_get_material_lum_db_without_model_raises():
    # bug B7: this was an UnboundLocalError
    with pytest.raises(ValueError, match="lum_db"):
        get_material({"material": {"lum_db": {"typo_model": "x"}}})


def test_get_material_tidy3d_db_without_nk_or_model_raises():
    with pytest.raises(ValueError, match="tidy3d_db"):
        get_material({"material": {"tidy3d_db": {"bogus": 1}}})


def test_get_material_non_mapping_material_raises():
    with pytest.raises(ValueError, match="material"):
        get_material({"material": "SiO2"})
