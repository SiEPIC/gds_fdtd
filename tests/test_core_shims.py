"""Deprecation shims + legacy technology class (core.py) — release coverage."""

from __future__ import annotations

import pathlib

import pytest

TESTS_DIR = pathlib.Path(__file__).parent


def test_deprecated_geometry_names_warn():
    import gds_fdtd.core as core

    with pytest.warns(DeprecationWarning, match="deprecated"):
        cls = core.structure  # old lowercase name, renamed in WP2.1
    from gds_fdtd.geometry import Structure

    assert cls is Structure


def test_current_geometry_names_pass_through():
    import gds_fdtd.core as core
    from gds_fdtd.geometry import Component, Port

    assert core.Component is Component and core.Port is Port


def test_unknown_attribute_raises():
    import gds_fdtd.core as core

    with pytest.raises(AttributeError, match="no attribute"):
        _ = core.definitely_not_a_symbol


def test_legacy_technology_from_yaml_roundtrip():
    """The legacy dict-flavored technology class still reads v1 files."""
    from gds_fdtd.core import technology

    tech = technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))
    d = tech.to_dict()
    assert d["name"] == "EBeam"
    assert len(d["device"]) == 2
    assert d["substrate"][0]["z_span"] == -2
    assert "technology" in repr(tech)


def test_load_cell_missing_top_cell_raises():
    from gds_fdtd.lyprocessor import load_cell

    with pytest.raises(ValueError, match="not found"):
        load_cell(str(TESTS_DIR / "si_sin_escalator.gds"), top_cell="nope")


def test_load_cell_ambiguous_top_cell_raises():
    from gds_fdtd.lyprocessor import load_cell

    with pytest.raises(ValueError, match="More than one top cell"):
        load_cell(str(TESTS_DIR.parent / "examples" / "devices.gds"))


def test_load_device_writes_extended_gds(tmp_path):
    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.lyprocessor import load_device

    tech = parse_yaml_tech(str(TESTS_DIR / "tech_lumerical.yaml"))
    comp = load_device(str(TESTS_DIR / "si_sin_escalator.gds"), tech, output_dir=str(tmp_path))
    assert comp.ports
    assert list(tmp_path.glob("*_with_extensions.gds"))
