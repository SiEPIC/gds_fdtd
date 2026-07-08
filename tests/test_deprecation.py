"""WP2.1: the old gds_fdtd.core geometry names keep working with a warning."""

from __future__ import annotations

import warnings

import pytest

from gds_fdtd import core, geometry


@pytest.mark.parametrize(
    "old,new",
    [
        ("port", "Port"),
        ("structure", "Structure"),
        ("region", "Region"),
        ("component", "Component"),
        ("layout", "LayoutSource"),
    ],
)
def test_old_core_names_warn_and_resolve(old, new):
    with pytest.warns(DeprecationWarning, match=f"gds_fdtd.core.{old} is deprecated"):
        obj = getattr(core, old)
    assert obj is getattr(geometry, new)


def test_helper_functions_reexported_silently():
    # unchanged names: re-exported without deprecation noise
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert core.is_point_inside_polygon is geometry.is_point_inside_polygon
        assert core.calculate_polygon_extension is geometry.calculate_polygon_extension
        assert core.initialize_ports_z is geometry.initialize_ports_z


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = core.definitely_not_a_thing
