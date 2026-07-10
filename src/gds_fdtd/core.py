"""
gds_fdtd simulation toolbox.

Legacy core module. The geometry classes moved to gds_fdtd.geometry in 0.5
(port -> Port, structure -> Structure, region -> Region, component -> Component,
layout -> LayoutSource); importing the old names from here still works but
emits a DeprecationWarning. ``parse_yaml_tech`` remains as a thin bridge that
routes through the modern ``gds_fdtd.technology.Technology`` model.
@author: Mustafa Hammood, 2025
"""

import warnings

from .geometry import (  # noqa: F401  (re-exported, non-deprecated helper names)
    c0_um,
    calculate_polygon_extension,
    initialize_ports_z,
    is_point_inside_polygon,
)

_DEPRECATED_GEOMETRY_NAMES = {
    "port": "Port",
    "structure": "Structure",
    "region": "Region",
    "component": "Component",
    "layout": "LayoutSource",
}


def __getattr__(name: str):
    """PEP 562 shim: serve the renamed geometry classes under their old names."""
    if name in _DEPRECATED_GEOMETRY_NAMES:
        new_name = _DEPRECATED_GEOMETRY_NAMES[name]
        warnings.warn(
            f"gds_fdtd.core.{name} is deprecated; use gds_fdtd.geometry.{new_name} "
            "instead (renamed in 0.5, removal at v1.0).",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import geometry

        return getattr(geometry, new_name)
    if name in ("Port", "Structure", "Region", "Component", "LayoutSource"):
        from . import geometry

        return getattr(geometry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def parse_yaml_tech(file_path: str) -> dict:
    """
    Legacy function for parsing YAML technology files.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed technology data in dictionary format.

    Note:
        Routes through the validated pydantic model (gds_fdtd.technology.Technology,
        the returned dict shape is identical to the legacy parser's.
    """
    from .technology import Technology

    return Technology.from_yaml(file_path).to_legacy_dict()
