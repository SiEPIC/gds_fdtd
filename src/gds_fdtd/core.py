"""
gds_fdtd simulation toolbox.

Legacy core module. The geometry classes moved to gds_fdtd.geometry in 0.5
(port -> Port, structure -> Structure, region -> Region, component -> Component,
layout -> LayoutSource); importing the old names from here still works but
emits a DeprecationWarning. The technology class and the legacy s-parameter
classes still live here until their v1.0 removal.
@author: Mustafa Hammood, 2025
"""

import warnings

import yaml

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


class technology:
    """
    Technology class for managing fabrication process definitions.

    This class handles technology information including layer definitions,
    material properties, and process parameters from various sources like YAML files.

    Attributes:
        name (str): Name of the technology.
        substrate (list): Substrate layer definitions.
        superstrate (list): Superstrate layer definitions.
        pinrec (list): Pin recognition layer definitions.
        devrec (list): Device recognition layer definitions.
        device (list): Device layer definitions with materials and geometry.
    """

    def __init__(self, name: str = "Unknown"):
        """
        Initialize a technology object.

        Args:
            name (str): Name of the technology. Defaults to "Unknown".
        """
        self.name = name
        self.substrate = []
        self.superstrate = []
        self.pinrec = []
        self.devrec = []
        self.device = []

    @classmethod
    def from_yaml(cls, file_path: str) -> "technology":
        """
        Create a technology object from a YAML file.

        Args:
            file_path (str): Path to the YAML technology file.

        Returns:
            technology: Technology object parsed from the YAML file.
        """
        with open(file_path) as file:
            data = yaml.safe_load(file)

        tech_data = data.get("technology", {})
        tech = cls(name=tech_data.get("name", "Unknown"))

        # Parse substrate layer
        substrate = tech_data.get("substrate", {})
        if substrate:
            tech.substrate.append(
                {
                    "z_base": substrate.get("z_base"),
                    "z_span": substrate.get("z_span"),
                    "material": substrate.get("material"),
                }
            )

        # Parse superstrate layer
        superstrate = tech_data.get("superstrate", {})
        if superstrate:
            tech.superstrate.append(
                {
                    "z_base": superstrate.get("z_base"),
                    "z_span": superstrate.get("z_span"),
                    "material": superstrate.get("material"),
                }
            )

        # Parse pinrec layers
        tech.pinrec = [
            {"layer": list(pinrec.get("layer"))} for pinrec in tech_data.get("pinrec", [])
        ]

        # Parse devrec layers
        tech.devrec = [
            {"layer": list(devrec.get("layer"))} for devrec in tech_data.get("devrec", [])
        ]

        # Parse device layers
        tech.device = [
            {
                "layer": list(device.get("layer")),
                "z_base": device.get("z_base"),
                "z_span": device.get("z_span"),
                "material": device.get("material"),
                "sidewall_angle": device.get("sidewall_angle"),
            }
            for device in tech_data.get("device", [])
        ]

        return tech

    def to_dict(self) -> dict:
        """
        Convert the technology object to a dictionary format.

        Returns:
            dict: Technology data in dictionary format (compatible with legacy code).
        """
        return {
            "name": self.name,
            "substrate": self.substrate,
            "superstrate": self.superstrate,
            "pinrec": self.pinrec,
            "devrec": self.devrec,
            "device": self.device,
        }

    def __repr__(self) -> str:
        """String representation of the technology object."""
        return f"technology(name='{self.name}', devices={len(self.device)} layers)"
