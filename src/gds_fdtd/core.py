"""
gds_fdtd simulation toolbox.

Legacy core module. The geometry classes moved to gds_fdtd.geometry in WP2.1
(port -> Port, structure -> Structure, region -> Region, component -> Component,
layout -> LayoutSource); importing the old names from here still works but
emits a DeprecationWarning. The technology class and the legacy s-parameter
classes still live here until WP2.2 / WP2.4.
@author: Mustafa Hammood, 2025
"""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
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
            "instead (renamed in WP2.1, removal at v1.0).",
            DeprecationWarning,
            stacklevel=2,
        )
        from . import geometry

        return getattr(geometry, new_name)
    if name in ("Port", "Structure", "Region", "Component", "LayoutSource"):
        from . import geometry

        return getattr(geometry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class s_parameters:
    def __init__(self, entries=None):
        if entries is None:
            self._entries = []
        else:
            self._entries = entries
        return

    @property
    def S(self):
        return dict(zip([i.label for i in self._entries], self._entries, strict=False))

    def add_param(self, sparam):
        self._entries.append(sparam)

    def entries_in_mode(self, mode_in=0, mode_out=0):
        entries = []
        for s in self._entries:
            if s.mode_in == mode_in and s.mode_out == mode_out:
                entries.append(s)
        return entries

    def entries_in_ports(self, input_entries=None, idx_in=0, idx_out=0):
        entries = []
        if input_entries is None:
            input_entries = self._entries

        for s in input_entries:
            if s.idx_in == idx_in and s.idx_out == idx_out:
                entries.append(s)
        return entries

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Wavelength [microns]")
        ax.set_ylabel("Transmission [dB]")
        for i in self._entries:
            logging.info("Mode amplitudes in each port: \n")
            mag = [10 * np.log10(abs(i) ** 2) for i in i.s]
            ax.plot(c0_um / i.freq, mag, label=i.label)
        ax.legend()
        fig.show()
        return fig, ax


class sparam:
    def __init__(self, idx_in, idx_out, mode_in, mode_out, freq, s):
        self.idx_in = idx_in
        self.idx_out = idx_out
        self.mode_in = mode_in
        self.mode_out = mode_out
        self.freq = freq
        self.s = s

    @property
    def label(self):
        return f"S{self.idx_out}{self.idx_in}_idx{self.mode_out}{self.mode_in}"

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot((c0_um) / np.array(self.freq), 10 * np.log10(np.abs(self.s) ** 2))
        ax.set_xlabel("Wavelength [um]")
        ax.set_ylabel("Transmission [dB]")
        ax.set_title("Frequency vs S")
        fig.show()
        return fig, ax


def parse_yaml_tech(file_path: str) -> dict:
    """
    Legacy function for parsing YAML technology files.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed technology data in dictionary format.

    Note:
        This function is deprecated. Use technology.from_yaml() instead.
    """
    tech = technology.from_yaml(file_path)
    return tech.to_dict()


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

    def add_device_layer(
        self,
        layer: list[int],
        z_base: float,
        z_span: float,
        material: dict,
        sidewall_angle: float = 90.0,
    ) -> None:
        """
        Add a device layer to the technology.

        Args:
            layer (list[int]): GDS layer specification as [layer_number, datatype].
            z_base (float): Base z-coordinate of the layer.
            z_span (float): Thickness of the layer.
            material (dict): Material properties dictionary.
            sidewall_angle (float, optional): Sidewall angle in degrees. Defaults to 90.0.
        """
        self.device.append(
            {
                "layer": layer,
                "z_base": z_base,
                "z_span": z_span,
                "material": material,
                "sidewall_angle": sidewall_angle,
            }
        )

    def get_layer_by_gds(self, gds_layer: list[int]) -> dict:
        """
        Get device layer information by GDS layer specification.

        Args:
            gds_layer (list[int]): GDS layer specification as [layer_number, datatype].

        Returns:
            dict: Device layer information, or None if not found.
        """
        for device in self.device:
            if device["layer"] == gds_layer:
                return device
        return None

    def __repr__(self) -> str:
        """String representation of the technology object."""
        return f"technology(name='{self.name}', devices={len(self.device)} layers)"
