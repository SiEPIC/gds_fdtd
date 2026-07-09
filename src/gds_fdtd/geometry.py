"""
gds_fdtd simulation toolbox.

Geometry primitives: Port, Structure, Region, Component (WP2.1 — renamed from
the lowercase classes previously in core.py; gds_fdtd.core re-exports the old
names with a DeprecationWarning).
@author: Mustafa Hammood, 2025
"""

import logging
import re

import pya
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

c0_um = 299792458000000.0  # speed of light in um/s


def is_point_inside_polygon(point: list[float], polygon_points: list[list[float]]) -> bool:
    """Test if a point inside a polygon using Shapely.

    Args:
        point (list): Point for test [x, y]
        polygon_points (list): List of points defining a polygon [[x1, y1], [x2,y2], ..]

    Returns:
        bool: Test result.
    """

    # Create a Shapely Point object for the given coordinate
    pt = Point(point)

    # Create a Shapely Polygon object from the list of polygon points
    polygon = Polygon(polygon_points)

    # Check if the point is inside the polygon
    return bool(pt.within(polygon) or polygon.touches(pt))


class LayoutSource:
    def __init__(self, name: str, ly: pya.Layout, cell: pya.Cell):
        self.name = name
        self.ly = ly
        self.cell = cell

    @property
    def dbu(self) -> float:
        return float(self.ly.dbu)


def calculate_polygon_extension(
    center: list[float], width: float, direction: float, buffer: float = 4.0
) -> list[list[float]]:
    """
    Calculate the polygon extension for a port.

    Args:
        center (list[float]): Center of the port [x, y]. Convention is in microns.
        width (float): Width of the port. Convention is in microns.
        direction (float): Direction of the port in degrees. Convention is in degrees.
        buffer (float): Buffer distance from the port. Convention is in microns.

    Returns:
        list[list[float]]: Polygon extension
    """
    if direction == 0:
        return [
            [center[0], center[1] + width / 2],
            [center[0] + buffer, center[1] + width / 2],
            [center[0] + buffer, center[1] - width / 2],
            [center[0], center[1] - width / 2],
        ]
    elif direction == 180:
        return [
            [center[0], center[1] + width / 2],
            [center[0] - buffer, center[1] + width / 2],
            [center[0] - buffer, center[1] - width / 2],
            [center[0], center[1] - width / 2],
        ]
    elif direction == 90:
        return [
            [center[0] - width / 2, center[1]],
            [center[0] - width / 2, center[1] + buffer],
            [center[0] + width / 2, center[1] + buffer],
            [center[0] + width / 2, center[1]],
        ]
    elif direction == 270:
        return [
            [center[0] - width / 2, center[1]],
            [center[0] - width / 2, center[1] - buffer],
            [center[0] + width / 2, center[1] - buffer],
            [center[0] + width / 2, center[1]],
        ]
    raise ValueError(f"Invalid direction: {direction}. Supported: 0, 90, 180, 270.")


class Port:
    """
    Represents an optical port object in a component.

    A port defines a connection point with properties like position, width, and direction.

    Attributes:
        name (str): Name of the port, typically containing a numeric identifier.
        center (list[float]): 3D coordinates of the port center [x, y, z]. Convention is in microns.
        width (float): Width of the port. Convention is in microns.
        direction (float): Direction of the port. Convention is in degrees. Directions supported are 0, 90, 180, 270.
        height (float): Height of the port, assigned during component initialization. Convention is in microns.
        material (str): Material of the port, assigned during component initialization.
        layer (list[int]): GDS layer information [layer_number, datatype], assigned during component initialization.
    """

    def __init__(
        self,
        name: str,
        center: list[float],
        width: float,
        direction: float,
    ):
        """
        Initialize a port object.

        Args:
            name (str): Name of the port, typically containing a numeric identifier.
            center (list[float]): 3D coordinates of the port center [x, y, z].
            width (float): Width of the port in microns.
            direction (float): Direction of the port in degrees.
        """
        self.name = name
        self.center = center
        self.width = width
        self.direction = direction
        # initialize height, material, and layer as None
        # will be assigned upon component __init__
        # TODO: feels like a better way to do this..
        self.height: float | None = None
        self.material: object = None
        self.layer: list[int] | None = None

        if self.direction not in [0, 90, 180, 270]:
            raise ValueError(
                f"Invalid direction: {self.direction}. Supported directions are 0, 90, 180, 270."
            )

    @property
    def x(self) -> float:
        """
        Get the x-coordinate of the port center.

        Returns:
            float: x-coordinate in microns.
        """
        return self.center[0]

    @property
    def y(self) -> float:
        """
        Get the y-coordinate of the port center.

        Returns:
            float: y-coordinate in microns.
        """
        return self.center[1]

    @property
    def z(self) -> float:
        """
        Get the z-coordinate of the port center.

        Returns:
            float: z-coordinate in microns.
        """
        return self.center[2]

    @property
    def idx(self) -> int:
        """
        Extract the index of the port from its trailing digits.

        "opt1" -> 1, "port42" -> 42, "opt10" -> 10 (previously the digits were
        reversed, so "port42" -> 24 and "opt10" collided with "opt1").

        Returns:
            int: The extracted port index.

        Raises:
            ValueError: If the port name has no trailing digits.
        """
        m = re.search(r"(\d+)$", self.name)
        if m is None:
            raise ValueError(
                f"Port name {self.name!r} has no trailing digits to derive an index from."
            )
        return int(m.group(1))

    def polygon_extension(self, buffer: float = 4.0) -> list[list[float]]:
        """
        Calculate the polygon extension for this port.

        This creates a rectangular polygon extending from the port in the direction
        specified by the port's direction attribute.

        Args:
            buffer (float, optional): Buffer distance from the port in microns. Defaults to 4.0.

        Returns:
            list[list[float]]: Polygon extension as a list of [x,y] coordinates.
        """
        return calculate_polygon_extension(self.center, self.width, self.direction, buffer)


class Structure:
    """
    Represents a physical structure in the component with geometric and material properties.

    This class defines a 3D structure with a 2D polygon base extruded vertically,
    including material properties and sidewall angle for fabrication realism.
    """

    def __init__(
        self,
        name: str,
        polygon: list[list[float]],
        z_base: float,
        z_span: float,
        material: str,
        sidewall_angle: float = 90.0,
        layer: list[int] | None = None,
        role: str = "device",
    ):
        """
        Initialize a structure with geometric and material properties.

        Args:
            name (str): Unique identifier for the structure.
            polygon (list[list[float]]): 2D polygon defining the structure's horizontal cross-section,
                                               formatted as [[x1,y1], [x2,y2], ...].
            z_base (float): Base z-coordinate in microns where the structure begins.
            z_span (float): Vertical height/thickness of the structure in microns.
            material (str): Material identifier for the structure.
            sidewall_angle (float, optional): Angle of the sidewalls in degrees, where 90.0 means vertical walls.
                                             Defaults to 90.0.
            layer (list[int], optional): GDS layer specification as [layer_number, datatype]. Defaults to [1, 0].
        """
        self.name = name
        self.polygon = polygon  # polygon should be in the form of list of list of 2 pts, i.e. [[0,0],[0,1],[1,1]]
        self.z_base = z_base
        self.z_span = z_span
        self.material = material
        self.sidewall_angle = sidewall_angle
        self.layer = list(layer) if layer is not None else [1, 0]
        if role not in ("device", "substrate", "superstrate"):
            raise ValueError(
                f"Invalid structure role {role!r}; expected 'device', 'substrate' or 'superstrate'."
            )
        self.role = role


class Region:
    """
    Represents a 3D region defined by a 2D polygon and vertical extent.

    This class defines a region with vertices in the x-y plane and a vertical
    extent defined by z_center and z_span.
    """

    def __init__(self, vertices: list[list[float]], z_center: float, z_span: float):
        """
        Initialize a region with vertices and vertical dimensions.

        Args:
            vertices (list[list[float]]): List of [x,y] coordinates defining the region's polygon.
            z_center (float): Center z-coordinate of the region in microns.
            z_span (float): Vertical extent/thickness of the region in microns.
        """
        self.vertices = vertices
        self.z_center = z_center
        self.z_span = z_span

    @property
    def x(self) -> list[float]:
        """
        Get all x-coordinates of the vertices.

        Returns:
            list[float]: List of x-coordinates.
        """
        return [i[0] for i in self.vertices]

    @property
    def y(self) -> list[float]:
        """
        Get all y-coordinates of the vertices.

        Returns:
            list[float]: List of y-coordinates.
        """
        return [i[1] for i in self.vertices]

    @property
    def x_span(self) -> float:
        """
        Calculate the span (width) of the region in the x-direction.

        Returns:
            float: Width of the region in microns.
        """
        return abs(min(self.x) - max(self.x))

    @property
    def y_span(self) -> float:
        """
        Calculate the span (height) of the region in the y-direction.

        Returns:
            float: Height of the region in microns.
        """
        return abs(min(self.y) - max(self.y))

    @property
    def x_center(self) -> float:
        """
        Calculate the center x-coordinate of the region.

        Returns:
            float: Center x-coordinate in microns.
        """
        return (min(self.x) + max(self.x)) / 2

    @property
    def y_center(self) -> float:
        """
        Calculate the center y-coordinate of the region.

        Returns:
            float: Center y-coordinate in microns.
        """
        return (min(self.y) + max(self.y)) / 2

    @property
    def x_min(self) -> float:
        """
        Get the minimum x-coordinate of the region.

        Returns:
            float: Minimum x-coordinate in microns.
        """
        return min(self.x)

    @property
    def x_max(self) -> float:
        """
        Get the maximum x-coordinate of the region.

        Returns:
            float: Maximum x-coordinate in microns.
        """
        return max(self.x)

    @property
    def y_min(self) -> float:
        """
        Get the minimum y-coordinate of the region.

        Returns:
            float: Minimum y-coordinate in microns.
        """
        return min(self.y)

    @property
    def y_max(self) -> float:
        """
        Get the maximum y-coordinate of the region.

        Returns:
            float: Maximum y-coordinate in microns.
        """
        return max(self.y)


def initialize_ports_z(ports: list["Port"], structures: list["Structure"]) -> None:
    """
    Initialize each port's z-center, height, material, and layer from the DEVICE
    structure containing it (last match wins, as before).

    WP2.3: structures are a flat list discriminated by Structure.role; the old
    list-nesting convention (type(s) == list) is gone. Substrate/superstrate no
    longer silently claim ports that sit outside every device structure — those
    now warn and keep height=None.
    """
    for p in ports:
        for s in structures:
            if s.role != "device":
                continue
            if is_point_inside_polygon(p.center[:2], s.polygon):
                p.center[2] = s.z_base + s.z_span / 2
                p.height = s.z_span
                p.material = s.material
                p.layer = s.layer
        if p.height is None:
            logging.warning(f"Cannot find height for port {p.name}")
    return


class Component:
    """
    A component consisting of structures, ports, and boundaries.

    This class represents a complete photonic component that can be simulated
    or exported to GDS format.
    """

    def __init__(
        self, name: str, structures: list[Structure], ports: list[Port], bounds: list[Region]
    ):
        """
        Initialize a photonic component.

        Args:
            name: The name of the component.
            structures: List of structures (geometries) in the component.
            ports: List of ports for input/output connections.
            bounds: Boundaries of the component.
        """
        self.name = name
        flat: list[Structure] = []
        for entry in structures:
            if isinstance(entry, list):  # legacy nested convention (pre-WP2.3)
                import warnings

                warnings.warn(
                    "Passing nested lists in Component.structures is deprecated; "
                    "pass a flat list of Structure objects with role= set "
                    "(flattened automatically, removal at v1.0).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                flat.extend(entry)
            else:
                flat.append(entry)
        self.structures = flat
        self.ports = ports
        self.bounds = bounds
        initialize_ports_z(
            ports=self.ports, structures=self.structures
        )  # initialize ports z center and z span

    def export_gds(
        self,
        export_dir: str | None = None,
        dbu: float = 0.001,
        layer: list[int] | None = None,
        buffer: float = 1.0,
    ) -> None:
        """
        Export the component to a GDS file.

        Args:
            export_dir: Directory to export the GDS file to. Defaults to current working directory.
            dbu: Database unit in microns. Defaults to 0.001 (1 nm).
            layer: GDS layer specification as [layer_number, datatype]. Used as fallback for port extensions if port has no layer info. Defaults to [1, 0].
            buffer: Buffer distance to extend ports beyond their bounds in microns. Defaults to 1.0.
        """
        import os

        import klayout.db as pya

        if layer is None:
            layer = [1, 0]

        layout = pya.Layout()
        layout.dbu = dbu  # Set the database unit to 0.001 um
        top_cell = layout.create_cell(self.name)

        # Dictionary to store created layers to avoid duplicates
        created_layers = {}

        # Export DEVICE structures using their individual layer information
        # (substrate/superstrate are background media, not layout geometry —
        # the old nested-only loop skipped them too)
        for structure in self.structures:
            if structure.role != "device":
                continue
            layer_key = tuple(structure.layer)
            if layer_key not in created_layers:
                layer_info = pya.LayerInfo(structure.layer[0], structure.layer[1])
                created_layers[layer_key] = layout.layer(layer_info)

            structure_layer_idx = created_layers[layer_key]

            pya_polygon = pya.Polygon(
                [
                    pya.Point(int(point[0] / layout.dbu), int(point[1] / layout.dbu))
                    for point in structure.polygon
                ]
            )
            top_cell.shapes(structure_layer_idx).insert(pya_polygon)

        # Export port extensions if buffer > 0 (using each port's layer information)
        if buffer > 0.0:
            for port in self.ports:
                # Use port's layer information if available, otherwise fallback to the layer parameter
                port_layer = port.layer if port.layer is not None else layer

                # Get or create the layer for this port extension
                port_layer_key = tuple(port_layer)
                if port_layer_key not in created_layers:
                    layer_info = pya.LayerInfo(port_layer[0], port_layer[1])
                    created_layers[port_layer_key] = layout.layer(layer_info)

                port_layer_idx = created_layers[port_layer_key]

                # Get the port extension polygon
                port_extension = port.polygon_extension(buffer=buffer)

                # Convert to KLayout polygon and add to the port's layer
                pya_port_polygon = pya.Polygon(
                    [
                        pya.Point(int(point[0] / layout.dbu), int(point[1] / layout.dbu))
                        for point in port_extension
                    ]
                )
                top_cell.shapes(port_layer_idx).insert(pya_port_polygon)

        if export_dir is None:
            export_dir = os.getcwd()
        layout.write(os.path.join(export_dir, f"{self.name}.gds"))
        return
