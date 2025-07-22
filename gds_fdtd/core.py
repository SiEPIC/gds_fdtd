"""
gds_fdtd simulation toolbox.

Core objects module.
@author: Mustafa Hammood, 2025
"""

import pya
import logging
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import yaml

c0_um = 299792458000000.0  # speed of light in um/s

def is_point_inside_polygon(
    point: list[float, float], polygon_points: list[list[float, float]]
) -> bool:
    """Test if a point inside a polygon using Shapely.

    Args:
        point (list): Point for test [x, y]
        polygon_points (list): List of points defining a polygon [[x1, y1], [x2,y2], ..]

    Returns:
        bool: Test result.
    """

    # Create a Shapely Point object for the given coordinate
    point = Point(point)

    # Create a Shapely Polygon object from the list of polygon points
    polygon = Polygon(polygon_points)

    # Check if the point is inside the polygon
    return point.within(polygon) or polygon.touches(point)


class layout:
    def __init__(self, name: str, ly: pya.Layout, cell: pya.Cell):
        self.name = name
        self.ly = ly
        self.cell = cell

    @property
    def dbu(self) -> float:
        return self.ly.dbu


def calculate_polygon_extension(
    center: list[float, float], width: float, direction: float, buffer: float = 1.0
) -> list[list[float, float]]:
    """
    Calculate the polygon extension for a port.

    Args:
        center (list[float, float]): Center of the port [x, y]. Convention is in microns.
        width (float): Width of the port. Convention is in microns.
        direction (float): Direction of the port in degrees. Convention is in degrees.
        buffer (float): Buffer distance from the port. Convention is in microns.

    Returns:
        list[list[float, float]]: Polygon extension
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


class port:
    """
    Represents an optical port object in a component.

    A port defines a connection point with properties like position, width, and direction.

    Attributes:
        name (str): Name of the port, typically containing a numeric identifier.
        center (list[float, float, float]): 3D coordinates of the port center [x, y, z]. Convention is in microns.
        width (float): Width of the port. Convention is in microns.
        direction (float): Direction of the port. Convention is in degrees. Directions supported are 0, 90, 180, 270.
        height (float): Height of the port, assigned during component initialization. Convention is in microns.
        material (str): Material of the port, assigned during component initialization.
        layer (list[int]): GDS layer information [layer_number, datatype], assigned during component initialization.
    """

    def __init__(
        self,
        name: str,
        center: list[float, float, float],
        width: float,
        direction: float,
    ):
        """
        Initialize a port object.

        Args:
            name (str): Name of the port, typically containing a numeric identifier.
            center (list[float, float, float]): 3D coordinates of the port center [x, y, z].
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
        self.height = None
        self.material = None
        self.layer = None

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
        Extract the index of the port from its name.

        The index is extracted by taking the digits in the name in reverse order.
        For example, "port42" would yield an index of 24.

        Returns:
            int: The extracted port index.
        """
        return int("".join(char for char in reversed(self.name) if char.isdigit()))

    def polygon_extension(self, buffer: float = 1.0) -> list[list[float, float]]:
        """
        Calculate the polygon extension for this port.

        This creates a rectangular polygon extending from the port in the direction
        specified by the port's direction attribute.

        Args:
            buffer (float, optional): Buffer distance from the port in microns. Defaults to 4.0.

        Returns:
            list[list[float, float]]: Polygon extension as a list of [x,y] coordinates.
        """
        return calculate_polygon_extension(
            self.center, self.width, self.direction, buffer
        )


class structure:
    """
    Represents a physical structure in the component with geometric and material properties.

    This class defines a 3D structure with a 2D polygon base extruded vertically,
    including material properties and sidewall angle for fabrication realism.
    """

    def __init__(
        self,
        name: str,
        polygon: list[list[float, float]],
        z_base: float,
        z_span: float,
        material: str,
        sidewall_angle: float = 90.0,
        layer: list[int] = [1, 0],
    ):
        """
        Initialize a structure with geometric and material properties.

        Args:
            name (str): Unique identifier for the structure.
            polygon (list[list[float, float]]): 2D polygon defining the structure's horizontal cross-section,
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
        self.layer = layer


class region:
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



def initialize_ports_z(ports: list["port"], structures: list["structure"]) -> None:
    """
    Initialize the z-coordinate, height, material, and layer of ports based on their position within structures.
    
    This function checks if each port is located within any structure and sets the port's z-coordinate,
    height, material, and layer accordingly.
    
    Args:
        ports (list["port"]): List of port objects to initialize.
        structures (list["structure"]): List of structure objects to check against.
        
    Returns:
        None
    """
    # iterate through each port
    for p in ports:
        # check if port location is within any structure
        for s in structures:
            # TODO: hack: if s is a list then it's not a box/clad region, find a better way to identify this..
            if type(s) == list:
                for poly in s:
                    if is_point_inside_polygon(p.center[:2], poly.polygon):
                        p.center[2] = s[0].z_base + s[0].z_span / 2
                        p.height = s[0].z_span
                        p.material = s[0].material
                        p.layer = s[0].layer  # Store the layer information from the structure
            else:
                # Handle individual structures (not in a list)
                if is_point_inside_polygon(p.center[:2], s.polygon):
                    p.center[2] = s.z_base + s.z_span / 2
                    p.height = s.z_span
                    p.material = s.material
                    p.layer = s.layer  # Store the layer information from the structure
        if p.height == None:
            logging.warning(f"Cannot find height for port {p.name}")
    return

class component:
    """
    A component consisting of structures, ports, and boundaries.
    
    This class represents a complete photonic component that can be simulated
    or exported to GDS format.
    """
    
    def __init__(self, name: str, structures: list[structure], ports: list[port], bounds: list[region]):
        """
        Initialize a photonic component.
        
        Args:
            name: The name of the component.
            structures: List of structures (geometries) in the component.
            ports: List of ports for input/output connections.
            bounds: Boundaries of the component.
        """
        self.name = name
        self.structures = structures
        self.ports = ports
        self.bounds = bounds
        initialize_ports_z(ports = self.ports, structures = self.structures)  # initialize ports z center and z span

    def export_gds(self, export_dir: str = None, dbu: float = 0.001, layer: list = [1, 0], buffer: float = 1.0) -> None:
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

        layout = pya.Layout()
        layout.dbu = dbu  # Set the database unit to 0.001 um
        top_cell = layout.create_cell(self.name)

        # Dictionary to store created layers to avoid duplicates
        created_layers = {}

        # Export component structures using their individual layer information
        for structure_group in self.structures:
            if isinstance(structure_group, list):
                for structure in structure_group:
                    # Get or create the layer for this structure
                    layer_key = tuple(structure.layer)
                    if layer_key not in created_layers:
                        layer_info = pya.LayerInfo(structure.layer[0], structure.layer[1])
                        created_layers[layer_key] = layout.layer(layer_info)
                    
                    structure_layer_idx = created_layers[layer_key]
                    
                    # Create and insert the polygon
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


class s_parameters:
    def __init__(self, entries=None):
        if entries is None:
            self._entries = []
        else:
            self._entries = entries
        return

    @property
    def S(self):
        return dict(zip([i.label for i in self._entries], self._entries))

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
        if input_entries == None:
            input_entries = self._entries

        for s in input_entries:
            if s.idx_in == idx_in and s.idx_out == idx_out:
                entries.append(s)
        if len(entries) == 1:
            entries = entries[0]
        return entries

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Wavelength [microns]")
        ax.set_ylabel("Transmission [dB]")
        for i in self._entries:
            logging.info("Mode amplitudes in each port: \n")
            mag = [10 * np.log10(abs(i) ** 2) for i in i.s]
            phase = [np.angle(i) ** 2 for i in i.s]
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
        ax.plot((c0_um) / np.array(self.freq), 10 * np.log10(self.s**2))
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
    def from_yaml(cls, file_path: str) -> 'technology':
        """
        Create a technology object from a YAML file.
        
        Args:
            file_path (str): Path to the YAML technology file.
            
        Returns:
            technology: Technology object parsed from the YAML file.
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        tech_data = data.get("technology", {})
        tech = cls(name=tech_data.get("name", "Unknown"))
        
        # Parse substrate layer
        substrate = tech_data.get("substrate", {})
        if substrate:
            tech.substrate.append({
                "z_base": substrate.get("z_base"),
                "z_span": substrate.get("z_span"),
                "material": substrate.get("material"),
            })

        # Parse superstrate layer
        superstrate = tech_data.get("superstrate", {})
        if superstrate:
            tech.superstrate.append({
                "z_base": superstrate.get("z_base"),
                "z_span": superstrate.get("z_span"),
                "material": superstrate.get("material"),
            })

        # Parse pinrec layers
        tech.pinrec = [
            {"layer": list(pinrec.get("layer"))}
            for pinrec in tech_data.get("pinrec", [])
        ]

        # Parse devrec layers
        tech.devrec = [
            {"layer": list(devrec.get("layer"))}
            for devrec in tech_data.get("devrec", [])
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
    
    def add_device_layer(self, layer: list[int], z_base: float, z_span: float, 
                        material: dict, sidewall_angle: float = 90.0) -> None:
        """
        Add a device layer to the technology.
        
        Args:
            layer (list[int]): GDS layer specification as [layer_number, datatype].
            z_base (float): Base z-coordinate of the layer.
            z_span (float): Thickness of the layer.
            material (dict): Material properties dictionary.
            sidewall_angle (float, optional): Sidewall angle in degrees. Defaults to 90.0.
        """
        self.device.append({
            "layer": layer,
            "z_base": z_base,
            "z_span": z_span,
            "material": material,
            "sidewall_angle": sidewall_angle,
        })
    
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
