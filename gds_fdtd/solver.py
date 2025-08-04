"""
gds_fdtd simulation toolbox.

FDTD solver module.
@author: Mustafa Hammood, 2025
"""
import os
from pathlib import Path
from gds_fdtd.core import component, port, technology
from gds_fdtd.sparams import sparameters
from abc import abstractmethod


class fdtd_port:
    """
    Represents a port in an FDTD simulation.

    Attributes:
        name (str): Name identifier for the port.
        position (list of float): A 3-element list specifying the port's (x, y, z) location.
        span (list of float|None): A 3-element list representing the span of the port.
        direction (str): The direction of the port. Must be either 'forward' or 'backward'.
        modes (list of int): List of mode indices (must be non-empty).
    """

    def __init__(
        self,
        name: str = "opt1",
        position: list[float] = [0.0, 0.0, 0.0],
        span: list[float | None] = [None, 2.5, 1.5],
        direction: str = "forward",
        modes: list[int] = [0],
    ):
        """
        Initialize an FDTD port with specified parameters.

        Parameters:
            name (str): Name for the port.
            position (list of float): Port position as [x, y, z]. Must have exactly 3 elements.
            span (list of float|None): Port span as a list of 3 values.
            direction (str): Direction of the port, either 'forward' or 'backward'.
            modes (list of int): List of mode indices (non-empty).
        """
        self.name = name
        self.position = position
        self.span = span
        self.direction = direction
        self.modes = modes

        if len(position) != 3:
            raise ValueError("Position must be a list of 3 floats")
        if len(span) != 3:
            raise ValueError("Span must be a list of 3 floats")
        if direction not in ["forward", "backward"]:
            raise ValueError("Direction must be either 'forward' or 'backward'")
        if len(modes) == 0:
            raise ValueError("Modes must be a list of integers")


class fdtd_field_monitor:
    """
    Represents a field monitor in an FDTD simulation.
    """

    def __init__(self, name: str, monitor_type: str):
        self.name = name
        self.monitor_type = monitor_type


    def visualize(self):
        """Visualize the field monitor."""
        raise NotImplementedError("Visualization not implemented for this field monitor type")


class fdtd_solver:
    """
    FDTD solver for electromagnetic simulations.

    Attributes:
        component (component): Component to simulate.
        tech (technology): Technology to use for the simulation.
        port_input (list of port): Input ports of the component.
        wavelength_start (float): Starting wavelength for simulation.
        wavelength_end (float): Ending wavelength for simulation.
        wavelength_points (int): Number of sampled wavelength points.
        mesh (int): Mesh resolution.
        boundary (list of str): Boundary conditions for the simulation domain.
        symmetry (list of int): Symmetry configuration.
        z_min (float): Minimum z coordinate of the simulation domain.
        z_max (float): Maximum z coordinate of the simulation domain.
        width_ports (float): Width of the ports in their injection direction.
        depth_ports (float): Depth of the ports in the z direction.
        buffer (float): Buffer region beyond the component's ports.
        modes (list of int): List of mode indices (non-empty).
        mode_freq_pts (int): Number of frequency points for mode calculation.
        run_time_factor (float): Multiplier for simulation runtime.
        field_monitors (list of str): List of field monitoring directions.
        working_dir (str): Base directory where a component-specific subdirectory will be created for FDTD project files.
        component_working_dir (str): Same as working_dir, the component-specific directory path.
        fdtd_ports (list of fdtd_port): List of converted FDTD port objects.

    """

    def __init__(
        self,
        component: component,
        tech: technology,
        port_input: list[port | None] = [None],
        wavelength_start: float = 1.5,
        wavelength_end: float = 1.6,
        wavelength_points: int = 100,
        mesh: int = 10,
        boundary: list[str] = ["PML", "PML", "PML"],
        symmetry: list[int] = [0, 0, 0],
        z_min: float = -1.0,
        z_max: float = 1.0,
        width_ports: float = 2.0,
        depth_ports: float = 1.5,
        buffer: float = 1.0,
        modes: list[int] = [1],
        mode_freq_pts: int = 3,
        run_time_factor: float = 3,
        field_monitors: list[str] = ["z"],
        working_dir: str = "./",
    ):
        """
        Initialize the FDTD solver with simulation configuration.

        Parameters:
            component (component): Component to simulate.
            tech (technology): Technology to use for the simulation.
            port_input (list of port): Input ports of the component.
            wavelength_start (float): Simulation start wavelength in micrometers.
            wavelength_end (float): Simulation end wavelength in micrometers.
            wavelength_points (int): Number of sample points between wavelengths.
            mesh (int): Mesh grid resolution.
            boundary (list of str): Boundary condition types for each dimension.
            symmetry (list of int): Symmetry settings for each axis.
            z_min (float): Minimum z coordinate of the simulation domain.
            z_max (float): Maximum z coordinate of the simulation domain.
            width_ports (float): Width of the ports in their injection direction.
            depth_ports (float): Depth of the ports in their injection direction.
            buffer (float): Buffer region beyond the component's ports.
            modes (list of int): List of mode indices (non-empty).
            mode_freq_pts (int): Number of frequency points for mode calculation.
            run_time_factor (float): Factor to adjust simulation runtime.
            field_monitors (list of fdtd_field_monitor): Field monitors.
            working_dir (str): Base directory where a component-specific subdirectory will be created for FDTD project files.
        """
        self.component = component
        self.tech = tech
        self.port_input = port_input if port_input else component.ports[0]
        self.wavelength_start = wavelength_start
        self.wavelength_end = wavelength_end
        self.wavelength_points = wavelength_points
        self.mesh = mesh
        self.boundary = boundary
        self.symmetry = symmetry
        self.z_min = z_min
        self.z_max = z_max
        self.width_ports = width_ports
        self.depth_ports = depth_ports
        self.buffer = buffer
        self.modes = modes
        self.mode_freq_pts = mode_freq_pts
        self.run_time_factor = run_time_factor
        self.field_monitors = field_monitors
        self.working_dir = working_dir

        # Create component-specific working directory under the base working directory
        self.component_working_dir = os.path.join(self.working_dir, self.component.name)
        Path(self.component_working_dir).mkdir(parents=True, exist_ok=True)
        print(f"FDTD working directory: {os.path.abspath(self.component_working_dir)}")
        
        # Update working_dir to point to the component-specific directory for file operations
        self.working_dir = self.component_working_dir

        # Auto-calculate center and span from component geometry
        self._calculate_simulation_domain()

        # Convert component ports to fdtd_port objects for modular solver implementation
        self.fdtd_ports: list[fdtd_port] = self._convert_component_ports_to_fdtd_ports()

        self.field_monitors_objs = []
        self._sparameters = None

    def _export_gds(self):
        """Export the component GDS to the working directory."""
        self._gds_filename = f"{self.component.name}.gds"
        self._gds_filepath = os.path.join(self.working_dir, self._gds_filename)
        self.component.export_gds(export_dir=self.working_dir, buffer=2 * self.buffer)


    def _calculate_simulation_domain(self):
        """Calculate the simulation domain center and span from the component geometry."""
        # This is a placeholder implementation - you'll need to adjust based on your component structure
        # Assuming component has a bounding box method or similar geometry information
        try:
            c = self.component
            self.center = [
                c.bounds.x_center,
                c.bounds.y_center,
                (self.z_max + self.z_min) / 2,
            ]
            self.span = [
                c.bounds.x_span + 2 * self.buffer,
                c.bounds.y_span + 2 * self.buffer,
                self.z_max - self.z_min,
            ]
        except AttributeError:    
            # Fallback to default values if component doesn't have bbox
            self.center = [0.0, 0.0, (self.z_max + self.z_min) / 2]
            self.span = [5.0, 5.0, self.z_max - self.z_min]
            print(
                "Warning: Could not determine component geometry, using default simulation domain"
            )

    def _convert_component_ports_to_fdtd_ports(self) -> list[fdtd_port]:
        """
        Convert component ports to fdtd_port objects for modular solver implementation.

        This method creates a solver-agnostic representation of ports that can be used
        by different FDTD solver implementations (Lumerical, Tidy3D, etc.).

        Ports are sorted by their index (extracted from port names) to ensure consistent ordering.

        Returns:
            list[fdtd_port]: List of fdtd_port objects with standardized attributes, sorted by port index.
        """
        fdtd_ports = []

        # Sort ports by their index to ensure consistent ordering
        # This uses the port.idx property which extracts the numeric part from port names
        sorted_ports = sorted(self.component.ports, key=lambda p: p.idx)

        for p in sorted_ports:
            # Map component port direction (degrees) to FDTD injection configuration
            if p.direction in [90, 270]:
                # Port facing north (90°) or south (270°) - injection along y-axis
                direction = "backward" if p.direction == 90 else "forward"
                span = [
                    self.width_ports,
                    None,
                    self.depth_ports,
                ]  # x_span, y_span=None (injection axis), z_span

            elif p.direction in [180, 0]:
                # Port facing west (180°) or east (0°) - injection along x-axis
                direction = "forward" if p.direction == 180 else "backward"
                span = [
                    None,
                    self.width_ports,
                    self.depth_ports,
                ]  # x_span=None (injection axis), y_span, z_span

            else:
                raise ValueError(
                    f"Port direction {p.direction}° not supported. Supported directions: 0°, 90°, 180°, 270°"
                )

            # Create standardized fdtd_port object
            fdtd_port_obj = fdtd_port(
                name=p.name,
                position=[p.center[0], p.center[1], p.center[2]],
                span=span,
                direction=direction,
                modes=self.modes,  # Use solver's mode configuration
            )

            fdtd_ports.append(fdtd_port_obj)

        return fdtd_ports

    def _get_active_ports(self) -> list[fdtd_port]:
        """Get the active ports from the component ports."""
        # Determine which ports should be active
        # active_ports should be a list of component port objects (type: port from component.ports)
        if self.port_input is None:
            # Default: activate all ports for full S-parameter matrix
            active_port_names = [fdtd_port.name for fdtd_port in self.fdtd_ports]
        elif isinstance(self.port_input, list):
            # List of component port objects
            active_port_names = []
            for component_port in self.port_input:
                if hasattr(component_port, "name"):
                    active_port_names.append(component_port.name)
                else:
                    raise ValueError(
                        f"Invalid port object in active_ports list: {component_port}"
                    )
        else:
            # Single component port object (user fed in 1 active port)
            if hasattr(self.port_input, "name"):
                active_port_names = [self.port_input.name]
            else:
                raise ValueError(
                    f"Invalid single port object: {self.port_input}. Expected component port object with 'name' attribute."
                )
        return active_port_names

    @property
    def sparameters(self) -> sparameters:
        """Get the S-parameters results."""
        if self._sparameters is None:
            print("S-parameters results not available. Please run the simulation first.")
        return self._sparameters

    # below are abstract methods that must be implemented by the solver
    @abstractmethod
    def setup(self) -> None:
        """Setup the simulation."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the simulation."""
        pass

    @abstractmethod
    def get_resources(self) -> None:
        """Get the resources used by the simulation."""
        pass

    @abstractmethod
    def get_results(self) -> None:
        """Get the results of the simulation."""
        pass

    @abstractmethod
    def get_log(self) -> None:
        """Get the log of the simulation."""
        pass

