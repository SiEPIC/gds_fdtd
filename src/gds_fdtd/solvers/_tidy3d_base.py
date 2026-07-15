"""
gds_fdtd simulation toolbox.

Internal base for the Tidy3D scene-building engine: the solver-agnostic
geometry/port/domain setup (``_TidyEngineBase`` + ``_TidyPort``) that the
Tidy3D engine builds on. Kept separate from the tidy3d-specific engine so this
setup logic stays importable and testable WITHOUT tidy3d installed. Not a
public module.
@author: Mustafa Hammood, 2025
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from gds_fdtd.errors import JobValidationError
from gds_fdtd.geometry import Component, Port
from gds_fdtd.logging_config import (
    log_dict,
    log_separator,
    setup_logging,
)
from gds_fdtd.spec import SimulationSpec

if TYPE_CHECKING:
    from gds_fdtd.technology import Technology


class _TidyPort:
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
        position: list[float] | None = None,
        span: list[float | None] | None = None,
        direction: str = "forward",
        modes: list[int] | None = None,
    ):
        """
        Initialize an FDTD port with specified parameters.

        Parameters:
            name (str): Name for the port.
            position (list of float): Port position as [x, y, z]. Must have exactly 3 elements.
                Defaults to [0.0, 0.0, 0.0].
            span (list of float|None): Port span as a list of 3 values.
                Defaults to [None, 2.5, 1.5].
            direction (str): Direction of the port, either 'forward' or 'backward'.
            modes (list of int): List of mode indices (non-empty). Defaults to [0].
        """
        self.name = name
        self.position = list(position) if position is not None else [0.0, 0.0, 0.0]
        self.span = list(span) if span is not None else [None, 2.5, 1.5]
        self.direction = direction
        self.modes = list(modes) if modes is not None else [0]
        position = self.position
        span = self.span
        modes = self.modes

        if len(position) != 3:
            raise JobValidationError("Position must be a list of 3 floats")
        if len(span) != 3:
            raise JobValidationError("Span must be a list of 3 floats")
        if direction not in ["forward", "backward"]:
            raise JobValidationError("Direction must be either 'forward' or 'backward'")
        if len(modes) == 0:
            raise JobValidationError("Modes must be a list of integers")


class _TidyEngineBase:
    """
    FDTD solver for electromagnetic simulations.

    Attributes:
        component (component): Component to simulate.
        tech (Technology): Technology to use for the simulation.
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
        tidy_ports (list of _TidyPort): List of converted FDTD port objects.

    """

    def __init__(
        self,
        component: Component,
        tech: Technology | None,
        port_input: Port | list[Port] | None = None,
        wavelength_start: float = 1.5,
        wavelength_end: float = 1.6,
        wavelength_points: int = 100,
        mesh: int = 10,
        boundary: list[str] | None = None,
        symmetry: list[int] | None = None,
        z_min: float = -1.0,
        z_max: float = 1.0,
        width_ports: float = 2.0,
        depth_ports: float = 1.5,
        buffer: float = 1.0,
        modes: list[int] | None = None,
        mode_freq_pts: int = 3,
        run_time_factor: float = 3,
        field_monitors: list[str] | None = None,
        working_dir: str = "./",
    ):
        """
        Initialize the FDTD solver with simulation configuration.

        Parameters:
            component (component): Component to simulate.
            tech (Technology): Technology to use for the simulation.
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
        # Normalize port_input to a list of component ports.
        # None => all ports are active (full S-matrix). The old default ([None])
        # was truthy, so the documented "first port" fallback was unreachable and
        # _get_active_ports crashed on the default (bug B5).
        if port_input is None:
            self.port_input = list(component.ports)
        elif isinstance(port_input, list):
            self.port_input = list(port_input)
        else:
            self.port_input = [port_input]
        for p in self.port_input:
            if not hasattr(p, "name"):
                raise JobValidationError(
                    f"Invalid port object in port_input: {p!r}. Expected component "
                    "port objects (or None for all ports)."
                )
        # all numeric settings are validated through one SimulationSpec;
        # the legacy attribute surface below is kept identical (mutable lists).
        self.spec = SimulationSpec(
            wavelength_start=wavelength_start,
            wavelength_end=wavelength_end,
            wavelength_points=wavelength_points,
            mesh=mesh,
            boundary=tuple(boundary) if boundary is not None else ("PML", "PML", "PML"),
            symmetry=tuple(symmetry) if symmetry is not None else (0, 0, 0),
            z_min=z_min,
            z_max=z_max,
            width_ports=width_ports,
            depth_ports=depth_ports,
            buffer=buffer,
            modes=tuple(modes) if modes is not None else (1,),
            mode_freq_pts=mode_freq_pts,
            run_time_factor=run_time_factor,
            field_monitors=tuple(field_monitors) if field_monitors is not None else ("z",),
        )
        self.wavelength_start = self.spec.wavelength_start
        self.wavelength_end = self.spec.wavelength_end
        self.wavelength_points = self.spec.wavelength_points
        self.mesh = self.spec.mesh
        self.boundary = list(self.spec.boundary)
        self.symmetry = list(self.spec.symmetry)
        self.z_min = self.spec.z_min
        self.z_max = self.spec.z_max
        self.width_ports = self.spec.width_ports
        self.depth_ports = self.spec.depth_ports
        self.buffer = self.spec.buffer
        self.modes = list(self.spec.modes)
        self.mode_freq_pts = self.spec.mode_freq_pts
        self.run_time_factor = self.spec.run_time_factor
        self.field_monitors = list(self.spec.field_monitors)
        self.working_dir = working_dir

        # Create component-specific working directory under the base working directory
        self.component_working_dir = os.path.join(self.working_dir, self.component.name)
        Path(self.component_working_dir).mkdir(parents=True, exist_ok=True)

        # Update working_dir to point to the component-specific directory for file operations
        self.working_dir = self.component_working_dir

        # Setup logging for this solver instance
        self.logger = setup_logging(self.working_dir, self.component.name)
        self.logger.info(f"FDTD working directory: {os.path.abspath(self.component_working_dir)}")

        # Log solver initialization
        solver_config = {
            "solver_type": self.__class__.__name__,
            "component": self.component.name,
            "wavelength_range": f"{wavelength_start} - {wavelength_end} um",
            "wavelength_points": wavelength_points,
            "mesh": mesh,
            "modes": modes,
            "field_monitors": field_monitors,
        }
        log_dict(self.logger, solver_config, "Solver Configuration")

        # Auto-calculate center and span from component geometry
        self._calculate_simulation_domain()

        # Convert component ports to _TidyPort objects for modular solver implementation
        self.tidy_ports: list[_TidyPort] = self._build_tidy_ports()

        # Log field monitor objects creation
        if self.field_monitors:
            self.logger.debug(f"Field monitors requested: {self.field_monitors}")

    def _export_gds(self) -> None:
        """Export the component GDS to the working directory."""
        self._gds_filename = f"{self.component.name}.gds"
        self._gds_filepath = os.path.join(self.working_dir, self._gds_filename)
        self.component.export_gds(export_dir=self.working_dir, buffer=2 * self.buffer)

    def _calculate_simulation_domain(self) -> None:
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
            self.logger.info(
                "Warning: Could not determine component geometry, using default simulation domain"
            )

    def _build_tidy_ports(self) -> list[_TidyPort]:
        """
        Convert component ports to _TidyPort objects for modular solver implementation.

        This method creates a solver-agnostic representation of ports that can be used
        by different FDTD solver implementations (Lumerical, Tidy3D, etc.).

        Ports are sorted by their index (extracted from port names) to ensure consistent ordering.

        Returns:
            list[_TidyPort]: List of _TidyPort objects with standardized attributes, sorted by port index.
        """
        ports = []

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
                raise JobValidationError(
                    f"Port direction {p.direction}° not supported. Supported directions: 0°, 90°, 180°, 270°"
                )

            # Create standardized _TidyPort object
            tidy_port = _TidyPort(
                name=p.name,
                position=[p.center[0], p.center[1], p.center[2]],
                span=span,
                direction=direction,
                modes=self.modes,  # Use solver's mode configuration
            )

            ports.append(tidy_port)

        return ports

    def _get_active_ports(self) -> list[str]:
        """Names of the ports to excite (port_input is normalized to a list in __init__)."""
        return [p.name for p in self.port_input]

    def _validate_simulation_parameters(self) -> None:
        """Re-validate the (possibly mutated) legacy attributes through SimulationSpec.

        The hand-written checks moved into gds_fdtd.spec.SimulationSpec
        validators. Rebuilding the spec here keeps setup()-time validation
        meaningful for callers that mutated the legacy list attributes after
        construction, and refreshes self.spec to match.
        """
        self.logger.info("Validating simulation parameters")
        try:
            self.spec = SimulationSpec(
                wavelength_start=self.wavelength_start,
                wavelength_end=self.wavelength_end,
                wavelength_points=self.wavelength_points,
                mesh=self.mesh,
                boundary=tuple(self.boundary),
                symmetry=tuple(self.symmetry),
                z_min=self.z_min,
                z_max=self.z_max,
                width_ports=self.width_ports,
                depth_ports=self.depth_ports,
                buffer=self.buffer,
                modes=tuple(self.modes),
                mode_freq_pts=self.mode_freq_pts,
                run_time_factor=self.run_time_factor,
                field_monitors=tuple(self.field_monitors),
            )
        except Exception as e:
            self.logger.error(str(e))
            raise JobValidationError(str(e)) from e
        self.logger.info("Simulation parameters validated successfully")

    def _calculate_simulation_time(
        self, max_dimension: float, max_group_index: float = 4.5
    ) -> float:
        """Calculate appropriate simulation time based on geometry and materials.

        Args:
            max_dimension: Maximum dimension of the simulation domain in meters
            max_group_index: Maximum group index of materials in the simulation

        Returns:
            Simulation time in seconds
        """
        c = 299792458  # speed of light in m/s
        v = c / max_group_index  # velocity of pulse in the medium
        time_span = self.run_time_factor * max_dimension / v
        return time_span

    def _print_simulation_summary(self) -> None:
        """Print and log a summary of the simulation configuration."""
        log_separator(self.logger, "FDTD SIMULATION SUMMARY")

        # Log detailed configuration
        summary_data = {
            "Component": self.component.name,
            "Technology": getattr(self.tech, "name", "Custom"),
            "Solver type": self.__class__.__name__,
            "Working directory": self.working_dir,
            "Wavelength range": f"{self.wavelength_start} - {self.wavelength_end} μm",
            "Wavelength points": self.wavelength_points,
            "Simulation domain": f"{self.span[0]:.1f} × {self.span[1]:.1f} × {self.span[2]:.1f} μm",
            "Domain center": f"({self.center[0]:.1f}, {self.center[1]:.1f}, {self.center[2]:.1f}) μm",
            "Mesh resolution": f"{self.mesh} cells/wavelength",
            "Run time factor": self.run_time_factor,
            "Total ports": len(self.tidy_ports),
            "Active ports": len(self._get_active_ports()),
            "Port dimensions": f"{self.width_ports} × {self.depth_ports} μm",
            "Modes per port": self.modes,
            "Boundaries": self.boundary,
            "Symmetry": self.symmetry,
        }

        log_dict(self.logger, summary_data, "Simulation Configuration")

        # Console output (formatted for readability)
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FDTD Simulation Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Component: {self.component.name}")
        self.logger.info(f"Technology: {getattr(self.tech, 'name', 'Custom')}")
        self.logger.info(f"Solver type: {self.__class__.__name__}")
        self.logger.info(f"Working directory: {self.working_dir}")
        self.logger.info("Simulation Parameters:")
        self.logger.info(f"  Wavelength range: {self.wavelength_start} - {self.wavelength_end} μm")
        self.logger.info(f"  Wavelength points: {self.wavelength_points}")
        self.logger.info(
            f"  Simulation domain: {self.span[0]:.1f} × {self.span[1]:.1f} × {self.span[2]:.1f} μm"
        )
        self.logger.info(
            f"  Domain center: ({self.center[0]:.1f}, {self.center[1]:.1f}, {self.center[2]:.1f}) μm"
        )
        self.logger.info(f"  Mesh resolution: {self.mesh} cells/wavelength")
        self.logger.info(f"  Run time factor: {self.run_time_factor}")
        self.logger.info("Port Configuration:")
        self.logger.info(f"  Total ports: {len(self.tidy_ports)}")
        self.logger.info(f"  Active ports: {len(self._get_active_ports())}")
        self.logger.info(f"  Port dimensions: {self.width_ports} × {self.depth_ports} μm")
        self.logger.info(f"  Modes per port: {self.modes}")
        self.logger.info("Boundary Conditions:")
        self.logger.info(f"  Boundaries: {self.boundary}")
        self.logger.info(f"  Symmetry: {self.symmetry}")
        self.logger.info("=" * 60 + "\n")
