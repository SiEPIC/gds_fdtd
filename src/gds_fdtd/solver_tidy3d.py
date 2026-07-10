"""
gds_fdtd simulation toolbox.

Tidy3D FDTD solver interface module.
@author: Mustafa Hammood, 2025
"""

import os

import numpy as np
import tidy3d as td
from tidy3d.plugins.smatrix import ModalComponentModeler, Port

from gds_fdtd.logging_config import log_simulation_complete, log_simulation_start
from gds_fdtd.solver import fdtd_solver
from gds_fdtd.sparams import sparameters


class fdtd_solver_tidy3d(fdtd_solver):
    """
    FDTD solver for electromagnetic simulations using Tidy3D.
    """

    def __init__(self, *args, visualize: bool = True, **kwargs):
        """Initialize the Tidy3D solver by calling the parent constructor."""
        super().__init__(*args, **kwargs)
        self.visualize = visualize
        self.simulation = None
        self.setup()

    def setup(self) -> None:
        """Setup the Tidy3D simulation using ComponentModeler for S-matrix calculation."""
        self.logger.info("Starting Tidy3D solver setup")

        # Validate simulation parameters
        self._validate_simulation_parameters()

        # Export GDS with port extensions to working directory
        self._export_gds()
        self.logger.info(f"GDS exported to: {self._gds_filepath}")

        # Calculate frequencies for S-matrix calculation
        self.freqs = td.C_0 / np.linspace(
            self.wavelength_start, self.wavelength_end, self.wavelength_points
        )
        self.lda0 = (self.wavelength_end + self.wavelength_start) / 2
        self.freq0 = td.C_0 / self.lda0
        self.logger.debug(
            f"Frequency calculation: {len(self.freqs)} points from {self.freqs[0]:.2e} to {self.freqs[-1]:.2e} Hz"
        )

        # Create base simulation and ports for S-matrix calculation
        self.base_simulation = self._create_base_simulation()
        self.smatrix_ports = self._create_smatrix_ports()
        self.logger.info(f"Created base simulation with {len(self.smatrix_ports)} ports")

        # Create the modeler for S-matrix calculation.
        # tidy3d >=2.9 renamed ComponentModeler -> ModalComponentModeler and
        # moved web options (verbose/path_dir) to tidy3d.web.run.
        self.component_modeler = ModalComponentModeler(
            simulation=self.base_simulation,
            ports=self.smatrix_ports,
            freqs=self.freqs,
        )
        self.logger.info("ModalComponentModeler created successfully")

        # Print setup summary
        self._print_simulation_summary()
        total_mode_combinations = len(self.smatrix_ports) * len(self.modes)
        setup_info = f"Tidy3D solver setup complete with ComponentModeler: {len(self.smatrix_ports)} ports × {len(self.modes)} modes = {total_mode_combinations} mode combinations"
        self.logger.info(setup_info)

        self.logger.info("Tidy3D solver setup complete with ComponentModeler:")
        self.logger.info(
            f"  • {len(self.smatrix_ports)} ports × {len(self.modes)} modes = {total_mode_combinations} mode combinations"
        )
        self.logger.info("  • Multi-modal S-matrix calculation ready")
        self.logger.info(
            f"  • ComponentModeler will auto-generate task names for {self.component.name}"
        )

    def _create_base_simulation(self):
        """Create base simulation without sources/monitors for ComponentModeler."""
        device = self.component

        # Create structures
        structures = self._create_structures()

        # Add field monitor if requested
        monitors = []
        if self.field_monitors:
            for field_monitor_axis in self.field_monitors:
                field_monitor = self._create_field_monitor(
                    device, freqs=self.freqs, axis=field_monitor_axis
                )
                monitors.append(field_monitor)
                self.logger.debug(f"Created Tidy3D field monitor: {field_monitor.name}")

        # simulation domain size (in microns)
        sim_size = [self.span[0], self.span[1], self.span[2]]

        # Run time from the shared base-class helper (includes the group-index
        # slowdown factor) so run_time_factor means the same physics as in the
        # Lumerical solver (bug B11). max dimension converted um -> m.
        run_time = self._calculate_simulation_time(max(sim_size) * 1e-6)

        # Honor the user's boundary settings (previously hardcoded to PML on
        # all sides while the Lumerical solver obeyed them; bug B10).
        boundary_spec = self._create_boundary_spec()

        # Create base simulation (no sources - ComponentModeler adds them)
        base_sim = td.Simulation(
            size=sim_size,
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=self.mesh, wavelength=self.lda0),
            structures=structures,
            sources=[],  # No sources - ComponentModeler will add them
            monitors=monitors,
            run_time=run_time,
            boundary_spec=boundary_spec,
            center=(self.center[0], self.center[1], self.center[2]),
            symmetry=tuple(self.symmetry),
        )

        return base_sim

    def _create_boundary_spec(self) -> "td.BoundarySpec":
        """Map the solver's boundary strings to a tidy3d BoundarySpec.

        Supported names (case-insensitive, matching the Lumerical adapter):
        "PML", "Metal" (perfect electric conductor), "Periodic".
        """
        mapping = {
            "pml": td.PML,
            "metal": td.PECBoundary,
            "periodic": td.Periodic,
        }
        axis_boundaries = []
        for axis_name, boundary_name in zip("xyz", self.boundary, strict=True):
            key = str(boundary_name).lower()
            if key not in mapping:
                raise ValueError(
                    f"Unsupported boundary {boundary_name!r} for axis {axis_name}. "
                    f"Supported: {sorted(mapping)}"
                )
            b = mapping[key]()
            axis_boundaries.append(td.Boundary(minus=b, plus=b))
        return td.BoundarySpec(x=axis_boundaries[0], y=axis_boundaries[1], z=axis_boundaries[2])

    def _create_smatrix_ports(self):
        """Create Tidy3D Port objects for S-matrix calculation with multi-modal support."""
        ports = []

        for fdtd_port in self.fdtd_ports:
            # Determine port direction and size based on fdtd_port configuration
            if fdtd_port.span[0] is None:  # x-axis injection
                direction = "+" if fdtd_port.direction == "forward" else "-"
                size = [0, self.width_ports, self.depth_ports]
            elif fdtd_port.span[1] is None:  # y-axis injection
                direction = "+" if fdtd_port.direction == "forward" else "-"
                size = [self.width_ports, 0, self.depth_ports]
            else:
                raise ValueError(f"Invalid span configuration for port {fdtd_port.name}")

            # Create Tidy3D Port object with multi-modal support
            port = Port(
                center=fdtd_port.position,
                size=size,
                direction=direction,
                name=fdtd_port.name,
                mode_spec=td.ModeSpec(
                    num_modes=max(self.modes)
                ),  # Ensure enough modes are calculated
            )
            ports.append(port)

        return ports

    def _background_polygon(self):
        """Rectangle flush with the port extensions, CENTERED ON THE COMPONENT.

        Previously this square was centered at the origin (sized via
        abs(center)+span/2), so off-origin components got an oversized,
        origin-anchored substrate/superstrate (bug B9).
        """
        b = self.component.bounds
        half_x = b.x_span / 2 + 2 * self.buffer
        half_y = b.y_span / 2 + 2 * self.buffer
        cx, cy = b.x_center, b.y_center
        return [
            (cx - half_x, cy - half_y),
            (cx + half_x, cy - half_y),
            (cx + half_x, cy + half_y),
            (cx - half_x, cy + half_y),
        ]

    @staticmethod
    def _is_background(structure) -> bool:
        """Background = substrate/superstrate, by ROLE, with a
        name-sniffing fallback only for td.Structure objects in tests."""
        role = getattr(structure, "role", None)
        if role is not None:
            return role in ("substrate", "superstrate")
        n = structure.name.lower()
        # name fallback (td.Structure has no role); tolerate the historical
        # "Subtrate" typo present in components built by older versions
        return "substrate" in n or "superstrate" in n or "subtrate" in n

    def _to_td_structure(self, s):
        """Convert one core.structure into a td.Structure."""
        if s.z_span < 0:
            bounds = (s.z_base + s.z_span, s.z_base)
        else:
            bounds = (s.z_base, s.z_base + s.z_span)

        polygon = self._background_polygon() if self._is_background(s) else s.polygon

        return td.Structure(
            geometry=td.PolySlab(
                vertices=polygon,
                slab_bounds=bounds,
                axis=2,
                sidewall_angle=(90 - s.sidewall_angle) * (np.pi / 180),
            ),
            medium=s.material["tidy3d"] if isinstance(s.material, dict) else s.material,
            name=s.name,
        )

    def _create_structures(self):
        """Create Tidy3D structure objects from the component."""
        device = self.component

        structures = [self._to_td_structure(s) for s in device.structures]

        # extend ports beyond sim region with 2*buffer
        for p in device.ports:
            structures.append(
                td.Structure(
                    geometry=td.PolySlab(
                        vertices=p.polygon_extension(buffer=2 * self.buffer),
                        slab_bounds=(
                            p.center[2] - p.height / 2,
                            p.center[2] + p.height / 2,
                        ),
                        axis=2,
                        sidewall_angle=(90 - device.structures[0].sidewall_angle) * (np.pi / 180),
                    ),
                    medium=p.material["tidy3d"] if isinstance(p.material, dict) else p.material,
                    name=f"port_{p.name}",
                )
            )
        return structures

    def _create_field_monitor(self, device, freqs=2e14, axis="z", z_center=None):
        """Create a field monitor for the specified axis."""
        # identify a device field z_center if None
        if z_center is None:
            # per-LAYER average (matching the old per-list semantics)
            z_by_layer = {}
            for s in device.structures:
                if s.role == "device":
                    z_by_layer.setdefault(tuple(s.layer), s.z_base + s.z_span / 2)
            z_center = np.average(list(z_by_layer.values()))
        # center the monitor on the component, not the origin (bug B9)
        cx, cy = device.bounds.x_center, device.bounds.y_center
        if axis == "z":
            center = [cx, cy, z_center]
            size = [td.inf, td.inf, 0]
        elif axis == "y":
            center = [cx, cy, z_center]
            size = [td.inf, 0, td.inf]
        elif axis == "x":
            center = [cx, cy, z_center]
            size = [0, td.inf, td.inf]
        else:
            raise ValueError(
                f"Invalid axis {axis!r} for field monitor. Valid selections are 'x', 'y', 'z'."
            )
        return td.FieldMonitor(
            center=center,
            size=size,
            freqs=freqs,
            name=f"{axis}_field",
        )

    def get_resources(self) -> None:
        """Get the resources used by the simulation."""
        if not hasattr(self, "component_modeler"):
            self.logger.info("No ComponentModeler available.")
            return

        total_simulations = len(self.smatrix_ports) * len(self.modes)
        self.logger.info("ComponentModeler Multi-Modal Configuration:")
        self.logger.info(f"  • {len(self.smatrix_ports)} ports")
        self.logger.info(f"  • {len(self.modes)} modes per port: {self.modes}")
        self.logger.info(f"  • Total simulations required: {total_simulations}")
        self.logger.info(f"  • Component: {self.component.name}")
        self.logger.info("Resource estimation handled by Tidy3D cloud platform")

    def run(self) -> None:
        """Run the simulation using ComponentModeler."""
        if not hasattr(self, "component_modeler"):
            error_msg = "No ComponentModeler created. Call setup() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        log_simulation_start(self.logger, "Tidy3D ComponentModeler", self.component.name)
        self.logger.info("Running S-matrix calculation with ComponentModeler...")

        # Run the S-matrix calculation through the tidy3d web API (2.11 workflow).
        # tidy3d.web is a LAZILY imported submodule: `import tidy3d as td` does
        # NOT provide the web attribute — import it explicitly (finding F10; the
        # original validation scripts masked this by importing tidy3d.web
        # themselves).
        import tidy3d.web as web

        try:
            self._modeler_data = web.run(
                self.component_modeler,
                task_name=f"gdsfdtd_{self.component.name}",
                path=os.path.join(self.working_dir, f"{self.component.name}_modeler.hdf5"),
                verbose=True,
            )
            smatrix_result = self._modeler_data.smatrix()
            self.logger.info("ModalComponentModeler simulation completed successfully")
        except Exception as e:
            self.logger.error(f"ModalComponentModeler simulation failed: {e}")
            raise

        # Convert results to sparameters format for interface compatibility
        self._convert_smatrix_to_sparameters(smatrix_result)

        log_simulation_complete(self.logger, "Tidy3D ComponentModeler")
        self.logger.info("S-matrix calculation completed successfully!")

    def _convert_smatrix_to_sparameters(self, smatrix_result):
        """Convert Tidy3D S-matrix results to sparameters format with multi-modal support."""
        # Initialize sparameters object
        self._sparameters = sparameters(self.component.name)

        # Extract wavelength information
        freqs = smatrix_result.coords["f"].values
        wavelengths = td.C_0 / freqs

        self.logger.info(f"Converting S-matrix results: {len(freqs)} frequency points")
        self.logger.info(f"Available ports: {list(smatrix_result.coords['port_in'].values)}")
        self.logger.info(f"Available modes: {list(smatrix_result.coords['mode_index_in'].values)}")

        # Process all S-matrix elements for multi-modal case
        for port_in in smatrix_result.coords["port_in"].values:
            for port_out in smatrix_result.coords["port_out"].values:
                for mode_in in smatrix_result.coords["mode_index_in"].values:
                    for mode_out in smatrix_result.coords["mode_index_out"].values:
                        # Only process modes that are in our requested mode list
                        # Convert Tidy3D 0-based to our 1-based mode indexing
                        mode_in_1based = mode_in + 1
                        mode_out_1based = mode_out + 1

                        if mode_in_1based not in self.modes or mode_out_1based not in self.modes:
                            continue

                        # Extract S-parameter data
                        s_data = smatrix_result.sel(
                            port_in=port_in,
                            port_out=port_out,
                            mode_index_in=mode_in,
                            mode_index_out=mode_out,
                        ).values

                        # Convert complex S-parameter to magnitude and phase
                        s_mag = np.abs(s_data)
                        s_phase = np.angle(s_data)

                        # Extract port numbers from port names
                        in_port_num = self._extract_port_number(port_in)
                        out_port_num = self._extract_port_number(port_out)

                        # Add to sparameters object with proper mode indexing
                        self._sparameters.add_data(
                            in_port=str(in_port_num),
                            out_port=str(out_port_num),
                            mode_label=1,
                            in_modeid=mode_in_1based,  # Use 1-based indexing for interface compatibility
                            out_modeid=mode_out_1based,  # Use 1-based indexing for interface compatibility
                            data_type="transmission",
                            group_delay=0.0,
                            f=list(freqs),
                            s_mag=list(s_mag),
                            s_phase=list(s_phase),
                        )

                        self.logger.info(
                            f"Added S-parameter: Port {in_port_num}(mode {mode_in_1based}) -> Port {out_port_num}(mode {mode_out_1based})"
                        )

        # Store the raw Tidy3D results for field visualization
        self.smatrix_result = smatrix_result

        self.logger.info(
            f"Multi-modal S-matrix conversion complete: {len(self._sparameters.data)} S-parameter entries"
        )

    def _extract_port_number(self, port_name):
        """Extract port number from port name."""
        # Find port in component ports by name and return its index
        for port in self.component.ports:
            if port.name == port_name:
                return port.idx
        # Fallback: extract digits from port name
        digits = [int(i) for i in port_name if i.isdigit()]
        return digits[0] if digits else 1

    def get_results(self) -> None:
        """Get the results of the simulation."""
        if not hasattr(self, "_sparameters") or self._sparameters is None:
            self.logger.info("No results available. Run simulation first.")
            return
        # Results are already stored in self._sparameters by _convert_smatrix_to_sparameters

    def get_log(self) -> str:
        """Get the log of the simulation."""
        try:
            data = getattr(self, "_modeler_data", None)
            if data is None:
                return "Log not available - run() has not completed."
            logs = []
            sim_data_map = getattr(data, "data", None) or {}
            items = sim_data_map.items() if hasattr(sim_data_map, "items") else []
            for task_name, sim_data in items:
                log = getattr(sim_data, "log", None)
                if log:
                    logs.append(f"=== {task_name} ===\n{log}")
            return "\n".join(logs) if logs else "No per-task logs exposed by this tidy3d version."
        except Exception as e:
            self.logger.info(f"Error retrieving Tidy3D log: {e}")
            return f"Log retrieval error: {str(e)}"

    def visualize_field_monitors(self, freq=None, axis: str = "z", savefig: str | None = None):
        """Plot the run's field profile.

        Uses the per-task SimulationData from the 2.11 modeler results.
        """
        data = getattr(self, "_modeler_data", None)
        if data is None:
            self.logger.info("No field data: run() has not completed.")
            return None
        from gds_fdtd.solvers.tidy3d import plot_tidy3d_fields

        return plot_tidy3d_fields(data, axis=axis, savefig=savefig)
