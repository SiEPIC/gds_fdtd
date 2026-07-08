"""
gds_fdtd simulation toolbox.

Tidy3D FDTD solver interface module.
@author: Mustafa Hammood, 2025
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
from tidy3d.plugins.smatrix import ComponentModeler, Port

from gds_fdtd.logging_config import log_simulation_complete, log_simulation_start
from gds_fdtd.solver import fdtd_field_monitor, fdtd_solver
from gds_fdtd.sparams import sparameters


class tidy3d_field_monitor(fdtd_field_monitor):
    """
    Tidy3D-specific field monitor with visualization capabilities.
    """

    def __init__(self, name: str, monitor_type: str, logger=None):
        super().__init__(name, monitor_type, logger)
        self.tidy3d_data = None

    def set_tidy3d_data(self, sim_data, freq_data=None):
        """
        Set Tidy3D-specific field data.

        Args:
            sim_data: Tidy3D simulation data containing field monitors
            freq_data: Frequency data for visualization
        """
        self.tidy3d_data = sim_data
        self.freq_data = freq_data

        # Extract field data for this monitor
        field_name = f"{self.monitor_type}_field"
        if hasattr(sim_data, field_name):
            self.field_data = getattr(sim_data, field_name)
            self.logger.debug(f"Tidy3D field data set for monitor {self.name}")
        else:
            self.logger.warning(f"Field monitor {field_name} not found in simulation data")

    def _create_field_plots(self, freq, field_component, figsize):
        """Create Tidy3D-specific field visualization plots."""
        if not self.has_data() or self.tidy3d_data is None:
            self.logger.error("No Tidy3D field data available for visualization")
            return

        field_name = f"{self.monitor_type}_field"

        try:
            # Use center frequency if not specified
            if freq is None:
                if hasattr(self.freq_data, "__len__") and len(self.freq_data) > 0:
                    freq = self.freq_data[len(self.freq_data) // 2]  # Center frequency
                else:
                    self.logger.warning("No frequency data available, using first available")

            fig, axes = plt.subplots(2, 2, figsize=figsize)
            fig.suptitle(
                f"Field Monitor: {self.name} ({self.monitor_type}-axis) at {freq / 1e12:.2f} THz"
            )

            # Plot field components
            field_components = ["Ex", "Ey", "Ez"] if field_component == "E" else [field_component]

            for i, component in enumerate(field_components[:3]):
                if i < 3:
                    ax = axes[i // 2, i % 2]
                    try:
                        self.tidy3d_data.plot_field(field_name, component, freq=freq, ax=ax)
                        ax.set_title(f"{component} field")
                    except Exception as e:
                        self.logger.warning(f"Could not plot {component}: {e}")
                        ax.text(
                            0.5,
                            0.5,
                            f"{component} field\n(Error: {str(e)})",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"{component} field")

            # Plot field magnitude
            ax = axes[1, 1]
            try:
                # Calculate field magnitude from components
                Ex = self.tidy3d_data[field_name].Ex.sel(f=freq, method="nearest")
                Ey = self.tidy3d_data[field_name].Ey.sel(f=freq, method="nearest")
                Ez = self.tidy3d_data[field_name].Ez.sel(f=freq, method="nearest")
                E_mag = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)

                # Simple plot of magnitude
                im = ax.imshow(np.real(E_mag), cmap="hot", origin="lower")
                ax.set_title("|E| magnitude")
                plt.colorbar(im, ax=ax)
            except Exception as e:
                self.logger.warning(f"Could not plot field magnitude: {e}")
                ax.text(
                    0.5,
                    0.5,
                    f"|E| magnitude\n(Error: {str(e)})",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("|E| magnitude")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Error creating Tidy3D field plots: {e}")
            super()._create_field_plots(freq, field_component, figsize)  # Fallback


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

        # Create ComponentModeler for S-matrix calculation
        self.component_modeler = ComponentModeler(
            simulation=self.base_simulation,
            ports=self.smatrix_ports,
            freqs=self.freqs,
            verbose=True,
            path_dir=self.working_dir,
        )
        self.logger.info("ComponentModeler created successfully")

        # Print setup summary
        self._print_simulation_summary()
        total_mode_combinations = len(self.smatrix_ports) * len(self.modes)
        setup_info = f"Tidy3D solver setup complete with ComponentModeler: {len(self.smatrix_ports)} ports × {len(self.modes)} modes = {total_mode_combinations} mode combinations"
        self.logger.info(setup_info)

        print("Tidy3D solver setup complete with ComponentModeler:")
        print(
            f"  • {len(self.smatrix_ports)} ports × {len(self.modes)} modes = {total_mode_combinations} mode combinations"
        )
        print("  • Multi-modal S-matrix calculation ready")
        print(f"  • ComponentModeler will auto-generate task names for {self.component.name}")

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

                # Create Tidy3D-specific field monitor object for visualization
                field_monitor_obj = tidy3d_field_monitor(
                    name=field_monitor.name, monitor_type=field_monitor_axis, logger=self.logger
                )
                if not hasattr(self, "field_monitors_objs"):
                    self.field_monitors_objs = []
                self.field_monitors_objs.append(field_monitor_obj)
                self.logger.debug(
                    f"Created field monitor object: {field_monitor.name} ({field_monitor_axis})"
                )

        # simulation domain size (in microns)
        sim_size = [self.span[0], self.span[1], self.span[2]]

        # run time calculation
        run_time = self.run_time_factor * max(sim_size) / td.C_0

        # Create boundary spec
        boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

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
            # Convert 1-based mode indices to 0-based for Tidy3D
            mode_indices = [m - 1 for m in self.modes]

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

    def _create_structures(self):
        """Create Tidy3D structure objects from the component."""
        device = self.component

        structures = []
        for s in device.structures:
            if type(s) == list:
                for i in s:
                    if i.z_span < 0:
                        bounds = (i.z_base + i.z_span, i.z_base)
                    else:
                        bounds = (i.z_base, i.z_base + i.z_span)

                    # Check if this is substrate/superstrate and extend to be flush with port extensions
                    if (
                        "substrate" in i.name.lower()
                        or "superstrate" in i.name.lower()
                        or "subtrate" in i.name.lower()
                    ):
                        # Extend substrate/superstrate to be flush with port extensions (2*buffer from component edge)
                        component_max_extent = max(
                            abs(device.bounds.x_center) + device.bounds.x_span / 2,
                            abs(device.bounds.y_center) + device.bounds.y_span / 2,
                        )
                        substrate_half_size = (
                            component_max_extent + 2 * self.buffer
                        )  # Flush with port extensions
                        extended_vertices = [
                            (-substrate_half_size, -substrate_half_size),
                            (substrate_half_size, -substrate_half_size),
                            (substrate_half_size, substrate_half_size),
                            (-substrate_half_size, substrate_half_size),
                        ]
                        polygon = extended_vertices
                    else:
                        polygon = i.polygon

                    structures.append(
                        td.Structure(
                            geometry=td.PolySlab(
                                vertices=polygon,
                                slab_bounds=bounds,
                                axis=2,
                                sidewall_angle=(90 - i.sidewall_angle) * (np.pi / 180),
                            ),
                            medium=i.material["tidy3d"]
                            if isinstance(i.material, dict)
                            else i.material,
                            name=i.name,
                        )
                    )
            else:
                if s.z_span < 0:
                    bounds = (s.z_base + s.z_span, s.z_base)
                else:
                    bounds = (s.z_base, s.z_base + s.z_span)

                # Check if this is substrate/superstrate and extend to be flush with port extensions
                if (
                    "substrate" in s.name.lower()
                    or "superstrate" in s.name.lower()
                    or "subtrate" in s.name.lower()
                ):
                    # Extend substrate/superstrate to be flush with port extensions (2*buffer from component edge)
                    component_max_extent = max(
                        abs(device.bounds.x_center) + device.bounds.x_span / 2,
                        abs(device.bounds.y_center) + device.bounds.y_span / 2,
                    )
                    substrate_half_size = (
                        component_max_extent + 2 * self.buffer
                    )  # Flush with port extensions
                    extended_vertices = [
                        (-substrate_half_size, -substrate_half_size),
                        (substrate_half_size, -substrate_half_size),
                        (substrate_half_size, substrate_half_size),
                        (-substrate_half_size, substrate_half_size),
                    ]
                    polygon = extended_vertices
                else:
                    polygon = s.polygon

                print(s.material["tidy3d"] if isinstance(s.material, dict) else s.material)
                structures.append(
                    td.Structure(
                        geometry=td.PolySlab(
                            vertices=polygon,
                            slab_bounds=bounds,
                            axis=2,
                            sidewall_angle=(90 - s.sidewall_angle) * (np.pi / 180),
                        ),
                        medium=s.material["tidy3d"] if isinstance(s.material, dict) else s.material,
                        name=s.name,
                    )
                )

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
            z_center = []
            for s in device.structures:
                if type(s) == list:  # i identify non sub/superstrate if s is a list
                    s = s[0]
                    z_center.append(s.z_base + s.z_span / 2)
            z_center = np.average(z_center)
        if axis == "z":
            center = [0, 0, z_center]
            size = [td.inf, td.inf, 0]
        elif axis == "y":
            center = [0, 0, z_center]
            size = [td.inf, 0, td.inf]
        elif axis == "x":
            center = [0, 0, z_center]
            size = [0, td.inf, td.inf]
        else:
            Exception("Invalid axis for field monitor. Valid selections are 'x', 'y', 'z'.")
        return td.FieldMonitor(
            center=center,
            size=size,
            freqs=freqs,
            name=f"{axis}_field",
        )

    def get_resources(self) -> None:
        """Get the resources used by the simulation."""
        if not hasattr(self, "component_modeler"):
            print("No ComponentModeler available.")
            return

        total_simulations = len(self.smatrix_ports) * len(self.modes)
        print("ComponentModeler Multi-Modal Configuration:")
        print(f"  • {len(self.smatrix_ports)} ports")
        print(f"  • {len(self.modes)} modes per port: {self.modes}")
        print(f"  • Total simulations required: {total_simulations}")
        print(f"  • Component: {self.component.name}")
        print("Resource estimation handled by Tidy3D cloud platform")

    def run(self) -> None:
        """Run the simulation using ComponentModeler."""
        if not hasattr(self, "component_modeler"):
            error_msg = "No ComponentModeler created. Call setup() first."
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        log_simulation_start(self.logger, "Tidy3D ComponentModeler", self.component.name)
        print("Running S-matrix calculation with ComponentModeler...")

        # Run the S-matrix calculation
        try:
            smatrix_result = self.component_modeler.run()
            self.logger.info("ComponentModeler simulation completed successfully")
        except Exception as e:
            self.logger.error(f"ComponentModeler simulation failed: {e}")
            raise

        # Convert results to sparameters format for interface compatibility
        self._convert_smatrix_to_sparameters(smatrix_result)

        # Set field data in field monitor objects if available
        self._set_field_data_in_monitors()

        log_simulation_complete(self.logger, "Tidy3D ComponentModeler")
        print("S-matrix calculation completed successfully!")

    def _set_field_data_in_monitors(self):
        """Set field data in field monitor objects from ComponentModeler results."""
        if not hasattr(self, "field_monitors_objs") or not self.field_monitors_objs:
            self.logger.debug("No field monitor objects to populate with data")
            return

        if not hasattr(self, "smatrix_result") or self.smatrix_result is None:
            self.logger.warning("No simulation results available for field monitors")
            return

        self.logger.info("Setting field data in monitor objects")

        # Access field data from ComponentModeler results
        # ComponentModeler may store results differently than direct simulation
        try:
            # Try to access individual simulation results from the batch
            # This depends on how ComponentModeler structures its results
            if hasattr(self.component_modeler, "batch_data") and self.component_modeler.batch_data:
                # Use the first simulation result that has field data
                for job_result in self.component_modeler.batch_data.values():
                    if hasattr(job_result, "simulation"):
                        # Found a simulation result with field data
                        for monitor_obj in self.field_monitors_objs:
                            monitor_obj.set_tidy3d_data(job_result, self.freqs)
                            self.logger.debug(f"Set field data for monitor: {monitor_obj.name}")
                        break
            else:
                self.logger.warning(
                    "ComponentModeler batch data not accessible for field visualization"
                )

        except Exception as e:
            self.logger.error(f"Error setting field data in monitors: {e}")
            self.logger.info("Field monitors will not have visualization data available")

    def _convert_smatrix_to_sparameters(self, smatrix_result):
        """Convert Tidy3D S-matrix results to sparameters format with multi-modal support."""
        # Initialize sparameters object
        self._sparameters = sparameters(self.component.name)

        # Extract wavelength information
        freqs = smatrix_result.coords["f"].values
        wavelengths = td.C_0 / freqs

        print(f"Converting S-matrix results: {len(freqs)} frequency points")
        print(f"Available ports: {list(smatrix_result.coords['port_in'].values)}")
        print(f"Available modes: {list(smatrix_result.coords['mode_index_in'].values)}")

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

                        print(
                            f"Added S-parameter: Port {in_port_num}(mode {mode_in_1based}) -> Port {out_port_num}(mode {mode_out_1based})"
                        )

        # Store the raw Tidy3D results for field visualization
        self.smatrix_result = smatrix_result

        print(
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
            print("No results available. Run simulation first.")
            return
        # Results are already stored in self._sparameters by _convert_smatrix_to_sparameters

    def get_log(self) -> str:
        """Get the log of the simulation."""
        try:
            if hasattr(self, "component_modeler") and self.component_modeler:
                # Try to get log from ComponentModeler's simulation data
                if hasattr(self.component_modeler, "sim_data") and self.component_modeler.sim_data:
                    log_text = str(self.component_modeler.sim_data.log)
                    return log_text
                else:
                    print("ComponentModeler simulation data not available for log extraction.")
                    return "Log not available - ComponentModeler simulation data not accessible."
            else:
                print("ComponentModeler not initialized.")
                return "Log not available - ComponentModeler not initialized."
        except Exception as e:
            print(f"Error retrieving Tidy3D log: {e}")
            return f"Log retrieval error: {str(e)}"

    def _extract_log_metrics(self, log_text: str) -> dict:
        """
        Extract Tidy3D-specific metrics from log text.

        Args:
            log_text: Raw Tidy3D log text

        Returns:
            dict: Extracted metrics specific to Tidy3D
        """
        log_metrics = super()._extract_log_metrics(log_text)

        try:
            # Grid points
            grid_match = re.search(r"Number of computational grid points: ([\d.e+-]+)", log_text)
            if grid_match:
                log_metrics["grid_points"] = grid_match.group(1)

            # Domain size
            domain_match = re.search(
                r"Simulation domain Nx, Ny, Nz: \[(\d+), (\d+), (\d+)\]", log_text
            )
            if domain_match:
                log_metrics["domain_nx"] = int(domain_match.group(1))
                log_metrics["domain_ny"] = int(domain_match.group(2))
                log_metrics["domain_nz"] = int(domain_match.group(3))

            # Time steps
            time_steps_match = re.search(r"Number of time steps: ([\d.e+-]+)", log_text)
            if time_steps_match:
                log_metrics["time_steps"] = time_steps_match.group(1)

            # Time step size
            time_step_match = re.search(r"Time step \(s\): ([\d.e+-]+)", log_text)
            if time_step_match:
                log_metrics["time_step_size"] = time_step_match.group(1)

            # Auto shutoff factor
            shutoff_match = re.search(r"Automatic shutoff factor: ([\d.e+-]+)", log_text)
            if shutoff_match:
                log_metrics["shutoff_factor"] = shutoff_match.group(1)

            # Solver times
            source_modes_time = re.search(r"Compute source modes time \(s\):\s*([\d.]+)", log_text)
            if source_modes_time:
                log_metrics["source_modes_time"] = float(source_modes_time.group(1))

            monitor_modes_time = re.search(
                r"Compute monitor modes time \(s\):\s*([\d.]+)", log_text
            )
            if monitor_modes_time:
                log_metrics["monitor_modes_time"] = float(monitor_modes_time.group(1))

            solver_time_match = re.search(r"Solver time \(s\):\s*([\d.]+)", log_text)
            if solver_time_match:
                log_metrics["solver_time"] = float(solver_time_match.group(1))

            post_processing_time = re.search(r"Post-processing time \(s\):\s*([\d.]+)", log_text)
            if post_processing_time:
                log_metrics["post_processing_time"] = float(post_processing_time.group(1))

            # Time-stepping performance
            time_stepping_speed = re.search(
                r"Time-stepping speed \(cells/s\):\s*([\d.e+]+)", log_text
            )
            if time_stepping_speed:
                log_metrics["time_stepping_speed"] = time_stepping_speed.group(1)

            # Field decay
            field_decay_matches = re.findall(r"field decay: ([\d.e+-]+)", log_text)
            if field_decay_matches:
                log_metrics["final_field_decay"] = field_decay_matches[-1]
                log_metrics["initial_field_decay"] = (
                    field_decay_matches[0]
                    if len(field_decay_matches) > 1
                    else field_decay_matches[0]
                )

            # FDTD solver details from the solver log section
            fdtd_setup_time = re.search(r"Solver setup time \(s\):\s*([\d.]+)", log_text)
            if fdtd_setup_time:
                log_metrics["fdtd_setup_time"] = float(fdtd_setup_time.group(1))

            fdtd_time_stepping = re.search(r"Time-stepping time \(s\):\s*([\d.]+)", log_text)
            if fdtd_time_stepping:
                log_metrics["fdtd_time_stepping"] = float(fdtd_time_stepping.group(1))

            data_write_time = re.search(r"Data write time \(s\):\s*([\d.]+)", log_text)
            if data_write_time:
                log_metrics["data_write_time"] = float(data_write_time.group(1))

            # Convergence information
            if "Field decay smaller than shutoff factor, exiting solver" in log_text:
                log_metrics["converged"] = True
            else:
                log_metrics["converged"] = False

            # Mode solver warnings
            mode_warnings = re.findall(
                r"WARNING: Mode '\d+' appears to undergo a discontinuous change", log_text
            )
            log_metrics["mode_warnings"] = len(mode_warnings)

        except Exception as e:
            self.logger.warning(f"Error extracting Tidy3D log metrics: {e}")

        return log_metrics

    def export_sparameters_dat(self, filepath: str = None):
        """Export S-parameters to an INTERCONNECT .dat file.

        Each entry is written with its actual port/mode ids and frequency
        vector (the previous writer assumed a dense n_ports² ordering and a
        regenerated frequency grid, mislabeling entries; bug B14 — and wrote
        power |S|² where the format stores magnitude).
        """
        if not hasattr(self, "_sparameters") or self._sparameters is None:
            print("No S-parameters available for export. Run simulation first.")
            return

        if filepath is None:
            filepath = os.path.join(self.working_dir, f"{self.component.name}_sparams.dat")

        from gds_fdtd.sparams import write_dat

        write_dat(self._sparameters, filepath)
        self.logger.info(f"S-parameters exported to: {filepath}")
        print(f"S-parameters exported to: {filepath}")

    def visualize_results(self):
        """Visualize the simulation results."""
        if not hasattr(self, "_sparameters") or self._sparameters is None:
            print("No results available for visualization.")
            return

        # Plot S-parameters
        self._sparameters.plot()

        # Export S-parameters to .dat file
        self.export_sparameters_dat()

    def visualize_field_monitors(self, freq=None):
        """Visualize field monitor data through field monitor objects."""
        self.logger.info("Starting field monitor visualization")

        if not self.visualize:
            self.logger.debug("Visualization disabled, skipping field monitor visualization")
            return

        # Use the base class method for modular field visualization
        self.visualize_all_field_monitors(freq=freq or self.freq0)
