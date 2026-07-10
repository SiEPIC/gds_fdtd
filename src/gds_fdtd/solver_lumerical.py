"""
gds_fdtd simulation toolbox.

Lumerical tools interface module.
@author: Mustafa Hammood, 2025
"""

import os
import warnings

from lumapi import FDTD

from gds_fdtd.logging_config import log_simulation_complete, log_simulation_start
from gds_fdtd.solver import fdtd_solver
from gds_fdtd.sparams import process_dat


class fdtd_solver_lumerical(fdtd_solver):
    """FDTD solver for electromagnetic simulations using Lumerical.

    .. deprecated:: 0.5
        Use :func:`gds_fdtd.solvers.get_solver` with ``"lumerical"``; removed in 1.0.
    """

    def __init__(self, *args, gpu: bool = False, **kwargs):
        """Initialize the Lumerical solver by calling the parent constructor.

        Args:
            gpu (bool): Whether to use GPU acceleration.
        """
        warnings.warn(
            "fdtd_solver_lumerical is deprecated since gds_fdtd 0.5 and will be "
            "removed in 1.0; use gds_fdtd.solvers.get_solver('lumerical')"
            "(component, technology, spec).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
        self.gpu = gpu
        self.setup()

    def setup(self) -> None:
        """Setup the Lumerical simulation."""
        self.logger.info("Starting Lumerical solver setup")

        # Validate simulation parameters
        self._validate_simulation_parameters()

        # Export GDS with port extensions to working directory
        self._export_gds()
        self.logger.info(f"GDS exported to: {self._gds_filepath}")

        # Initialize FDTD
        self.fdtd = FDTD()
        self.logger.info("Lumerical FDTD session initialized")

        # set the working directory
        self.fdtd.eval(f'cd("{self.working_dir}");')
        self.logger.debug(f"Lumerical working directory set to: {self.working_dir}")

        # Setup the layer builder with technology information
        self._setup_layer_builder()

        # Setup the FDTD simulation
        self._setup_fdtd()

        # Setup the field monitors
        self._setup_field_monitors()

        # Setup the s-parameters sweep
        self._setup_s_parameters_sweep()

        # Save the Lumerical FDTD project to working directory
        project_filepath = os.path.join(self.working_dir, f"{self.component.name}.fsp")
        self.fdtd.save(project_filepath)
        self.logger.info(f"FDTD project saved: {project_filepath}")
        self.logger.info(f"FDTD project saved: {project_filepath}")

        # Get the resources used by the simulation
        self.get_resources()

        # Print simulation summary
        self._print_simulation_summary()

    def get_resources(self) -> None:
        """Report estimated simulation resources, when the API provides them.

        Lumerical 2024 `runsystemcheck` returns Memory_Recommended /
        Total_FDTD_Yee_Nodes; **Lumerical 2025 (v252) returns an empty dict**
        (verified 2026-07-07, finding F6) — this must not crash setup.
        """
        self.fdtd.eval("reqs = runsystemcheck;")
        reqs = self.fdtd.getv("reqs") or {}

        memory = reqs.get("Memory_Recommended")  # Bytes (2024 API)
        nodes = reqs.get("Total_FDTD_Yee_Nodes")  # MNodes (2024 API)
        if memory is not None:
            self.logger.info(f"Memory: {memory / 1e9} GB")
        if nodes is not None:
            self.logger.info(f"Nodes: {nodes} MNodes")
        if memory is None and nodes is None:
            self.logger.info(
                "runsystemcheck reported no resource estimates (Lumerical 2025+ "
                "returns an empty result); continuing."
            )

    def run(self) -> None:
        """Run the simulation."""
        log_simulation_start(self.logger, "Lumerical FDTD", self.component.name)

        self._set_device_type("GPU" if self.gpu else "CPU")

        self.logger.info("Starting S-parameter sweep")
        try:
            self.fdtd.runsweep("sparams")
            self.logger.info("S-parameter sweep completed successfully")
        except Exception as e:
            self.logger.error(f"S-parameter sweep failed: {e}")
            raise

        self.get_results()
        log_simulation_complete(self.logger, "Lumerical FDTD")

    def _set_device_type(self, device: str) -> None:
        """Select CPU/GPU compute, handling the 2024->2025 API change.

        Lumerical 2025 (v252): setresource("FDTD", index, "device type", dev)
        (verified 2026-07-07, finding F7 — the 2024 two-arg form raises
        LumApiError with a migration hint). 2024: setresource("FDTD", dev, 1).
        Try the current syntax first, fall back to 2024.
        """
        try:
            self.fdtd.setresource("FDTD", 1, "device type", device)
            self.logger.info(f"{device} compute selected (Lumerical 2025+ syntax)")
        except Exception:
            self.fdtd.setresource("FDTD", device, 1)
            self.logger.info(f"{device} compute selected (Lumerical 2024 syntax)")

    def get_results(self) -> None:
        """Get the results of the simulation."""
        results_filepath = os.path.join(self.working_dir, f"{self.component.name}.dat")
        self.fdtd.exportsweep("sparams", results_filepath)
        sparams_sweep = self.fdtd.getsweepresult("sparams", "S parameters")

        self._sparameters = process_dat(results_filepath)

    def _setup_s_parameters_sweep(self) -> None:
        """
        Setup the s-parameters sweep with automatic port and mode configuration.

        This method:
        1. Creates an S-parameter sweep
        2. Automatically generates port-mode combinations based on fdtd_ports and modes
        3. Sets active only the ports that should be excited
        """

        active_port_names = self._get_active_ports()

        # Create S-parameter sweep
        self.fdtd.addsweep(3)  # 3 is s-parameter sweep
        self.fdtd.setsweep("s-parameter sweep", "name", "sparams")
        self.fdtd.setsweep("sparams", "Excite all ports", 0)  # We'll manually set active ports

        # Automatically generate indices based on sorted fdtd_ports and modes
        indices = []

        for fdtd_port in self.fdtd_ports:
            port_name = fdtd_port.name

            # Determine if this port should be active
            is_active = fdtd_port.name in active_port_names

            for mode_idx in self.modes:
                mode_name = f"mode {mode_idx}"  # Lumerical uses "mode 1", "mode 2", etc.

                indices.append(
                    {
                        "Port": port_name,
                        "Mode": mode_name,
                        "Active": 1 if is_active else 0,
                    }
                )

        # before adding entries to the sweep, we need to remove all existing entries
        while True:
            try:
                self.fdtd.removesweepparameter("sparams", 1)
            except Exception:
                self.logger.info("Done removing sweep parameters")
                break

        # Add all port-mode combinations to the sweep
        for idx in indices:
            self.fdtd.addsweepparameter("sparams", idx)

        # Print summary of S-parameter sweep configuration
        active_combinations = [idx for idx in indices if idx["Active"] == 1]
        total_combinations = len(indices)
        self.logger.info("S-parameter sweep configured:")
        self.logger.info(f"  Total port-mode combinations: {total_combinations}")
        self.logger.info(f"  Active combinations: {len(active_combinations)}")
        self.logger.info(f"  Active ports: {active_port_names}")
        self.logger.info(f"  Modes: {self.modes}")

        if len(active_combinations) > 0:
            self.logger.info("  Active combinations:")
            for combo in active_combinations:
                self.logger.info(f"    {combo['Port']} - {combo['Mode']}")
        else:
            self.logger.info("  Warning: No active port-mode combinations found!")

    def _setup_layer_builder(self) -> None:
        """
        Setup the Lumerical layer builder with technology information.

        """
        self.fdtd.addlayerbuilder()

        self.fdtd.setnamed("layer group", "x", self.center[0] * 1e-6)  # Convert to meters
        self.fdtd.setnamed("layer group", "y", self.center[1] * 1e-6)
        self.fdtd.setnamed("layer group", "z", 0)
        self.fdtd.setnamed("layer group", "x span", (self.span[0] + 2 * self.buffer) * 1e-6)
        self.fdtd.setnamed("layer group", "y span", (self.span[1] + 2 * self.buffer) * 1e-6)
        self.fdtd.setnamed(
            "layer group", "gds position reference", "Centered at custom coordinates"
        )
        self.fdtd.setnamed("layer group", "gds center x", -self.center[0] * 1e-6)
        self.fdtd.setnamed("layer group", "gds center y", -self.center[1] * 1e-6)

        # Load the GDS file into the layer builder
        self.fdtd.eval(f'loadgdsfile("{self._gds_filename}");')

        # Get list of layers that actually exist in the GDS file
        self.fdtd.eval("gds_layers = getlayerlist;")
        gds_layers_result = self.fdtd.getv("gds_layers")

        # Convert technology object to dict if needed
        if hasattr(self.tech, "to_dict"):
            tech_dict = self.tech.to_dict()
        else:
            tech_dict = self.tech

        # Helper function to check if a layer exists in GDS
        def layer_exists_in_gds(layer_spec):
            """Check if a layer specification exists in the loaded GDS file."""
            if gds_layers_result is None:
                return False
            # Convert layer spec to string format expected by Lumerical
            layer_str = f"{layer_spec[0]}:{layer_spec[1]}"
            return layer_str in str(gds_layers_result)

        # Set up substrate layer (always add as it's a background layer)
        if tech_dict["substrate"]:
            substrate = tech_dict["substrate"][0]
            self.fdtd.eval('addlayer("substrate");')

            # Set z start position and thickness
            z_start = substrate["z_base"] * 1e-6  # Convert to meters
            thickness = substrate["z_span"] * 1e-6  # Keep original sign for thickness

            self.fdtd.eval(f'setlayer("substrate", "start position", {z_start});')
            self.fdtd.eval(f'setlayer("substrate", "thickness", {-abs(thickness)});')

            # For negative thickness (downward growth), adjust z start position
            if substrate["z_span"] < 0:
                # For downward growth, set z start to be at the top of the layer
                z_start_adjusted = z_start  # z_base is already at the top for negative growth
                self.fdtd.eval(f'setlayer("substrate", "start position", {z_start_adjusted});')

            if substrate["material"] and "lum_db" in substrate["material"]:
                material_name = substrate["material"]["lum_db"]["model"]
                self.fdtd.eval(f'setlayer("substrate", "background material", "{material_name}");')
            # Substrate typically covers the entire simulation domain
            self.fdtd.eval('setlayer("substrate", "layer number", "");')  # No specific GDS layer

        # Set up superstrate layer (always add as it's a background layer)
        if tech_dict["superstrate"]:
            superstrate = tech_dict["superstrate"][0]
            self.fdtd.eval('addlayer("superstrate");')

            # Set z start position and thickness
            z_start = superstrate["z_base"] * 1e-6  # Convert to meters
            thickness = abs(superstrate["z_span"]) * 1e-6  # Always positive for superstrate

            self.fdtd.eval(f'setlayer("superstrate", "start position", {z_start});')
            self.fdtd.eval(f'setlayer("superstrate", "thickness", {thickness});')

            if superstrate["material"] and "lum_db" in superstrate["material"]:
                material_name = superstrate["material"]["lum_db"]["model"]
                self.fdtd.eval(
                    f'setlayer("superstrate", "background material", "{material_name}");'
                )
            # Superstrate typically covers the entire simulation domain
            self.fdtd.eval('setlayer("superstrate", "layer number", "");')  # No specific GDS layer

        # Set up device layers from technology - only if they exist in GDS
        layers_added = 0
        for idx, device_layer in enumerate(tech_dict["device"]):
            gds_layer = device_layer["layer"]

            # Only add layer if it exists in the GDS file
            if layer_exists_in_gds(gds_layer):
                layer_name = f"device_{idx}"

                # Add the layer
                self.fdtd.eval(f'addlayer("{layer_name}");')

                # Set z start position and thickness
                z_start = device_layer["z_base"] * 1e-6  # Convert to meters
                thickness = abs(device_layer["z_span"]) * 1e-6  # Always positive thickness

                self.fdtd.eval(f'setlayer("{layer_name}", "start position", {z_start});')
                self.fdtd.eval(f'setlayer("{layer_name}", "thickness", {thickness});')

                # Set GDS layer mapping
                layer_spec = f"{gds_layer[0]}:{gds_layer[1]}"
                self.fdtd.eval(f'setlayer("{layer_name}", "layer number", "{layer_spec}");')

                # Set the angle
                self.fdtd.eval(
                    f'setlayer("{layer_name}", "sidewall angle", {device_layer["sidewall_angle"]});'
                )

                # Set pattern material if available
                if device_layer["material"] and "lum_db" in device_layer["material"]:
                    material_name = device_layer["material"]["lum_db"]["model"]
                    self.fdtd.eval(
                        f'setlayer("{layer_name}", "pattern material", "{material_name}");'
                    )

                layers_added += 1
                self.logger.info(
                    f"Added layer {layer_name} for GDS layer {layer_spec} at z={device_layer['z_base']}μm, thickness={device_layer['z_span']}μm"
                )
            else:
                self.logger.info(
                    f"Skipping device layer {idx} (GDS layer {gds_layer[0]}:{gds_layer[1]}) - not found in GDS file"
                )

        self.logger.info(
            f"Layer builder setup complete with {layers_added} device layers (out of {len(tech_dict['device'])} defined in technology)"
        )
        if gds_layers_result is not None:
            self.logger.info(f"GDS file contains layers: {gds_layers_result}")

    def _setup_fdtd(self) -> None:
        """Setup the FDTD simulation."""
        self.fdtd.addfdtd()
        self.fdtd.setnamed("FDTD", "x", self.center[0] * 1e-6)
        self.fdtd.setnamed("FDTD", "y", self.center[1] * 1e-6)
        self.fdtd.setnamed("FDTD", "z", self.center[2] * 1e-6)
        self.fdtd.setnamed("FDTD", "x span", (self.span[0]) * 1e-6)
        self.fdtd.setnamed("FDTD", "y span", (self.span[1]) * 1e-6)
        self.fdtd.setnamed("FDTD", "z span", self.span[2] * 1e-6)

        # GPU: Lumerical 2024 uses 'express mode' here; 2025 switched to
        # setresource, handled by _set_device_type() at run() (finding F7).
        # Tolerate either vintage.
        if self.gpu:
            try:
                self.fdtd.setnamed("FDTD", "express mode", True)
            except Exception:
                self.logger.debug(
                    "'express mode' unavailable (Lumerical 2025+); using setresource at run()"
                )
        else:
            self.fdtd.setnamed("FDTD", "express mode", False)

        self._setup_wavelength()
        self._setup_boundary_symmetry()
        self._setup_mesh()
        self._setup_time()
        self._setup_ports()

    def _setup_wavelength(self) -> None:
        self.fdtd.setglobalsource("wavelength start", self.wavelength_start * 1e-6)
        self.fdtd.setglobalsource("wavelength stop", self.wavelength_end * 1e-6)
        self.fdtd.setglobalmonitor("frequency points", self.wavelength_points)

    def _setup_field_monitors(self) -> None:
        """Setup the field monitors."""
        self.logger.info(f"Setting up field monitors: {self.field_monitors}")

        for m in self.field_monitors:
            if m == "z":
                self.fdtd.addprofile(
                    name="profile_z",
                    monitor_type="2D Z-normal",
                )
                self.fdtd.set("x", self.center[0] * 1e-6)
                self.fdtd.set("y", self.center[1] * 1e-6)
                self.fdtd.set("z", self.center[2] * 1e-6)
                self.fdtd.set("x span", self.span[0] * 1e-6)
                self.fdtd.set("y span", self.span[1] * 1e-6)
                self.logger.debug("Created Z-normal field monitor")

            if m == "x":
                self.fdtd.addprofile(
                    name="profile_x",
                    monitor_type="2D X-normal",
                )
                self.fdtd.set("x", self.center[0] * 1e-6)
                self.fdtd.set("y", self.center[1] * 1e-6)
                self.fdtd.set("z", self.center[2] * 1e-6)
                self.fdtd.set("y span", self.span[1] * 1e-6)
                self.fdtd.set("z span", self.span[2] * 1e-6)
                self.logger.debug("Created X-normal field monitor")

            if m == "y":
                self.fdtd.addprofile(
                    name="profile_y",
                    monitor_type="2D Y-normal",
                )
                self.fdtd.set("x", self.center[0] * 1e-6)
                self.fdtd.set("y", self.center[1] * 1e-6)
                self.fdtd.set("z", self.center[2] * 1e-6)
                self.fdtd.set("x span", self.span[0] * 1e-6)
                self.fdtd.set("z span", self.span[2] * 1e-6)
                self.logger.debug("Created Y-normal field monitor")

    def _setup_ports(self) -> None:
        """
        Setup ports in Lumerical FDTD using the standardized fdtd_port objects.

        This method translates the solver-agnostic fdtd_port objects into
        Lumerical-specific port configurations.
        """
        # Setup each port in Lumerical using the pre-converted fdtd_ports
        for fdtd_port_obj in self.fdtd_ports:
            port = self.fdtd.addport()
            self.fdtd.set("name", fdtd_port_obj.name)

            # Set position
            self.fdtd.set("x", fdtd_port_obj.position[0] * 1e-6)
            self.fdtd.set("y", fdtd_port_obj.position[1] * 1e-6)
            self.fdtd.set("z", fdtd_port_obj.position[2] * 1e-6)

            # Set direction
            direction_map = {"forward": "Forward", "backward": "Backward"}
            self.fdtd.set("direction", direction_map[fdtd_port_obj.direction])

            # Determine injection axis and spans based on which span element is None
            # The None element indicates the injection axis direction
            if fdtd_port_obj.span[1] is None:  # y_span is None -> y-axis injection
                self.fdtd.set("injection axis", "y-axis")
                self.fdtd.set("x span", fdtd_port_obj.span[0] * 1e-6)  # width_ports
                self.fdtd.set("z span", fdtd_port_obj.span[2] * 1e-6)  # depth_ports
            elif fdtd_port_obj.span[0] is None:  # x_span is None -> x-axis injection
                self.fdtd.set("injection axis", "x-axis")
                self.fdtd.set("y span", fdtd_port_obj.span[1] * 1e-6)  # width_ports
                self.fdtd.set("z span", fdtd_port_obj.span[2] * 1e-6)  # depth_ports
            else:
                raise ValueError(
                    f"Invalid span configuration for port {fdtd_port_obj.name}: {fdtd_port_obj.span}. "
                    f"Exactly one span element must be None to indicate injection axis."
                )

            # set the port modes
            self.fdtd.set("mode selection", "user select")
            self.fdtd.eval(f"updateportmodes({fdtd_port_obj.modes});")
            self.fdtd.set("number of field profile samples", self.mode_freq_pts)

    def _setup_time(self) -> None:
        """Setup the appropriate simulation time span."""
        # Get the maximum dimension in meters
        max_span = max(self.span[0], self.span[1], self.span[2]) * 1e-6

        # Use the common time calculation method from base class
        time_span = self._calculate_simulation_time(max_span)

        # Set the time span
        self.fdtd.setnamed("FDTD", "simulation time", time_span)

    def _setup_mesh(self) -> None:
        """Setup the mesh."""
        # mesh mapping for Lumerical FDTD (format: mesh option: number of mesh cells per wavelength)
        # refer to: https://optics.ansys.com/hc/en-us/articles/360034382534-FDTD-solver-Simulation-Object
        possible_meshes = {
            1: 6,
            2: 10,
            3: 14,
            4: 18,
            5: 22,
            6: 26,
            7: 30,
            8: 34,
        }

        # map user input for mesh cells per wavelength to the closest mesh option based on possible_meshes dictionary
        # e.g. if self.mesh = 11, then mesh_option = 2 (10 mesh cells per wavelength is closest available option)

        # Find the closest mesh option
        min_diff = float("inf")
        mesh_option = 2  # default to mid-low

        for option, cells_per_wavelength in possible_meshes.items():
            diff = abs(self.mesh - cells_per_wavelength)
            if diff < min_diff:
                min_diff = diff
                mesh_option = option

        self.logger.info(
            f"User requested {self.mesh} mesh cells per wavelength, using mesh option {mesh_option} ({possible_meshes[mesh_option]} cells per wavelength)"
        )

        self.fdtd.setnamed("FDTD", "mesh accuracy", mesh_option)

    def _setup_boundary_symmetry(self) -> None:
        """Setup the boundary and symmetry conditions."""
        self.fdtd.setnamed("FDTD", "x max bc", self.boundary[0])
        self.fdtd.setnamed("FDTD", "y max bc", self.boundary[1])
        self.fdtd.setnamed("FDTD", "z max bc", self.boundary[2])

        if self.symmetry[0] == 0:
            self.fdtd.setnamed("FDTD", "x min bc", self.boundary[0])
        if self.symmetry[1] == 0:
            self.fdtd.setnamed("FDTD", "y min bc", self.boundary[1])
        if self.symmetry[2] == 0:
            self.fdtd.setnamed("FDTD", "z min bc", self.boundary[2])

        if self.symmetry[0] == 1:
            self.fdtd.setnamed("FDTD", "x min bc", "Symmetric")
        if self.symmetry[1] == 1:
            self.fdtd.setnamed("FDTD", "y min bc", "Symmetric")
        if self.symmetry[2] == 1:
            self.fdtd.setnamed("FDTD", "z min bc", "Symmetric")

        if self.symmetry[0] == -1:
            self.fdtd.setnamed("FDTD", "x min bc", "Anti-Symmetric")
        if self.symmetry[1] == -1:
            self.fdtd.setnamed("FDTD", "y min bc", "Anti-Symmetric")
        if self.symmetry[2] == -1:
            self.fdtd.setnamed("FDTD", "z min bc", "Anti-Symmetric")

    def get_log(self) -> str:
        """Get the log of the simulation."""
        try:
            if hasattr(self, "fdtd") and self.fdtd:
                # Try to get the log from Lumerical FDTD
                # Note: This may vary by Lumerical version
                try:
                    log_text = self.fdtd.getresult("FDTD", "log")
                    if log_text:
                        return str(log_text)
                    else:
                        return "No log data available from Lumerical FDTD."
                except:
                    # Fallback - try alternative log access methods
                    try:
                        # Try getting simulation log through eval
                        self.fdtd.eval("log_data = getresult('FDTD', 'log');")
                        log_data = self.fdtd.getv("log_data")
                        if log_data:
                            return str(log_data)
                    except:
                        pass
                    return "Log retrieval not available for this Lumerical version."
            else:
                return "FDTD solver not initialized."
        except Exception as e:
            return f"Error retrieving log: {str(e)}"
