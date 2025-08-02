"""
gds_fdtd simulation toolbox.

Lumerical tools interface module.
@author: Mustafa Hammood, 2025
"""
import os
from lumapi import FDTD
from gds_fdtd.solver import fdtd_solver, fdtd_field_monitor
from gds_fdtd.sparams import process_dat

class fdtd_solver_lumerical(fdtd_solver):
    """
    FDTD solver for electromagnetic simulations using Lumerical.
    """

    def __init__(self, *args, gpu: bool = False, **kwargs):
        """Initialize the Lumerical solver by calling the parent constructor.
        Args:
            gpu (bool): Whether to use GPU acceleration.
        """
        super().__init__(*args, **kwargs)
        self.gpu = gpu
        self.setup()

    def setup(self) -> None:
        """Setup the Lumerical simulation."""
        # Export GDS with port extensions to working directory
        self._export_gds()

        # Initialize FDTD
        self.fdtd = FDTD()

        # set the working directory
        self.fdtd.eval(f'cd("{self.working_dir}");')

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
        print(f"FDTD project saved: {project_filepath}")

        # Get the resources used by the simulation
        self.get_resources()

    def get_resources(self) -> None:
        """Get the resources used by the simulation."""
        # TODO: I don't think this will work for Lumerical 2025 with GPU
        # ref: https://optics.ansys.com/hc/en-us/articles/4403937981715-runsystemcheck-Script-command
        self.fdtd.eval("reqs = runsystemcheck;")

        reqs = self.fdtd.getv("reqs")

        memory = reqs['Memory_Recommended']  # in Bytes
        nodes = reqs['Total_FDTD_Yee_Nodes']  # in MNodes
        print(f"Memory: {memory/1e9} GB")
        print(f"Nodes: {nodes} MNodes")

    def run(self) -> None:
        """Run the simulation."""
        # note: this only works for Lumerical 2024.
        # For Lumerical 2025, Lumerical has changed the syntax for setting GPU.
        # I Don't have a 2025 license, so I can't test it.
        # If you do, please update the code to make it work for 2025 as well as 2024.
        # lumerical: please make your api backwards compatible :)))
        if self.gpu:
            self.fdtd.setresource("FDTD", "GPU", 1)
        else:
            self.fdtd.setresource("FDTD", "CPU", 1)
        self.fdtd.runsweep("sparams")

        self.get_results()

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

        Args:
            active_ports (list, optional): List of port names to activate. If None, activates all ports.
        """
        # Create S-parameter sweep
        self.fdtd.addsweep(3)  # 3 is s-parameter sweep
        self.fdtd.setsweep("s-parameter sweep", "name", "sparams")
        self.fdtd.setsweep(
            "sparams", "Excite all ports", 0
        )  # We'll manually set active ports

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

        # Automatically generate indices based on sorted fdtd_ports and modes
        indices = []

        for i, fdtd_port in enumerate(self.fdtd_ports):
            port_name = fdtd_port.name

            # Determine if this port should be active
            is_active = fdtd_port.name in active_port_names

            for mode_idx in self.modes:
                mode_name = (
                    f"mode {mode_idx}"  # Lumerical uses "mode 1", "mode 2", etc.
                )

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
            except Exception as e:
                print(f"Done removing sweep parameters")
                break

        # Add all port-mode combinations to the sweep
        for idx in indices:
            self.fdtd.addsweepparameter("sparams", idx)

        # Print summary of S-parameter sweep configuration
        active_combinations = [idx for idx in indices if idx["Active"] == 1]
        total_combinations = len(indices)
        print(f"S-parameter sweep configured:")
        print(f"  Total port-mode combinations: {total_combinations}")
        print(f"  Active combinations: {len(active_combinations)}")
        print(f"  Active ports: {active_port_names}")
        print(f"  Modes: {self.modes}")

        if len(active_combinations) > 0:
            print("  Active combinations:")
            for combo in active_combinations:
                print(f"    {combo['Port']} - {combo['Mode']}")
        else:
            print("  Warning: No active port-mode combinations found!")

    def _setup_layer_builder(self) -> None:
        """
        Setup the Lumerical layer builder with technology information.

        """
        self.fdtd.addlayerbuilder()

        self.fdtd.setnamed(
            "layer group", "x", self.center[0] * 1e-6
        )  # Convert to meters
        self.fdtd.setnamed("layer group", "y", self.center[1] * 1e-6)
        self.fdtd.setnamed("layer group", "z", 0)
        self.fdtd.setnamed(
            "layer group", "x span", (self.span[0] + 2 * self.buffer) * 1e-6
        )
        self.fdtd.setnamed(
            "layer group", "y span", (self.span[1] + 2 * self.buffer) * 1e-6
        )
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
                z_start_adjusted = (
                    z_start  # z_base is already at the top for negative growth
                )
                self.fdtd.eval(
                    f'setlayer("substrate", "start position", {z_start_adjusted});'
                )

            if substrate["material"] and "lum_db" in substrate["material"]:
                material_name = substrate["material"]["lum_db"]["model"]
                self.fdtd.eval(
                    f'setlayer("substrate", "background material", "{material_name}");'
                )
            # Substrate typically covers the entire simulation domain
            self.fdtd.eval(
                'setlayer("substrate", "layer number", "");'
            )  # No specific GDS layer

        # Set up superstrate layer (always add as it's a background layer)
        if tech_dict["superstrate"]:
            superstrate = tech_dict["superstrate"][0]
            self.fdtd.eval('addlayer("superstrate");')

            # Set z start position and thickness
            z_start = superstrate["z_base"] * 1e-6  # Convert to meters
            thickness = (
                abs(superstrate["z_span"]) * 1e-6
            )  # Always positive for superstrate

            self.fdtd.eval(f'setlayer("superstrate", "start position", {z_start});')
            self.fdtd.eval(f'setlayer("superstrate", "thickness", {thickness});')

            if superstrate["material"] and "lum_db" in superstrate["material"]:
                material_name = superstrate["material"]["lum_db"]["model"]
                self.fdtd.eval(
                    f'setlayer("superstrate", "background material", "{material_name}");'
                )
            # Superstrate typically covers the entire simulation domain
            self.fdtd.eval(
                'setlayer("superstrate", "layer number", "");'
            )  # No specific GDS layer

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
                thickness = (
                    abs(device_layer["z_span"]) * 1e-6
                )  # Always positive thickness

                self.fdtd.eval(
                    f'setlayer("{layer_name}", "start position", {z_start});'
                )
                self.fdtd.eval(f'setlayer("{layer_name}", "thickness", {thickness});')

                # Set GDS layer mapping
                layer_spec = f"{gds_layer[0]}:{gds_layer[1]}"
                self.fdtd.eval(
                    f'setlayer("{layer_name}", "layer number", "{layer_spec}");'
                )

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
                print(
                    f"Added layer {layer_name} for GDS layer {layer_spec} at z={device_layer['z_base']}μm, thickness={device_layer['z_span']}μm"
                )
            else:
                print(
                    f"Skipping device layer {idx} (GDS layer {gds_layer[0]}:{gds_layer[1]}) - not found in GDS file"
                )

        print(
            f"Layer builder setup complete with {layers_added} device layers (out of {len(tech_dict['device'])} defined in technology)"
        )
        if gds_layers_result is not None:
            print(f"GDS file contains layers: {gds_layers_result}")

    def _setup_fdtd(self) -> None:
        """Setup the FDTD simulation."""
        self.fdtd.addfdtd()
        self.fdtd.setnamed("FDTD", "x", self.center[0] * 1e-6)
        self.fdtd.setnamed("FDTD", "y", self.center[1] * 1e-6)
        self.fdtd.setnamed("FDTD", "z", self.center[2] * 1e-6)
        self.fdtd.setnamed("FDTD", "x span", (self.span[0]) * 1e-6)
        self.fdtd.setnamed("FDTD", "y span", (self.span[1]) * 1e-6)
        self.fdtd.setnamed("FDTD", "z span", self.span[2] * 1e-6)

        # configure GPU acceleration
        # TODO: IMPORTANT: This is valid for Lumerical 2024.
        # For Lumerical 2025, Lumerical has changed the syntax for setting GPU.
        # If you do, please update the code to make it work for 2025 as well as 2024
        # I Don't have a 2025 license, so I can't test it.
        if self.gpu:
            self.fdtd.setnamed("FDTD", "express mode", True)
        else:
            self.fdtd.setnamed("FDTD", "express mode", False)

        self._setup_wavelength()
        self._setup_boundary_symmetry()
        self._setup_mesh()
        self._setup_time()
        self._setup_ports()

    def _setup_wavelength(self) -> None:
        self.fdtd.setglobalsource("wavelength start", self.wavelength_start*1e-6)
        self.fdtd.setglobalsource("wavelength stop", self.wavelength_end*1e-6)
        self.fdtd.setglobalmonitor("frequency points", self.wavelength_points)

    def _setup_field_monitors(self) -> None:
        """Setup the field monitors."""
        field_monitors = []
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
                field_monitors.append(
                    fdtd_field_monitor(name="profile_z", monitor_type="z")
                )
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
                field_monitors.append(
                    fdtd_field_monitor(name="profile_x", monitor_type="x")
                )
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
                field_monitors.append(
                    fdtd_field_monitor(name="profile_y", monitor_type="y")
                )

        self.field_monitors_objs = field_monitors

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
        c = 299792458  # speed of light in m/s
        # get the maximum time span from the component using the maximum span in the x, y, and z directions
        max_span = (
            max(self.span[0], self.span[1], self.span[2]) * 1e-6
        )  # convert to meters
        # assume the maximum group index is 4.5
        # TODO: get the maximum group index from the component based on the material properties and center wavelength of the simulation
        max_group_index = 4.5
        # velocity of pulse in the medium
        v = c / max_group_index
        # calculate the time span based on the maximum group index and the speed of light, converted to fs
        time_span = self.run_time_factor * max_span / (v)
        # set the time span
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

        print(
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
