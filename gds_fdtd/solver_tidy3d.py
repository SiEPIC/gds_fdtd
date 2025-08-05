"""
gds_fdtd simulation toolbox.

Tidy3D FDTD solver interface module.
@author: Mustafa Hammood, 2025
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from gds_fdtd.solver import fdtd_solver
from gds_fdtd.core import sparam
from gds_fdtd.sparams import sparameters

class fdtd_solver_tidy3d(fdtd_solver):
    """
    FDTD solver for electromagnetic simulations using Tidy3D.
    
    This implementation directly replicates the working t3d_tools.py approach.
    """

    def __init__(self, *args, visualize: bool = True, **kwargs):
        """Initialize the Tidy3D solver by calling the parent constructor."""
        super().__init__(*args, **kwargs)
        self.visualize = visualize
        self.simulation = None
        self.setup()

    def setup(self) -> None:
        """Setup the Tidy3D simulation using the exact t3d_tools.py pattern."""
        # Validate simulation parameters
        self._validate_simulation_parameters()
        
        # Export GDS with port extensions to working directory
        self._export_gds()

        # Calculate frequencies exactly like t3d_tools.py
        lda0 = (self.wavelength_end + self.wavelength_start) / 2
        freq0 = td.C_0 / lda0
        freqs = td.C_0 / np.linspace(self.wavelength_start, self.wavelength_end, self.wavelength_points)
        fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

        # Store for later use
        self.lda0 = lda0
        self.freq0 = freq0
        self.freqs = freqs
        self.fwidth = fwidth

        # Create simulation using exact t3d_tools.py workflow
        self.simulation = self._make_t3d_sim_like_original()

        # Print setup summary
        self._print_simulation_summary()
        print(f"Tidy3D solver setup complete with {len(self.simulation.sim_jobs)} simulation jobs")

    def _make_t3d_sim_like_original(self):
        """Replicate the exact make_t3d_sim function workflow from t3d_tools.py"""
        
        # Convert solver parameters to t3d_tools.py format
        device = self.component
        in_port = self.port_input if isinstance(self.port_input, list) else [self.port_input] if self.port_input else [device.ports[0]]
        mode_index = [m-1 for m in self.modes]  # Convert to 0-based like t3d_tools.py
        
        # define structures from device
        structures = self._make_structures()

        # define monitors - Use standardized fdtd_port objects like Lumerical
        monitors = []
        for fdtd_port in self.fdtd_ports:
            monitors.append(
                self._make_port_monitor(
                    fdtd_port,
                    freqs=self.freqs,
                    depth=self.depth_ports,
                    width=self.width_ports,
                    num_modes=len(self.modes),
                )
            )

        # make field monitor and create field monitor objects
        if self.field_monitors:
            for field_monitor_axis in self.field_monitors:
                field_monitor = self._make_field_monitor(device, freqs=self.freqs, axis=field_monitor_axis)
                monitors.append(field_monitor)
                
                # Create field monitor object for visualization
                from gds_fdtd.solver import fdtd_field_monitor
                field_monitor_obj = fdtd_field_monitor(
                    name=field_monitor.name,
                    monitor_type=field_monitor_axis
                )
                if not hasattr(self, 'field_monitors_objs'):
                    self.field_monitors_objs = []
                self.field_monitors_objs.append(field_monitor_obj)

        # simulation domain size (in microns) - Use solver's calculated span that includes buffer
        sim_size = [self.span[0], self.span[1], self.span[2]]  # Use solver's span, not device.bounds
        
        # run time calculation
        run_time = (
            self.run_time_factor * max(sim_size) / td.C_0
        )

        # Create boundary spec
        boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

        # define sim jobs - Use standardized fdtd_port objects like Lumerical  
        sim_jobs = []
        for m in mode_index:
            for fdtd_port in self.fdtd_ports:
                # Only create jobs for active ports
                if fdtd_port.name in [p.name for p in in_port]:
                    source = self._make_source(
                        fdtd_port=fdtd_port,
                        depth=self.depth_ports,
                        width=self.width_ports,
                        freq0=self.freq0,
                        num_freqs=3,
                        fwidth=self.fwidth,
                        num_modes=len(self.modes),
                        mode_index=m,
                    )
                    sim = {}
                    sim["name"] = f"{device.name}_{fdtd_port.name}_idx{m}"
                    sim["source"] = source
                    sim["in_port"] = fdtd_port  # Use fdtd_port object
                    sim["num_modes"] = len(self.modes)
                    sim["sim"] = td.Simulation(
                        size=sim_size,
                        grid_spec=td.GridSpec.auto(
                            min_steps_per_wvl=self.mesh, wavelength=self.lda0
                        ),
                        structures=structures,
                        sources=[source],
                        monitors=monitors,
                        run_time=run_time,
                        boundary_spec=boundary_spec,
                        center=(
                            self.center[0],  # Use solver's center, not device.bounds
                            self.center[1],  # Use solver's center, not device.bounds  
                            self.center[2],  # Use solver's center, not device.bounds
                        ),
                        symmetry=tuple(self.symmetry),
                    )
                    sim_jobs.append(sim)

        # Create sim_tidy3d-like object
        class TempSimulation:
            def __init__(self, in_port, device, wavl_min, wavl_max, wavl_pts, sim_jobs):
                self.in_port = in_port
                self.device = device
                self.wavl_min = wavl_min
                self.wavl_max = wavl_max
                self.wavl_pts = wavl_pts
                self.sim_jobs = sim_jobs
                self.results = None

        return TempSimulation(
            in_port=in_port,
            wavl_max=self.wavelength_end,
            wavl_min=self.wavelength_start,
            wavl_pts=self.wavelength_points,
            device=device,
            sim_jobs=sim_jobs,
        )

    def _make_structures(self):
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
                    if "substrate" in i.name.lower() or "superstrate" in i.name.lower() or "subtrate" in i.name.lower():
                        # Extend substrate/superstrate to be flush with port extensions (2*buffer from component edge)
                        component_max_extent = max(
                            abs(device.bounds.x_center) + device.bounds.x_span/2,
                            abs(device.bounds.y_center) + device.bounds.y_span/2
                        )
                        substrate_half_size = component_max_extent + 2 * self.buffer  # Flush with port extensions
                        extended_vertices = [
                            (-substrate_half_size, -substrate_half_size),
                            (substrate_half_size, -substrate_half_size), 
                            (substrate_half_size, substrate_half_size),
                            (-substrate_half_size, substrate_half_size)
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
                            medium=i.material["tidy3d"] if isinstance(i.material, dict) else i.material,
                            name=i.name,
                        )
                    )
            else:
                if s.z_span < 0:
                    bounds = (s.z_base + s.z_span, s.z_base)
                else:
                    bounds = (s.z_base, s.z_base + s.z_span)
                
                # Check if this is substrate/superstrate and extend to be flush with port extensions
                if "substrate" in s.name.lower() or "superstrate" in s.name.lower() or "subtrate" in s.name.lower():
                    # Extend substrate/superstrate to be flush with port extensions (2*buffer from component edge)
                    component_max_extent = max(
                        abs(device.bounds.x_center) + device.bounds.x_span/2,
                        abs(device.bounds.y_center) + device.bounds.y_span/2
                    )
                    substrate_half_size = component_max_extent + 2 * self.buffer  # Flush with port extensions
                    extended_vertices = [
                        (-substrate_half_size, -substrate_half_size),
                        (substrate_half_size, -substrate_half_size), 
                        (substrate_half_size, substrate_half_size),
                        (-substrate_half_size, substrate_half_size)
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
                        sidewall_angle=(90 - device.structures[0].sidewall_angle)
                        * (np.pi / 180),
                    ),
                    medium=p.material["tidy3d"] if isinstance(p.material, dict) else p.material,
                    name=f"port_{p.name}",
                )
            )
        return structures

    def _make_port_monitor(self, fdtd_port, freqs=2e14, num_modes=1, depth=2, width=3):
        """Create mode monitor exactly at fdtd_port position (consistent with Lumerical)."""
        # Monitor position - exactly at port position (no buffer like Lumerical)
        center = [fdtd_port.position[0], fdtd_port.position[1], fdtd_port.position[2]]
        
        # Determine size based on fdtd_port span (None indicates injection axis)
        if fdtd_port.span[0] is None:  # x-axis injection
            size = [0, width, depth]
        elif fdtd_port.span[1] is None:  # y-axis injection  
            size = [width, 0, depth]
        else:
            raise ValueError(f"Invalid span configuration for port {fdtd_port.name}")

        return td.ModeMonitor(
            center=center,
            size=size,
            freqs=freqs,
            mode_spec=td.ModeSpec(num_modes=num_modes),
            name=fdtd_port.name,
        )

    def _make_source(self, fdtd_port, num_modes=1, mode_index=0, width=3.0, depth=2.0, freq0=2e14, num_freqs=5, fwidth=1e13, buffer=-0.2):
        """Create mode source with buffer offset from fdtd_port position."""
        # Source position - apply buffer offset from port position
        center = [fdtd_port.position[0], fdtd_port.position[1], fdtd_port.position[2]]
        
        # Apply buffer offset and direction based on fdtd_port mapping
        # This replicates the t3d_tools.py logic with fdtd_port objects
        if fdtd_port.span[0] is None:  # x-axis injection
            if fdtd_port.direction == "forward":
                center[0] += buffer  # buffer = -0.2, so moves toward device center
                direction = "+"  # From t3d_tools.py: port directions 180,270 → "+"
            else:  # backward
                center[0] -= buffer  # Moves toward device center
                direction = "-"  # From t3d_tools.py: port directions 0,90 → "-"
            size = [0, width, depth]
        elif fdtd_port.span[1] is None:  # y-axis injection
            if fdtd_port.direction == "forward":
                center[1] += buffer  # buffer = -0.2, so moves toward device center
                direction = "+"
            else:  # backward
                center[1] -= buffer  # Moves toward device center
                direction = "-" 
            size = [width, 0, depth]
        else:
            raise ValueError(f"Invalid span configuration for port {fdtd_port.name}")

        return td.ModeSource(
            center=center,
            size=size,
            direction=direction,
            source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
            mode_spec=td.ModeSpec(num_modes=num_modes),
            mode_index=mode_index,
            num_freqs=num_freqs,
            name=f"msource_{fdtd_port.name}_idx{mode_index}",
        )

    def _make_field_monitor(self, device, freqs=2e14, axis="z", z_center=None):
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
        if not self.simulation or not self.simulation.sim_jobs:
            print("No simulation jobs available.")
            return
            
        print(f"Simulation jobs created: {len(self.simulation.sim_jobs)}")
        print("Resource estimation not implemented for Tidy3D (cloud-based)")

    def run(self) -> None:
        """Run the simulation using the exact t3d_tools.py pattern."""
        if not self.simulation:
            raise RuntimeError("No simulation created. Call setup() first.")
            
        # Upload jobs to Tidy3D cloud
        self._upload_jobs()
        
        # Execute jobs and extract S-parameters
        self._execute_jobs()

    def _upload_jobs(self):
        """Upload simulation jobs to Tidy3D cloud."""
        from tidy3d import web

        # divide between job and sim, how to attach them?
        for sim_job in self.simulation.sim_jobs:
            sim = sim_job["sim"]
            name = sim_job["name"]
            sim_job["job"] = web.Job(simulation=sim, task_name=name)

    def _execute_jobs(self):
        """Execute jobs and extract S-parameters (updated for fdtd_port consistency)"""
        def get_directions(fdtd_ports):
            directions = []
            for fdtd_port in fdtd_ports:
                # Convert fdtd_port directions to monitor directions
                if fdtd_port.direction == "forward":
                    directions.append("+")
                else:  # backward
                    directions.append("-")
            return tuple(directions)

        def get_source_direction(fdtd_port):
            # Convert fdtd_port direction to source direction (opposite of monitor)
            if fdtd_port.direction == "forward":
                return "-"
            else:  # backward
                return "+"

        def get_port_name(port_name):
            return [int(i) for i in port_name if i.isdigit()][0]

        def measure_transmission(in_fdtd_port, in_mode_idx: int, out_mode_idx: int):
            """
            Constructs a "row" of the scattering matrix.
            """
            num_ports = len(self.fdtd_ports)

            if isinstance(self.simulation.results, list):
                if len(self.simulation.results) == 1:
                    results = self.simulation.results[0]
                else:
                    # TBD: Handle the case where self.results is a list with more than one item
                    print("Multiple results handler is WIP, using first results entry")
                    results = self.simulation.results[-1]
            else:
                results = self.simulation.results

            input_amp = results[in_fdtd_port.name].amps.sel(
                direction=get_source_direction(in_fdtd_port),
                mode_index=in_mode_idx,
            )
            amps = np.zeros((num_ports, self.simulation.wavl_pts), dtype=complex)
            directions = get_directions(self.fdtd_ports)
            for i, (monitor, direction) in enumerate(
                zip(results.simulation.monitors[:num_ports], directions)
            ):
                amp = results[monitor.name].amps.sel(
                    direction=direction, mode_index=out_mode_idx
                )
                amp_normalized = amp / input_amp
                amps[i] = np.squeeze(amp_normalized.values)

            return amps

        self.simulation.s_parameters = sparameters(self.simulation.device.name)  # initialize empty s parameters

        self.simulation.results = []
        for sim_job in self.simulation.sim_jobs:
            if not os.path.exists(self.simulation.device.name):
                os.makedirs(self.simulation.device.name)
            self.simulation.results.append(
                sim_job["job"].run(
                    path=os.path.join(self.simulation.device.name, f"{sim_job['name']}.hdf5")
                )
            )
            for mode in range(sim_job["num_modes"]):
                amps_arms = measure_transmission(
                    in_fdtd_port=sim_job["in_port"],  # Now using fdtd_port
                    in_mode_idx=sim_job["source"].mode_index,
                    out_mode_idx=mode,
                )

                print("Mode amplitudes in each port: \n")
                wavl = np.linspace(self.simulation.wavl_min, self.simulation.wavl_max, self.simulation.wavl_pts)
                for amp, monitor in zip(
                    amps_arms, self.simulation.results[-1].simulation.monitors
                ):
                    print(f'\tmonitor     = "{monitor.name}"')
                    print(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
                    print(f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n")

                                        # Convert complex amplitude to magnitude and phase for sparameters interface
                    in_port_num = next(p.idx for p in self.component.ports if p.name == sim_job["in_port"].name)
                    out_port_num = get_port_name(monitor.name)
                    
                    # Convert amplitude to magnitude and phase
                    s_mag = np.abs(amp)
                    s_phase = np.angle(amp)
                    
                    self.simulation.s_parameters.add_data(
                        in_port=str(in_port_num),
                        out_port=str(out_port_num),
                        mode_label=1,
                        in_modeid=sim_job["source"].mode_index + 1,  # Convert to 1-based
                        out_modeid=mode + 1,  # Convert to 1-based
                        data_type="transmission",
                        group_delay=0.0,
                        f=list(td.C_0 / wavl),  # frequency in Hz
                        s_mag=list(s_mag),
                        s_phase=list(s_phase),
                    )
        if isinstance(self.simulation.results, list) and len(self.simulation.results) == 1:
            self.simulation.results = self.simulation.results[0]

        # Store results for later access
        self._sparameters = self.simulation.s_parameters

    def get_results(self) -> None:
        """Get the results of the simulation."""
        if not hasattr(self.simulation, 's_parameters') or self.simulation.s_parameters is None:
            print("No results available. Run simulation first.")
            return
        self._sparameters = self.simulation.s_parameters

    def get_log(self) -> None:
        """Get the log of the simulation."""
        print("Tidy3D simulation logs are available through the web interface.")
        print("Individual job logs can be accessed via the Tidy3D web platform.")

    def export_sparameters_dat(self, filepath: str = None):
        """Export S-parameters to .dat file using s_parameter_writer."""
        if not hasattr(self.simulation, 's_parameters') or self.simulation.s_parameters is None:
            print("No S-parameters available for export. Run simulation first.")
            return
            
        if filepath is None:
            filepath = os.path.join(self.working_dir, f"{self.component.name}_sparams.dat")
            
        try:
            from gds_fdtd.sparams import s_parameter_writer
            
            # Create writer instance
            writer = s_parameter_writer()
            writer.name = filepath.replace('.dat', '')
            
            # Set wavelength range
            writer.wavl = [self.wavelength_start * 1e-6, self.wavelength_end * 1e-6, 
                          (self.wavelength_end - self.wavelength_start) * 1e-6 / (self.wavelength_points - 1)]
            
            # Set number of ports
            writer.n_ports = len(self.fdtd_ports)
            
            # Convert S-parameters data to writer format
            writer.data = []
            for data_entry in self.simulation.s_parameters.data:
                # Convert s_mag to power (magnitude squared)
                s_power = [abs(mag)**2 for mag in data_entry.s_mag]
                s_phase = data_entry.s_phase
                writer.data.append([s_power, s_phase])
            
            # Write the file
            writer.write_S()
            print(f"S-parameters exported to: {filepath}")
            
        except Exception as e:
            print(f"Error exporting S-parameters: {e}")

    def visualize_results(self):
        """Visualize the simulation results."""
        if not hasattr(self.simulation, 's_parameters') or self.simulation.s_parameters is None:
            print("No results available for visualization.")
            return
            
        # Plot S-parameters
        self.simulation.s_parameters.plot()
        
        # Export S-parameters to .dat file
        self.export_sparameters_dat()
        
    def visualize_field_monitors(self):
        """Visualize field monitor data through field monitor objects."""
        if not hasattr(self.simulation, 'results') or self.simulation.results is None:
            print("No simulation results available for field visualization.")
            print("Run solver.run() first to generate field data.")
            return
            
        if not self.visualize:
            return
            
        results = self.simulation.results
        if not isinstance(results, list):
            results = [results]
            
        # Get field monitors from field_monitors_objs
        field_monitors = getattr(self, 'field_monitors_objs', [])
        if not field_monitors:
            print("No field monitor objects available.")
            return
            
        freq = td.C_0 / ((self.simulation.wavl_max + self.simulation.wavl_min) / 2)
        
        for field_monitor in field_monitors:
            try:
                field_name = f"{field_monitor.monitor_type}_field"
                
                for result in results:
                    print(f"Visualizing field monitor: {field_name}")
                    
                    # Create flexible field plots
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle(f'Field Monitor: {field_name} at {freq/1e14:.2f} THz')
                    
                    # Plot Ex, Ey, Ez, |E|
                    field_components = ['Ex', 'Ey', 'Ez']
                    
                    for i, component in enumerate(field_components):
                        try:
                            ax = axes[i//2, i%2]
                            result.plot_field(field_name, component, freq=freq, ax=ax)
                            ax.set_title(f'{component} field')
                        except Exception as e:
                            print(f"Could not plot {component}: {e}")
                    
                    # Plot |E| magnitude
                    try:
                        ax = axes[1, 1]
                        # Calculate |E| from Ex, Ey, Ez
                        Ex = result[field_name].Ex.sel(f=freq, method='nearest')
                        Ey = result[field_name].Ey.sel(f=freq, method='nearest')
                        Ez = result[field_name].Ez.sel(f=freq, method='nearest')
                        E_mag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
                        
                        # Simple plot of magnitude
                        im = ax.imshow(np.real(E_mag), cmap='hot', origin='lower')
                        ax.set_title('|E| magnitude')
                        plt.colorbar(im, ax=ax)
                    except Exception as e:
                        print(f"Could not plot |E| magnitude: {e}")
                    
                    plt.tight_layout()
                    plt.show()
                    
            except Exception as e:
                print(f"Could not create field plots for {field_monitor.name}: {e}")