                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    #%% 
"""
Example of simulating a component in Tidy3D using the new fdtd_solver_tidy3d class.

@author: Mustafa Hammood
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.lyprocessor import load_cell

if __name__ == "__main__":
    # Load technology file - need to modify for Tidy3D materials
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # Load GDS component
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")
    cell, layout = load_cell(file_gds, top_cell='crossing_te1550')
    component = load_component_from_tech(cell=cell, tech=technology)

    # Create Tidy3D solver
    solver = fdtd_solver_tidy3d(
        component=component,
        tech=technology,
        port_input=[component.ports[0]],  # Input port(s)
        visualize=False,  # Set to True to show simulation setup plots
        wavelength_start=1.5,
        wavelength_end=1.6,
        wavelength_points=50,  # Fewer points for faster simulation
        mesh=6,  # Grid cells per wavelength
        boundary=["PML", "PML", "PML"],
        symmetry=[0, 0, 0],
        z_min=-1.0,
        z_max=1.11,
        width_ports=2.0,
        depth_ports=1.5,
        buffer=1.0,
        modes=[1, 2],  # TE and TM modes
        run_time_factor=50,
        field_monitors=["z"],
        working_dir=os.getcwd(),  # Files will be saved to ./crossing_te1550/ subdirectory
    )
    
    # This requires Tidy3D cloud credentials and will consume simulation credits   
    solver.run()
    
    # Plot all the s-parameters
    solver.sparameters.plot()
    #%%
    # Example fetching specific s-parameters of interest (multi-modal)
    wavl = solver.sparameters.wavelength
    s41_te_te = solver.sparameters.S(in_port=1, out_port=4, in_modeid=1, out_modeid=1)  # TE->TE
    s41_tm_tm = solver.sparameters.S(in_port=1, out_port=4, in_modeid=2, out_modeid=2)  # TM->TM

    # Multi-modal transmission plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(wavl, 10*np.log10(np.abs(s41_te_te.s_mag)**2), label='S41 TE→TE', linewidth=2)
    ax.plot(wavl, 10*np.log10(np.abs(s41_tm_tm.s_mag)**2), label='S41 TM→TM', linewidth=2)
    ax.set_xlabel('Wavelength [μm]')
    ax.set_ylabel('Transmission [dB]')
    ax.set_title('Multi-Modal S-Parameters: Port 1 → Port 4')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
        # Visualize results (S-parameters plots and export to .dat)
    solver.visualize_results()
    
    # Show field monitor data
    print("\nField Monitor Visualization:")
    print("-" * 40)
    solver.visualize_field_monitors()
    
    # Show field monitor information
    for monitor in solver.field_monitors_objs:
        print(f"Field Monitor Info: {monitor.get_field_info()}")
        
    print("\nLogging Information:")
    print("-" * 40)
    log_files = [f for f in os.listdir(solver.working_dir) if f.endswith('.log')]
    if log_files:
        print(f"Detailed simulation log saved to: {log_files[0]}")
        print("Log contains detailed simulation information for debugging")
    else:
        print("No log files found")

# %% 