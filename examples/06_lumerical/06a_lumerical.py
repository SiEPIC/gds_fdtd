#%% 
"""
Example of simulating a component in lumerical.
@author: Mustafa Hammood
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from gds_fdtd.solver_lumerical import fdtd_solver_lumerical
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.lyprocessor import load_cell

os.environ['QT_QPA_PLATFORM'] = 'xcb'  # i need to do this to get my lumerical gui to work in linux... comment out if not necessary

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_lumerical.yaml")  # note materials in yaml
    technology = parse_yaml_tech(tech_path)

    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")
    cell, layout = load_cell(file_gds, top_cell='crossing_te1550')
    component = load_component_from_tech(cell=cell, tech=technology)

    solver = fdtd_solver_lumerical(
        component=component,
        tech=technology,
        port_input=[component.ports[0]],
        gpu=False,
        wavelength_start=1.5,
        wavelength_end=1.6,
        wavelength_points=100,
        mesh=6,
        boundary=["PML", "PML", "Metal"],
        symmetry=[0, 0, 0],
        z_min=-1.0,
        z_max=1.11,
        width_ports=2.0,
        depth_ports=1.5,
        buffer=1.0,
        modes=[1, 2],
        run_time_factor=5,
        field_monitors=["z"],
        working_dir=os.getcwd(),  # Files will be saved to ./crossing_te1550/ subdirectory
    )
    solver.run()
    input("Proceed to terminate the GUI?")

    # plot all the s-parameters
    solver.sparameters.plot()

    # example fetching specific s-parameters of interest
    wavl = solver.sparameters.wavelength
    s21_te = solver.sparameters.S(in_port=1, out_port=4, in_modeid=1, out_modeid=1)
    s21_tm = solver.sparameters.S(in_port=1, out_port=4, in_modeid=2, out_modeid=2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(wavl, 10*np.log10(np.abs(s21_te.s_mag)**2), label='S41 TE->TE')
    ax.plot(wavl, 10*np.log10(np.abs(s21_tm.s_mag)**2), label='S41 TM->TM')
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('Transmission [dB]')
    ax.set_ylim(-1, 0)
    ax.legend()
    plt.show()

# %%
