#%% 
"""
Example of simulating a component in lumerical.
@author: Mustafa Hammood
"""
import os
from lumapi import FDTD  # can also be mode/device
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
        port_input=component.ports[0],
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
        modes=[1],
        run_time_factor=5,
        field_monitors=["z"],
        working_dir=os.getcwd(),  # Files will be saved to ./crossing_te1550/ subdirectory
    )
    solver.run()
    input("Proceed to terminate the GUI?")

# %%
