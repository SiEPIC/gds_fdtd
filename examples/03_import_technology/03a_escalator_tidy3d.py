# %%
"""
Example defining a tidy3d simulation by importing a GDS with an associated technology file.

@author: Mustafa Hammood
"""
import os
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech, make_t3d_sim
from gds_fdtd.lyprocessor import load_cell

if __name__ == "__main__":

    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # Define the path to the GDS file
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")

    cell = load_cell(file_gds, top_cell="si_sin_escalator_te1550")

    device = load_component_from_tech(cell=cell, tech=technology, z_span=5)

    simulation = make_t3d_sim(
        device=device,
        in_port=device.ports[0],
        wavl_min=1.4,
        wavl_max=1.7,
        wavl_pts=501,
        grid_cells_per_wvl=8,
        symmetry=(
            0,
            0,
            0,
        ),  # ensure structure is symmetric across symmetry axis before triggering this!
        z_span=5,
        field_monitor_axis="y",
    )
    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    simulation.execute()
    #  visualize the results
    simulation.visualize_results()

# %%
