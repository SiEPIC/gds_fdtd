# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mustafa Hammood
"""
import gds_fdtd as gtd
import tidy3d as td
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import make_t3d_sim, load_component_from_tech
import os

if __name__ == "__main__":

    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # Define the path to the GDS file
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")

    cell, layout = load_cell(file_gds, top_cell='bragg_te1550', prefab="ANT_NanoSOI_ANF1_d10")
    component = load_component_from_tech(cell=cell, tech=technology)
    simulation = make_t3d_sim(
        c=component,
        in_port=component.ports[0],
        wavl_min=1.4,
        wavl_max=1.6,
        wavl_pts=101,
        grid_cells_per_wvl=6,
        symmetry=(
            0,
            0,
            1,
        ),  # ensure structure is symmetric across symmetry axis before triggering this!
        z_span=3,
    )

    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    simulation.execute()
    #  visualize the results
    simulation.visualize_results()

# %%
