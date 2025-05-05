#%%
"""
Simulation of a halfring device using tidy3d.
@author: Mustafa Hammood
"""
import os
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import make_t3d_sim
import siepic_ebeam_pdk as pdk  # noqa: F401
from pya import Layout
from lumapi import FDTD

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # load cell from siepic_ebeam_pdk
    ly = Layout()
    ly.technology_name = pdk.tech.name
    cell = ly.create_cell("ebeam_dc_halfring_straight", "EBeam", {})

    component = load_component_from_tech(cell=cell, tech=technology)

    # simulate with lumerical
    simulation = make_t3d_sim(
        device=component,
        in_port=component.ports[0],
        wavl_min=1.45,
        wavl_max=1.65,
        wavl_pts=101,
        grid_cells_per_wvl=8,
        symmetry=(
            0,
            0,
            0,
        ),  # ensure structure is symmetric across symmetry axis before triggering this!
        z_span=3,
        field_monitor_axis="y",
    )

    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    simulation.execute()


    #  visualize the results
    simulation.visualize_results()

    s_parameters = simulation.s_parameters()
    s21 = s_parameters._entries[1]
    s21.plot()
# %%
