#%%
"""
Simulation of a halfring device using lumerical.
@author: Mustafa Hammood
"""
import os
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lum_tools import make_sim_lum
import siepic_ebeam_pdk as pdk  # noqa: F401
from pya import Layout
from lumapi import FDTD

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(__file__), "tech.yaml")
    technology = parse_yaml_tech(tech_path)

    # load cell from siepic_ebeam_pdk
    ly = Layout()
    ly.technology_name = pdk.tech.name
    cell = ly.create_cell("ebeam_dc_halfring_straight", "EBeam", {})

    component = load_component_from_tech(cell=cell, tech=technology)

    # simulate with lumerical
    make_sim_lum(
        c=component,
        lum=FDTD(),
        wavl_min=1.45,
        wavl_max=1.65,
        wavl_pts=101,
        width_ports=4.0,
        depth_ports=4.0,
        symmetry=(0, 0, 0),
        pol="TE",
        num_modes=1,
        boundary="pml",
        mesh=3,
        run_time_factor=50,)

# %%
