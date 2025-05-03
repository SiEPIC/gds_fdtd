# %%
"""
@author: Mustafa Hammood
Build a tidy3d simulation from gdsfactory cell.
"""
import gds_fdtd as gtd
import gdsfactory as gf
import os
from gds_fdtd.simprocessor import from_gdsfactory, make_sim


if __name__ == '__main__':
    tech_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech.yaml")
    technology = gtd.core.parse_yaml_tech(tech_file_path)

    device = from_gdsfactory(c=gf.components.bend_circular(), tech=technology)
    simulation = make_sim(
        device=device,
        in_port=device.ports[0],
        z_span=2.,
        symmetry=(0, 0, 1),
        mode_index=0,
        grid_cells_per_wvl=6,
    )
    # %%
    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    #simulation.execute()
    #  visualize the results
    #simulation.visualize_results()
# %%
