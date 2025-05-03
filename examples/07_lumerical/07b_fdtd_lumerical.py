#%% send a component to a lumerical instance
import os
from lumapi import FDTD
from gds_fdtd.lum_tools import make_sim_lum
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.lyprocessor import load_layout

os.environ['QT_QPA_PLATFORM'] = 'xcb'  # i need to do this to get my lumerical gui to work in linux... comment out if not necessary

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(__file__), "tech.yaml")  # note materials definition format in yaml
    technology = parse_yaml_tech(tech_path)

    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")
    layout = load_layout(file_gds, top_cell="cdc_sin_te1310")
    component = load_component_from_tech(ly=layout, tech=technology)

    make_sim_lum(
        c=component,
        lum=FDTD(),
        wavl_min=1.2,
        wavl_max=1.4,
        wavl_pts=101,
        width_ports=4.0,
        depth_ports=4.0,
        symmetry=(0, 0, 0),
        pol="TE",
        num_modes=1,
        boundary="pml",
        mesh=3,
        run_time_factor=50,
    )

    input('Proceed to terminate the GUI?')
# %%
