#%% send a component to a lumerical instance
import os
import lumapi
from gds_fdtd.lum_tools import to_lumerical, make_sim_lum
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.lyprocessor import load_layout

os.environ['QT_QPA_PLATFORM'] = 'xcb'  # i need to do this to get my lumerical gui to work in linux... comment out if not necessary

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(__file__), "tech.yaml")  # note materials definition format in yaml
    technology = parse_yaml_tech(tech_path)

    file_gds = os.path.join(os.path.dirname(__file__), "wg.gds")
    layout = load_layout(file_gds)
    component = load_component_from_tech(ly=layout, tech=technology)

    fdtd = lumapi.FDTD()  # can also be mode/device
    print(type(fdtd))

    to_lumerical(c=component, lum=fdtd)

    make_sim_lum(
        c=component,
        lum=fdtd,
        wavl_min=1.5,
        wavl_max=1.6,
        wavl_pts=101,
        width_ports=3.0,
        depth_ports=2.0,
        symmetry=(0, 0, 0),
        pol="TE",
        num_modes=1,
        boundary="pml",
        mesh=1,
        run_time_factor=50,
    )

    input('Proceed to terminate the GUI?')
# %%
