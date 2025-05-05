#%% 
"""
Example of sending a component geometry to a lumerical instance.
@author: Mustafa Hammood
"""
import os
from lumapi import FDTD  # can also be mode/device
from gds_fdtd.lum_tools import to_lumerical
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.lyprocessor import load_cell

os.environ['QT_QPA_PLATFORM'] = 'xcb'  # i need to do this to get my lumerical gui to work in linux... comment out if not necessary

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_lumerical.yaml")  # note materials in yaml
    technology = parse_yaml_tech(tech_path)

    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")
    cell, layout = load_cell(file_gds, top_cell='polarizer_tm_sin')
    component = load_component_from_tech(cell=cell, tech=technology)

    to_lumerical(c=component, lum=FDTD())
    input('Proceed to terminate the GUI?')
# %%
