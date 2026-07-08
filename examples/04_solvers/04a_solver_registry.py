"""The solver-agnostic workflow: one component, any engine.

Everything here is OFFLINE and free — validate/build/estimate never spend
credits, licenses, or GPU time. Swap engines by changing one string.
"""

import os

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import available_solvers, get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    print("registered solvers:", available_solvers())

    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech_lumerical.yaml"))
    cell, layout = load_cell(os.path.join(here, "devices.gds"), top_cell="crossing_te1550")
    component = load_component_from_tech(cell=cell, tech=tech)

    spec = SimulationSpec(wavelength_points=51, mesh=8, z_min=-1.0, z_max=1.11)

    solver = get_solver("lumerical")(component, technology=tech, spec=spec)
    print(solver.describe())
    artifacts = solver.build()  # generates the complete .lsf setup script offline
    print("generated setup script:", artifacts.files["lsf"])
    print("estimate:", solver.estimate())

    # run() opens a licensed lumapi session:
    # smatrix = solver.run()
