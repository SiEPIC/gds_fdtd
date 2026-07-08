"""Crossing on the tidy3d cloud — the standard agnostic flow.

Identical setup to 02a with one changed string: get_solver("tidy3d").
validate/build/estimate are free and offline; ONLY solver.run() spends
FlexCredits (~0.3 FC at these settings — check estimate + the web UI first).
"""

import os

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.plotting import plot_component, plot_smatrix
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech.yaml"))  # ONE tech, every engine
    cell, layout = load_cell(os.path.join(here, "devices.gds"), top_cell="crossing_te1550")
    component = load_component_from_tech(cell=cell, tech=tech)

    solver = get_solver("tidy3d")(
        component,
        technology=tech,
        spec=SimulationSpec(
            wavelength_points=51,
            mesh=6,
            z_min=-1.0,
            z_max=1.11,
            modes=(1, 2),  # TE + TM
        ),
        workdir=os.getcwd(),
    )

    # STEP 1: geometry, ports, simulation region
    plot_component(component, spec=solver.spec, savefig=f"{component.name}_geometry.png")

    # STEP 2: offline setup (free)
    print(solver.describe())
    assert solver.validate() == []
    print("build summary:", solver.build().summary)
    print("estimate:", solver.estimate())

    # STEP 3+4: run (SPENDS FLEXCREDITS), S-parameters, fields
    smatrix = solver.run()
    plot_smatrix(smatrix, kind="db")[0].savefig(f"{component.name}_sparams.png", dpi=150)
    solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
    smatrix.to_dat(f"{component.name}.dat")  # -> INTERCONNECT
