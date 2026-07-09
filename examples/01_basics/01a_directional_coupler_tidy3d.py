"""Directional coupler on the tidy3d cloud via the modern Solver API.

Costs FlexCredits when run: validate/build/estimate are free and offline;
only solver.run() talks to the cloud. Requires: pip install gds_fdtd[tidy3d]
and a configured tidy3d API key.
"""

import os

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech.yaml"))
    cell, layout = load_cell(os.path.join(here, "devices.gds"), top_cell="directional_coupler_te1550")
    component = load_component_from_tech(cell=cell, tech=tech)

    solver = get_solver("tidy3d")(
        component,
        technology=tech,
        spec=SimulationSpec(
            wavelength_start=1.5,
            wavelength_end=1.6,
            wavelength_points=51,
            mesh=10,
            z_min=-1.0,
            z_max=1.11,
            symmetry=(0, 0, 1),  # TE symmetry — structure must be symmetric in z!
        ),
        workdir=os.getcwd(),
    )
    # STEP 1: geometry, ports, simulation region
    from gds_fdtd.plotting import plot_component

    plot_component(component, spec=solver.spec, savefig=f"{component.name}_geometry.png")

    # STEP 2: offline setup (free)
    print(solver.describe())
    problems = solver.validate()
    assert not problems, problems

    artifacts = solver.build()  # offline: full ModalComponentModeler + GDS export
    print("build summary:", artifacts.summary)
    print("estimate:", solver.estimate())

    # THE LINE BELOW SPENDS FLEXCREDITS — review the setup in the web UI first.
    # smatrix = solver.run()
    # smatrix.to_touchstone(f"{component.name}.s4p")
    # from gds_fdtd.plotting import plot_smatrix
    # plot_smatrix(smatrix, kind="db")

    # STANDARD VISUALIZATION STEPS 3+4 (after run):
    # from gds_fdtd.plotting import plot_smatrix
    # plot_smatrix(smatrix, kind="db")[0].savefig(f"{component.name}_sparams.png", dpi=150)
    # solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
