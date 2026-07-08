"""Waveguide crossing on the tidy3d cloud via the modern Solver API.

Same flow as 01a on the crossing cell; shows S-parameter access by port name.
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
    cell, layout = load_cell(os.path.join(here, "devices.gds"), top_cell="crossing_te1550")
    component = load_component_from_tech(cell=cell, tech=tech)

    solver = get_solver("tidy3d")(
        component,
        technology=tech,
        spec=SimulationSpec(wavelength_points=51, mesh=10, z_min=-1.0, z_max=1.11),
        workdir=os.getcwd(),
    )

    # STANDARD VISUALIZATION STEP 1: geometry, ports, simulation region
    from gds_fdtd.plotting import plot_component

    plot_component(component, spec=solver.spec, savefig=f"{component.name}_geometry.png")

    # STEP 2: offline setup (free)
    assert solver.validate() == []
    solver.build()

    # THE LINES BELOW SPEND FLEXCREDITS (~0.3 FC for this example).
    # smatrix = solver.run()
    #
    # STEP 3: S-parameters
    # from gds_fdtd.plotting import plot_smatrix
    # fig, _ = plot_smatrix(smatrix, kind="db")
    # fig.savefig(f"{component.name}_sparams.png", dpi=150)
    # print("thru:", smatrix.magnitude_db(out="opt4", in_="opt1").max(), "dB")
    #
    # STEP 4: field profile (z-plane, first excitation)
    # solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
