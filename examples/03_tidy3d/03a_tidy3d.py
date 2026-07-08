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
            mesh=10,  # the converged value from the 03b sweep
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

    # STEP 3+4: run (SPENDS FLEXCREDITS ~0.35), S-parameters, fields.
    # Curated paths: thru / cross / reflection for each polarization —
    # mode-preserving only (TE<->TM conversion is noise-floor for this device).
    smatrix = solver.run()
    paths = [
        ("opt4", "opt1", 1, 1), ("opt4", "opt1", 2, 2),  # thru TE / TM
        ("opt2", "opt1", 1, 1), ("opt2", "opt1", 2, 2),  # cross TE / TM
        ("opt1", "opt1", 1, 1), ("opt1", "opt1", 2, 2),  # reflection TE / TM
    ]
    fig, _ = plot_smatrix(smatrix, kind="db", paths=paths)
    fig.savefig(f"{component.name}_sparams.png", dpi=150, bbox_inches="tight")
    solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
    smatrix.to_dat(f"{component.name}.dat")  # -> INTERCONNECT
