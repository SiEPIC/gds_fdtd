"""Lithography-aware simulation: PreFab prediction -> tidy3d.

load_device() returns the component and writes derived GDS files (the input is
never modified); prefab_model= adds a <cell>_predicted.gds fabrication
prediction. Requires: pip install gds_fdtd[prefab,tidy3d] + PreFab account.
"""

import os

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell, load_device
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech.yaml"))
    gds = os.path.join(here, "devices.gds")

    # nominal geometry
    component = load_device(gds, tech=tech, top_cell="bragg_te1550", output_dir=os.getcwd())

    # lithography-predicted geometry (runs the PreFab model, writes
    # bragg_te1550_predicted.gds into output_dir):
    # component = load_device(gds, tech=tech, top_cell="bragg_te1550",
    #                         prefab_model="ANT_NanoSOI_ANF1_d10", output_dir=os.getcwd())
    # predicted_cell, _ = load_cell(os.path.join(os.getcwd(), "bragg_te1550_predicted.gds"))
    # component = load_component_from_tech(cell=predicted_cell, tech=tech)

    solver = get_solver("tidy3d")(
        component,
        technology=tech,
        spec=SimulationSpec(wavelength_start=1.4, wavelength_end=1.6, wavelength_points=101,
                            mesh=8, z_min=-1.0, z_max=1.11, symmetry=(0, 0, 1)),
    )

    # STANDARD VISUALIZATION STEP 1: geometry, ports, simulation region
    from gds_fdtd.plotting import plot_component

    plot_component(component, spec=solver.spec, savefig=f"{component.name}_geometry.png")
    solver.build()
    # smatrix = solver.run()  # SPENDS FLEXCREDITS
    #
    # STEP 3: S-parameters   |   STEP 4: field profile
    # from gds_fdtd.plotting import plot_smatrix
    # plot_smatrix(smatrix, kind="db")[0].savefig(f"{component.name}_sparams.png", dpi=150)
    # solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
