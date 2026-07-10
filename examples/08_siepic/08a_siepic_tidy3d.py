"""Simulate a SiEPIC EBeam PDK cell on tidy3d.

Requires: pip install gds_fdtd[siepic,tidy3d] siepic_ebeam_pdk.
"""

import os

import siepic_ebeam_pdk as pdk  # noqa: F401  (registers the PDK technology)
from pya import Layout

from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = Technology.from_yaml(os.path.join(here, "tech.yaml"))

    ly = Layout()
    ly.technology_name = pdk.tech.name
    cell = ly.create_cell("ebeam_dc_halfring_straight", "EBeam", {})
    component = load_component_from_tech(cell=cell, tech=tech)

    solver = get_solver("tidy3d")(
        component,
        technology=tech,
        spec=SimulationSpec(
            wavelength_start=1.45,
            wavelength_end=1.65,
            wavelength_points=101,
            mesh=8,
            z_min=-1.0,
            z_max=1.11,
        ),
    )

    # STANDARD VISUALIZATION STEP 1: geometry, ports, simulation region
    from gds_fdtd.plotting import plot_component

    plot_component(component, spec=solver.spec, savefig=f"{component.name}_geometry.png")
    solver.build()
    smatrix = solver.run()  # SPENDS FLEXCREDITS
    #
    # STEP 3: S-parameters   |   STEP 4: field profile
    # from gds_fdtd.plotting import plot_smatrix
    # plot_smatrix(smatrix, kind="db")[0].savefig(f"{component.name}_sparams.png", dpi=150)
    # solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
