"""Simulate a SiEPIC EBeam PDK cell on tidy3d.

Requires: pip install gds_fdtd[siepic,tidy3d] siepic_ebeam_pdk.
"""

import os

import siepic_ebeam_pdk as pdk  # noqa: F401  (registers the PDK technology)
from pya import Layout

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech_tidy3d.yaml"))

    ly = Layout()
    ly.technology_name = pdk.tech.name
    cell = ly.create_cell("ebeam_dc_halfring_straight", "EBeam", {})
    component = load_component_from_tech(cell=cell, tech=tech)

    solver = get_solver("tidy3d")(
        component,
        technology=tech,
        spec=SimulationSpec(wavelength_start=1.45, wavelength_end=1.65,
                            wavelength_points=101, mesh=8, z_min=-1.0, z_max=1.11),
    )
    solver.build()
    # smatrix = solver.run()  # SPENDS FLEXCREDITS
