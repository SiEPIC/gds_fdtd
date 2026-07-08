"""Simulate a SiEPIC EBeam PDK cell on Lumerical FDTD (local license).

Requires: pip install gds_fdtd[siepic] siepic_ebeam_pdk + Lumerical installed.
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
    tech = parse_yaml_tech(os.path.join(here, "tech_lumerical.yaml"))

    ly = Layout()
    ly.technology_name = pdk.tech.name
    cell = ly.create_cell("ebeam_dc_halfring_straight", "EBeam", {})
    component = load_component_from_tech(cell=cell, tech=tech)

    solver = get_solver("lumerical")(
        component,
        technology=tech,
        spec=SimulationSpec(wavelength_start=1.45, wavelength_end=1.65,
                            wavelength_points=101, mesh=10, z_min=-1.0, z_max=1.11),
    )
    artifacts = solver.build()  # offline setup script + GDS
    print("setup script:", artifacts.files["lsf"])
    # smatrix = solver.run()  # opens a licensed lumapi session
    # smatrix.to_dat(f"{component.name}.dat")
