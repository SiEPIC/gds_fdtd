"""gdsfactory S-bend on beamz — free FDTD with real physics to look at.

Same three-line setup as every engine. The S-bend's mode transition gives
real structure in the S-parameters and the field plot, and both ports face
x (beamz v1 rejects y-facing ports - finding F14: its modal normalization
on y-normal monitors is wrong by tens of dB, so the adapter fails loudly
instead of returning garbage). Runs locally on CPU in a few minutes.
"""

import os

import gdsfactory as gf

from gds_fdtd.layout.gdsfactory import from_gdsfactory
from gds_fdtd.plotting import plot_component, plot_smatrix
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology

if __name__ == "__main__":
    gf.gpdk.PDK.activate()

    here = os.path.dirname(os.path.dirname(__file__))
    tech = Technology.from_yaml(os.path.join(here, "tech.yaml"))
    component = from_gdsfactory(gf.components.bend_s(size=(10, 3)), tech)

    solver = get_solver("beamz")(
        component,
        technology=tech,
        spec=SimulationSpec(wavelength_points=5, mesh=8),
    )

    # STEP 1: geometry, ports, simulation region
    plot_component(component, spec=solver.spec, savefig=f"{component.name}_geometry.png")

    # STEP 2: offline setup (free)
    assert solver.validate() == []
    print("build:", solver.build().summary)

    # STEP 3+4: run locally (free), S-parameters, fields
    smatrix = solver.run()
    print("thru:", float(smatrix.magnitude_db(out=2, in_=1).mean()), "dB mean")
    plot_smatrix(smatrix, kind="db")[0].savefig(f"{component.name}_sparams.png", dpi=150)
    solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
