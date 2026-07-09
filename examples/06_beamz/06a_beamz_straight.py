"""gdsfactory straight through beamz — free, open-source, and fully agnostic.

beamz (Apache-2.0) runs locally on CPU/GPU via JAX: pip install gds_fdtd[beamz].
Setup is IDENTICAL to every other engine — the unified tech carries a neutral
nk per material and from_gdsfactory attaches the source component, so no
beamz-specific kwargs exist anymore.
"""

import os

import gdsfactory as gf

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.layout.gdsfactory import from_gdsfactory
from gds_fdtd.plotting import plot_component, plot_smatrix
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    gf.gpdk.PDK.activate()  # gdsfactory >= 9.44 requires an active PDK

    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech.yaml"))  # ONE tech, every engine
    component = from_gdsfactory(gf.components.straight(length=5), tech)

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

    # STEP 3+4: run locally (~2 min per port on CPU; costs nothing)
    smatrix = solver.run()
    plot_smatrix(smatrix, kind="db")[0].savefig(f"{component.name}_sparams.png", dpi=150)
    solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
