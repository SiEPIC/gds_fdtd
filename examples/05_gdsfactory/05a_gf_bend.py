"""Convert a gdsfactory component and set up any solver (gdsfactory >= 9).

The converter puts port planes ON the simulation bounds (stubs reach the PML)
and reads ports/polygons in um straight from the gf 9 API.
"""

import os

import gdsfactory as gf

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.layout.gdsfactory import from_gdsfactory
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    gf.gpdk.PDK.activate()  # gdsfactory >= 9.44 requires an active PDK

    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech.yaml"))

    component = from_gdsfactory(gf.components.bend_circular(), tech)
    print("ports:", [(p.name, p.center, p.direction) for p in component.ports])

    solver = get_solver("lumerical")(
        component,
        technology=tech,
        spec=SimulationSpec(z_min=-1.0, z_max=1.11),
    )

    # STANDARD VISUALIZATION STEP 1: geometry, ports, simulation region
    from gds_fdtd.plotting import plot_component

    plot_component(component, spec=solver.spec, savefig=f"{component.name}_geometry.png")
    artifacts = solver.build()  # offline .lsf + GDS
    print("setup script written:", artifacts.files["lsf"])
    smatrix = solver.run()  # requires a Lumerical license
    #
    # STEP 3: S-parameters   |   STEP 4: field profile
    from gds_fdtd.plotting import plot_smatrix
    plot_smatrix(smatrix, kind="db")[0].savefig(f"{component.name}_sparams.png", dpi=150)
    solver.plot_fields(axis="z", savefig=f"{component.name}_fields.png")
