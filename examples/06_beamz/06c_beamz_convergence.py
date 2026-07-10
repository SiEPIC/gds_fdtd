"""Mesh convergence on beamz — the FREE way to learn the convergence workflow.

Identical code to 02b/03b except the engine string, but every sweep point
costs nothing: run it, change the values, run it again (cached points are
never recomputed). Do your convergence study here, then spend credits or
license time only at the converged mesh.
"""

import os

import gdsfactory as gf

from gds_fdtd.convergence import sweep
from gds_fdtd.layout.gdsfactory import from_gdsfactory
from gds_fdtd.plotting import plot_component
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology

ENGINE = "beamz"  # free — compare with 02b (lumerical) and 03b (tidy3d)

if __name__ == "__main__":
    gf.gpdk.PDK.activate()

    here = os.path.dirname(os.path.dirname(__file__))
    tech = Technology.from_yaml(os.path.join(here, "tech.yaml"))
    component = from_gdsfactory(gf.components.straight(length=5), tech)

    spec = SimulationSpec(wavelength_points=5)
    plot_component(component, spec=spec, savefig=f"{component.name}_geometry.png")

    report = sweep(
        get_solver(ENGINE),
        component,
        tech,
        spec,
        field="mesh",
        values=[8, 10, 12],  # below mesh 8 the discretization is too coarse to be meaningful
        cache_dir=".gds_fdtd_cache",
    )
    print(report.summary())
    print("recommended mesh:", report.recommend(tol_db=0.05))
    report.plot(tol_db=0.05, savefig=f"{component.name}_convergence.png")
