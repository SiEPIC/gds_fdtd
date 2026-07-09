"""Mesh convergence on tidy3d — identical code to 02b except ENGINE.

convergence.sweep() reruns one job while stepping a SimulationSpec field and
recommends the converged value; with cache_dir, repeat sweeps are free.
Each new mesh value costs ~0.05 FC (estimate first).
"""

import os

from gds_fdtd.convergence import sweep
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.plotting import plot_component
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

ENGINE = "tidy3d"  # <- the only line that differs between 02b and 03b

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech.yaml"))
    cell, layout = load_cell(
        os.path.join(here, "devices.gds"), top_cell="si_sin_escalator_te1550"
    )
    component = load_component_from_tech(cell=cell, tech=tech)

    spec = SimulationSpec(wavelength_points=11, z_min=-1.0, z_max=1.11)
    plot_component(component, spec=spec, savefig=f"{component.name}_geometry.png")

    report = sweep(
        get_solver(ENGINE),
        component,
        tech,
        spec,
        field="mesh",
        values=[6, 8, 10],
        cache_dir=".gds_fdtd_cache",  # reruns cost nothing
    )
    print(report.summary())
    report.plot(tol_db=0.05, savefig=f"{component.name}_convergence.png")
