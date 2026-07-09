"""Convergence sweeps + simulation caching (WP5.5).

``sweep`` reruns one job while stepping a single SimulationSpec field and
reports how much the S-matrix still moves between successive values — the
principled way to pick a mesh instead of guessing. With ``cache_dir`` every
(job, value) point runs AT MOST ONCE ever: rerunning this script (or
sweeping a superset of values later) only pays for genuinely new points.

The engine here is tidy3d (cloud credits!) — the sweep below costs roughly
one small simulation per mesh value on the first run and is free afterwards.
"""

import os

from gds_fdtd.convergence import sweep
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech.yaml"))
    cell, layout = load_cell(os.path.join(here, "devices.gds"), top_cell="si_sin_escalator_te1550")
    component = load_component_from_tech(cell=cell, tech=tech)

    spec = SimulationSpec(wavelength_points=11, z_min=-1.0, z_max=1.11)

    # STANDARD VISUALIZATION STEP 1: geometry, ports, simulation region
    from gds_fdtd.plotting import plot_component

    plot_component(component, spec=spec, savefig=f"{component.name}_geometry.png")

    # each value costs one run() on the FIRST execution; cached afterwards
    report = sweep(
        get_solver("tidy3d"),
        component,
        tech,
        spec,
        field="mesh",
        values=[6, 8, 10],
        cache_dir=".gds_fdtd_cache",
    )
    print(report.summary())
    print("recommended mesh:", report.recommend(tol_db=0.05))
    report.plot(tol_db=0.05, savefig=f"{component.name}_convergence.png")

    # the converged result is just the last S-matrix of the sweep
    from gds_fdtd.plotting import plot_smatrix

    fig, _ = plot_smatrix(report.smatrices[-1], kind="db")
    fig.savefig(f"{component.name}_sparams.png", dpi=150, bbox_inches="tight")
