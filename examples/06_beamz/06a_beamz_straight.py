"""gdsfactory straight through beamz — a fully open-source, zero-cost FDTD run.

beamz (Apache-2.0) runs locally on CPU or GPU via JAX: pip install gds_fdtd[beamz].
BeamzSolver v1 requires the source gdsfactory component (gf_component=) and a
single device layer; refractive indices resolve from n_core=/n_clad= kwargs,
the material's refractiveindex.info reference (rii:), or tidy3d_db nk entries.
"""

import gdsfactory as gf

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.layout.gdsfactory import from_gdsfactory
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec

if __name__ == "__main__":
    gf.gpdk.PDK.activate()  # gdsfactory >= 9.44 requires an active PDK

    import os

    here = os.path.dirname(os.path.dirname(__file__))
    tech = parse_yaml_tech(os.path.join(here, "tech_tidy3d.yaml"))
    gf_component = gf.components.straight(length=5)
    component = from_gdsfactory(gf_component, tech)

    solver = get_solver("beamz")(
        component,
        technology=tech,
        spec=SimulationSpec(wavelength_points=5, mesh=8),
        gf_component=gf_component,
        n_core=3.47,  # or put an rii: entry in the technology material
    )
    assert solver.validate() == []
    print("build:", solver.build().summary)

    # ~2 min per port on CPU; free.
    # smatrix = solver.run()
    # print("S21 [dB]:", smatrix.magnitude_db(out=2, in_=1))
