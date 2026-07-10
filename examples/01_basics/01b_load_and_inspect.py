"""First contact, no engine required: load a layout and inspect it.

Everything here is offline and dependency-free (KLayout + matplotlib only):
load a GDS cell against the technology, look at the detected ports, and
render the standard geometry view. Every simulation example starts exactly
like this before an engine enters the picture.
"""

import os

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.plotting import plot_component
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = Technology.from_yaml(os.path.join(here, "tech.yaml"))  # ONE tech, every engine
    cell, layout = load_cell(os.path.join(here, "devices.gds"), top_cell="crossing_te1550")
    component = load_component_from_tech(cell=cell, tech=tech)

    print(f"component {component.name!r}:")
    for p in component.ports:
        print(f"  port {p.name}: center={p.center} um, width={p.width} um, dir={p.direction} deg")
    for s in component.structures:
        print(f"  {s.role:<11} {s.name}: layer {s.layer}, z {s.z_base}..{s.z_base + s.z_span} um")

    # the standard geometry view: polygons, ports, devrec bounds, FDTD region,
    # and the port-extension stubs the solvers add through the PML
    plot_component(component, spec=SimulationSpec(), savefig=f"{component.name}_geometry.png")
