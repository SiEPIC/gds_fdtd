"""
gds_fdtd simulation toolbox.

Simulation processing module.
@author: Mustafa Hammood, 2025
"""

import tidy3d as td
import numpy as np
import matplotlib.pyplot as plt
from .core import structure, region, port, component
from .t3d_tools import sim_tidy3d
from .lyprocessor import (
    load_structure,
    load_region,
    load_ports,
    load_structure_from_bounds,
    dilate,
    dilate_1d,
)


def make_source(
    port: port,
    num_modes: int = 1,
    mode_index: int = 0,
    width: float = 3.0,
    depth: float = 2.0,
    freq0: float = 2e14,
    num_freqs: int = 5,
    fwidth: float = 1e13,
    buffer: float = -0.2,
):
    """Create a simulation mode source on an input port.

    Args:
        port (port object): Port to add source to.
        mode_index (int, optional): Mode index to launch. Defaults to 0.
        num_modes (int, optional): Number of modes to launch in the source. Defaults to 1.
        width (int, optional): Source width. Defaults to 3 microns.
        depth (int, optional): Source depth. Defaults to 2 microns.
        freq0 (_type_, optional): Source's centre frequency. Defaults to 2e14 hz.
        num_freqs (int, optional): Frequency evaluation mode ports. Defaults to 5.
        fwidth (_type_, optional): Source's bandwidth. Defaults to 1e13 hz.
        buffer (float, optional): Distance between edge of simulation and source. Defaults to 0.1 microns.

    Returns:
        td.ModeSource: Generated source.
    """
    import tidy3d as td

    if port.direction == 0:
        x_buffer = -buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 180:
        x_buffer = buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 90:
        x_buffer = 0
        y_buffer = -buffer
        size = [width, 0, depth]
    elif port.direction == 270:
        x_buffer = 0
        y_buffer = buffer
        size = [width, 0, depth]
    if port.direction in [180, 270]:
        direction = "+"
    else:
        direction = "-"
    msource = td.ModeSource(
        center=[port.x + x_buffer, port.y + y_buffer, port.z],
        size=size,
        direction=direction,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        mode_spec=td.ModeSpec(num_modes=num_modes),
        mode_index=mode_index,
        num_freqs=num_freqs,
        name=f"msource_{port.name}_idx{mode_index}",
    )
    return msource


def make_structures(device, buffer: float=4.):
    """Create a tidy3d structure object from a device objcet.

    Args:
        device (device object): Device to create the structure from.
        buffer (int, optional): Extension of ports beyond simulation region . Defaults to 2 microns.

    Returns:
        list: list of structures generated from the device.
    """
    import tidy3d as td
    import numpy as np

    # TODO find a better way to handle material..
    
    # TODO fix box tox handling here
    structures = []
    for s in device.structures:
        if type(s) == list:
            for i in s:
                if i.z_span < 0:
                    bounds = (i.z_base + i.z_span, i.z_base)
                else:
                    bounds = (i.z_base, i.z_base + i.z_span)
                structures.append(
                    td.Structure(
                        geometry=td.PolySlab(
                            vertices=i.polygon,
                            slab_bounds=bounds,
                            axis=2,
                            sidewall_angle=(90 - i.sidewall_angle) * (np.pi / 180),
                        ),
                        medium=i.material["tidy3d"] if isinstance(i.material, dict) else i.material,
                        name=i.name,
                    )
                )
        else:
            if s.z_span < 0:
                bounds = (s.z_base + s.z_span, s.z_base)
            else:
                bounds = (s.z_base, s.z_base + s.z_span)
            print(s.material["tidy3d"] if isinstance(s.material, dict) else s.material)
            structures.append(
                td.Structure(
                    geometry=td.PolySlab(
                        vertices=s.polygon,
                        slab_bounds=bounds,
                        axis=2,
                        sidewall_angle=(90 - s.sidewall_angle) * (np.pi / 180),
                    ),
                    medium=s.material["tidy3d"] if isinstance(s.material, dict) else s.material,
                    name=s.name,
                )
            )

    # extend ports beyond sim region
    for p in device.ports:
        structures.append(
            td.Structure(
                geometry=td.PolySlab(
                    vertices=p.polygon_extension(buffer=buffer),
                    slab_bounds=(
                        p.center[2] - p.height / 2,
                        p.center[2] + p.height / 2,
                    ),
                    axis=2,
                    sidewall_angle=(90 - device.structures[0].sidewall_angle)
                    * (np.pi / 180),
                ),
                medium=p.material["tidy3d"] if isinstance(p.material, dict) else p.material,
                name=f"port_{p.name}",
            )
        )
    return structures


def make_port_monitor(port, freqs=2e14, num_modes=1, buffer=-0.1, depth=2, width=3):
    """
    Create mode monitor object for a given port.

    Args:
        port (port object): Port to create the monitor for.
        freqs (float, optional): Mode monitor's central frequency. Defaults to 2e14 hz.
        num_modes (int, optional): Number of modes to be captured. Defaults to 1.
        buffer (float, optional): Distance between monitor and port location. Defaults to 0.2 microns.
        depth (int, optional): Monitor's depth. Defaults to 2 microns.
        width (int, optional): Monitors width. Defaults to 3 microns.

    Returns:
        monitor: Generated ModeMonitor object.
    """

    if port.direction == 0:
        x_buffer = -buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 180:
        x_buffer = buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 90:
        x_buffer = 0
        y_buffer = -buffer
        size = [width, 0, depth]
    elif port.direction == 270:
        x_buffer = 0
        y_buffer = buffer
        size = [width, 0, depth]
    # mode monitor
    monitor = td.ModeMonitor(
        center=[port.x + x_buffer, port.y + y_buffer, port.z],
        size=size,
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=num_modes),
        name=port.name,
    )

    return monitor


def make_field_monitor(device, freqs=2e14, axis="z", z_center=None):
    """Make a field monitor for an input device

    Args:
        device (device object): Device to create field monitor for.
        freqs (float, optional): _description_. Defaults to 2e14.
        z_center (float, optional): Z center for field monitor. Defaults to None.
        axis (string, optional): Field monitor's axis. Valid options are 'x', 'y', 'z' Defaults to 'z'.

    Returns:
        FieldMonitor: Generated Tidy3D field monitor object
    """
    import numpy as np

    # identify a device field z_center if None
    if z_center is None:
        z_center = []
        for s in device.structures:
            if type(s) == list:  # i identify non sub/superstrate if s is a list
                s = s[0]
                z_center.append(s.z_base + s.z_span / 2)
        z_center = np.average(z_center)
    if axis == "z":
        center = [0, 0, z_center]
        size = [td.inf, td.inf, 0]
    elif axis == "y":
        center = [0, 0, z_center]
        size = [td.inf, 0, td.inf]
    elif axis == "x":
        center = [0, 0, z_center]
        size = [0, td.inf, td.inf]
    else:
        Exception("Invalid axis for field monitor. Valid selections are 'x', 'y', 'z'.")
    return td.FieldMonitor(
        center=center,
        size=size,
        freqs=freqs,
        name=f"{axis}_field",
    )

def make_t3d_sim(
    device,
    wavl_min: float = 1.45,
    wavl_max: float = 1.65,
    wavl_pts: int = 101,
    width_ports: float = 3.0,
    depth_ports: float = 2.0,
    symmetry: tuple[int, int, int] = (0, 0, 0),
    num_freqs: int = 5,
    in_port: port | str | None = None,
    mode_index: list | int = 0,
    num_modes: int = 1,
    boundary: td.BoundarySpec = td.BoundarySpec.all_sides(boundary=td.PML()),
    grid_cells_per_wvl: int = 15,
    run_time_factor: float = 50,
    z_span: float | None = None,
    field_monitor_axis: str | None = None,
    visualize: bool = True,
):
    """Generate a single port excitation simulation.

    Args:
        device (device object): Device to simulate.
        wavl_min (float, optional): Start wavelength. Defaults to 1.45 microns.
        wavl_max (float, optional): End wavelength. Defaults to 1.65 microns.
        wavl_pts (int, optional): Number of wavelength evaluation pts. Defaults to 101.
        width_ports (int, optional): Width of source and monitors. Defaults to 3 microns.
        depth_ports (int, optional): Depth of source and monitors. Defaults to 2 microns.
        symmetry (tuple, optional): Enforcing symmetry along axes. Defaults to (0, 0, 0).
        num_freqs (int, optional): Number of source's frequency mode evaluation pts. Defaults to 5 microns.
        in_port (port object, optional): Input port. Defaults to None.
        mode_index(list, optional): Mode index to inject in source. Defaults to [0].
        num_modes(int, optional): Number of source's and monitors modes. Defaults to 1.
        boundary (td.BonudarySpec object, optional): Configure boundary conditions. Defaults to td.BoundarySpec.all_sides(boundary=td.PML()).
        grid_cells_per_wvl (int, optional): Mesh settings, grid cells per wavelength. Defaults to 15.
        run_time_factor (int, optional): Runtime multiplier factor. Set larger if runtime is insufficient. Defaults to 50.
        z_span (float, optional): Simulation's depth. Defaults to None.
        field_monitor_axis (str, optional): Flag to create a field monitor. Options are 'x', 'y', 'z', or none. Defaults to None.
        visualize (bool, optional): Simulation visualization flag. Defaults to True.

    Returns:
        simulation: Generated simulation instance.
    """

    # if no input port defined, use first as default
    if in_port is None:
        in_port = [device.ports[0]]
    if not isinstance(in_port, list):
        in_port = [in_port]
    if in_port == "all":
        in_port = device.ports[0]

    if not isinstance(mode_index, list):
        mode_index = [mode_index]

    lda0 = (wavl_max + wavl_min) / 2
    freq0 = td.C_0 / lda0
    freqs = td.C_0 / np.linspace(wavl_min, wavl_max, wavl_pts)
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

    # define structures from device
    structures = make_structures(device)

    # define monitors
    monitors = []
    for p in device.ports:
        monitors.append(
            make_port_monitor(
                p,
                freqs=freqs,
                depth=depth_ports,
                width=width_ports,
                num_modes=num_modes,
            )
        )

    # make field monitor
    # TODO: handle mode index cases in making field monitor
    if field_monitor_axis is not None:
        monitors.append(
            make_field_monitor(device, freqs=freqs, axis=field_monitor_axis)
        )
    # simulation domain size (in microns)
    if z_span == None:
        sim_size = [device.bounds.x_span, device.bounds.y_span, device.bounds.z_span]
    else:
        sim_size = [device.bounds.x_span, device.bounds.y_span, z_span]
    run_time = (
        run_time_factor * max(sim_size) / td.C_0
    )  # 85/fwidth  # sim. time in secs

    """
    define sim jobs: create source on a given port, for each mode index
    i.e., TE and TM mode indices on 3 input ports would result in 6 sources (simulation jobs)
    some thoughts:
    when jobs are defined and sent to api, we lose the link between the simulation object and tidy3d simulation
    we maintain some data by turning the job into a dictionary object
    TODO: in the future, sim_job should be an object, and simulation should be a partial object of sim_job
    """
    sim_jobs = []
    for m in mode_index:
        for p in in_port:
            source = make_source(
                port=p,
                depth=depth_ports,
                width=width_ports,
                freq0=freq0,
                num_freqs=num_freqs,
                fwidth=fwidth,
                num_modes=num_modes,
                mode_index=m,
            )
            sim = {}
            sim["name"] = f"{device.name}_{p.name}_idx{m}"
            sim["source"] = source
            sim["in_port"] = p
            sim["num_modes"] = num_modes
            sim["sim"] = td.Simulation(
                size=sim_size,
                grid_spec=td.GridSpec.auto(
                    min_steps_per_wvl=grid_cells_per_wvl, wavelength=lda0
                ),
                structures=structures,
                sources=[source],
                monitors=monitors,
                run_time=run_time,
                boundary_spec=boundary,
                center=(
                    device.bounds.x_center,
                    device.bounds.y_center,
                    device.bounds.z_center,
                ),
                symmetry=symmetry,
            )
            sim_jobs.append(sim)

    # initialize the simulation
    simulation = sim_tidy3d(
        in_port=in_port,
        wavl_max=wavl_max,
        wavl_min=wavl_min,
        wavl_pts=wavl_pts,
        device=device,
        sim_jobs=sim_jobs,
    )

    if visualize:
        for sim_job in simulation.sim_jobs:
            sim = sim_job["sim"]
            for m in sim.monitors:
                m.help()

        source.source_time.plot(np.linspace(0, run_time, 1001))
        plt.show()

        # visualize geometry
        for sim_job in simulation.sim_jobs:
            sim = sim_job["sim"]
            gridsize = (3, 2)
            fig = plt.figure(figsize=(12, 8))
            ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
            ax2 = plt.subplot2grid(gridsize, (2, 0))
            ax3 = plt.subplot2grid(gridsize, (2, 1))

            sim.plot(z=device.bounds.z_center, ax=ax1)
            sim.plot(x=0.0, ax=ax2)
            sim.plot(x=device.bounds.x_center, ax=ax3)
            ax1.set_title(sim_job["name"])
            plt.show()
    return simulation


def get_material(device: dict):
    """
    TODO: find a better way to handle this
    maybe use opticalmaterialspy as a universal base?
    """
    material = {'tidy3d': None, 'lum': None}
    if "tidy3d_db" in device["material"]:
        # load material from tidy3d material database, format [material, model]
        if "model" in device["material"]["tidy3d_db"]:
            mat_tidy3d = td.material_library[device["material"]["tidy3d_db"]["model"][0]][device["material"]["tidy3d_db"]["model"][1]]
        # load tidy3d constant index material, format: refractive index
        elif "nk" in device["material"]["tidy3d_db"]:
            mat_tidy3d = td.Medium(permittivity=device["material"]["tidy3d_db"]["nk"] ** 2)
        material['tidy3d'] = mat_tidy3d

    if "lum_db" in device["material"]:
        # load material from lumerical material database, format: material model name
        if "model" in device["material"]["lum_db"]:
            mat_lum = device["material"]["lum_db"]["model"]
        material['lum'] = mat_lum

    return material


def load_component_from_tech(cell, tech, z_span=4, z_center=None):
    # load the structures in the device
    device_wg = []
    for idx, d in enumerate(tech["device"]):
        device_wg.append(
            load_structure(
                cell,
                name=f"dev_{idx}",
                layer=d["layer"],
                z_base=d["z_base"],
                z_span=d["z_span"],
                material=get_material(d),
            )
        )
    # Removing empty lists due to no structures existing in an input layer
    device_wg = [dev for dev in device_wg if dev]

    # get z_center based on structures center (minimize symmetry failures)
    if not z_center:
        z_center = np.average([d[0].z_base + d[0].z_span / 2 for d in device_wg])

    # load all the ports in the device and (optional) initialize each to have a center
    ports = load_ports(cell, layer=tech["pinrec"][0]["layer"])
    # load the device simulation region
    bounds = load_region(
        cell, layer=tech["devrec"][0]["layer"], z_center=z_center, z_span=z_span
    )

    # make the superstrate and substrate based on device bounds
    # this information isn't typically captured in a 2D layer stack
    device_super = load_structure_from_bounds(
        bounds,
        name="Superstrate",
        z_base=tech["superstrate"][0]["z_base"],
        z_span=tech["superstrate"][0]["z_span"],
        material=get_material(tech["superstrate"][0]),
    )
    device_sub = load_structure_from_bounds(
        bounds,
        name="Subtrate",
        z_base=tech["substrate"][0]["z_base"],
        z_span=tech["substrate"][0]["z_span"],
        material=get_material(tech["substrate"][0]),
    )

    # create the device by loading the structures
    return component(
        name=cell.name,
        structures=[device_sub, device_super] + device_wg,
        ports=ports,
        bounds=bounds,
    )

def build_sim_from_tech(tech: dict, cell, in_port=0, **kwargs):

    z_span = kwargs.pop("z_span", 4)  # Default value 4 if z_span is not provided

    device = load_component_from_tech(cell=cell, tech=tech, z_span=z_span)

    if isinstance(in_port, int):
        return make_t3d_sim(
            device=device,
            in_port=device.ports[in_port],
            z_span=z_span,
            **kwargs,
        )
    elif in_port == "all":
        return make_t3d_sim(
            device=device,
            in_port=device.ports[:],
            z_span=z_span,
            **kwargs,
        )


def from_gdsfactory(c: 'gf.Component', tech: dict, z_span: float = 4.) -> 'component':
    """Convert gdsfactory Component to a component.

    Args:
        c (gf.Component): gdsfactory component.
        tech (dict): dictionary technology stack (can be parsed from yaml) 
        z_span (float, optional): z bounds of the device (can be used to override simulation settings..). Defaults to 4..

    Returns:
        component: parsed gdsfactory component.
    """
    try:
        import gdsfactory as gf
    except ImportError:
        raise ImportError("gdsfactory is not installed. Please install it using 'pip install .[gdsfactory]'")

    device_wg = []
    ports = []

    # for each layer in the device
    for idx, layer in enumerate(c.get_polygons()):
        l = c.extract(layers={layer})       

        for i, s in enumerate(l.get_polygons()):
            name = f"poly_{idx}_{i}"
            device_wg.append(
                structure(
                    name=name,
                    polygon=l.get_polygons()[1],
                    z_base=tech["device"][idx]["z_base"],
                    z_span=tech["device"][idx]["z_span"],
                    material=get_material(tech["device"][idx]),
                    sidewall_angle=tech["device"][idx]["sidewall_angle"],
                )
            )

        # get device ports
        for p in c.ports:
            if p.layer == layer:
                z_pos = (
                    tech["device"][idx]["z_base"] + tech["device"][idx]["z_span"] / 2
                )
                ports.append(
                    port(
                        name=c.name,
                        center=list(p.center) + [z_pos],
                        width=p.width,
                        direction=p.orientation,
                    )
                )

    # get z_center based on structures center (minimize symmetry failures)
    z_center = np.average([d.z_base + d.z_span / 2 for d in device_wg])

    # expand bbox region to account for evanescent field
    def min_dim(square):
        x_dim = abs(square[0][0] - square[1][0])
        y_dim = abs(square[0][1] - square[1][1])
        if x_dim < y_dim:
            return "x"
        elif x_dim > y_dim:
            return "y"
        else:
            return "xy"

    # expand the bbox region by 1.3 um (on each side) on the smallest dimension
    bbox_poly = [[c.bbox().p1.x, c.bbox().p1.y], [c.bbox().p2.x, c.bbox().p2.y]]
    bbox = dilate_1d(bbox_poly, extension=0, dim=min_dim(bbox_poly))
    bbox_dilated = dilate(bbox, extension=1.9)
    bounds = region(vertices=bbox_dilated, z_center=z_center, z_span=z_span)

    # make the superstrate and substrate based on device bounds
    # this information isn't typically captured in a 2D layer stack
    device_super = load_structure_from_bounds(
        bounds,
        name="Superstrate",
        z_base=tech["superstrate"][0]["z_base"],
        z_span=tech["superstrate"][0]["z_span"],
        material=get_material(tech["superstrate"][0]),
    )
    device_sub = load_structure_from_bounds(
        bounds,
        name="Subtrate",
        z_base=tech["substrate"][0]["z_base"],
        z_span=tech["substrate"][0]["z_span"],
        material=get_material(tech["substrate"][0]),
    )

    # create the device by loading the structures
    return component(
        name=c.name,
        structures=[device_sub, device_super] + [device_wg],
        ports=ports,
        bounds=bounds,
    )
