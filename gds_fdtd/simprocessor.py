"""
gds_fdtd simulation toolbox.

Simulation processing module.
@author: Mustafa Hammood, 2025
"""

import numpy as np
from .core import structure, region, port, component
from .lyprocessor import (
    load_structure,
    load_region,
    load_ports,
    load_structure_from_bounds,
    dilate,
    dilate_1d,
)


def get_material(device: dict):
    """
    Load material properties for different solvers.
    
    Args:
        device: Device dictionary containing material specifications
        
    Returns:
        dict: Material dictionary with solver-specific materials
    """
    material = {'tidy3d': None, 'lum': None}
    
    if "tidy3d_db" in device["material"]:
        material['tidy3d'] = _load_tidy3d_material(device["material"]["tidy3d_db"])

    if "lum_db" in device["material"]:
        # load material from lumerical material database, format: material model name
        if "model" in device["material"]["lum_db"]:
            mat_lum = device["material"]["lum_db"]["model"]
        material['lum'] = mat_lum

    return material


def _load_tidy3d_material(material_spec: dict):
    """
    Load Tidy3D material from specification.
    
    Args:
        material_spec: Material specification dictionary
        
    Returns:
        tidy3d.Medium: Tidy3D material object
    """
    try:
        import tidy3d as td
    except ImportError:
        raise ImportError("tidy3d is required for Tidy3D material loading. Install with: pip install tidy3d")
    
    # Handle simple refractive index specification
    if "nk" in material_spec:
        n_value = material_spec["nk"]
        if isinstance(n_value, (int, float)):
            # Simple real refractive index
            return td.Medium(permittivity=n_value**2)
        elif isinstance(n_value, list) and len(n_value) == 2:
            # Complex refractive index [n, k]
            n, k = n_value
            return td.Medium(permittivity=(n + 1j*k)**2)
    
    # Handle material database model specification
    if "model" in material_spec:
        model_spec = material_spec["model"]
        if isinstance(model_spec, list) and len(model_spec) == 2:
            material_name, variant = model_spec
            try:
                # Try to load from Tidy3D material database
                return td.material_library[material_name][variant]
            except KeyError:
                print(f"Warning: Material {material_name}[{variant}] not found in Tidy3D library")
                print(f"Available variants for {material_name}: {list(td.material_library[material_name].keys()) if material_name in td.material_library else 'Material not found'}")
                # Fallback to silicon with warning
                return td.material_library["cSi"]["Li1993_293K"]
        elif isinstance(model_spec, str):
            # Single material name, try to get default variant
            try:
                material_dict = td.material_library[model_spec]
                # Get first available variant as default
                first_variant = next(iter(material_dict.keys()))
                return material_dict[first_variant]
            except (KeyError, StopIteration):
                print(f"Warning: Material {model_spec} not found in Tidy3D library")
                # Fallback to silicon
                return td.material_library["cSi"]["Li1993_293K"]
    
    # Fallback: return silicon if nothing else works
    print("Warning: Could not parse material specification, using default silicon")
    return td.material_library["cSi"]["Li1993_293K"]


def load_component_from_tech(cell, tech, z_span=4, z_center=None):
    # Convert technology object to dict if needed (for backward compatibility)
    if hasattr(tech, 'to_dict'):
        tech_dict = tech.to_dict()
    else:
        tech_dict = tech
    
    # load the structures in the device
    device_wg = []
    for idx, d in enumerate(tech_dict["device"]):
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
    ports = load_ports(cell, layer=tech_dict["pinrec"][0]["layer"])
    # load the device simulation region
    bounds = load_region(
        cell, layer=tech_dict["devrec"][0]["layer"], z_center=z_center, z_span=z_span
    )

    # make the superstrate and substrate based on device bounds
    # this information isn't typically captured in a 2D layer stack
    device_super = load_structure_from_bounds(
        bounds,
        name="Superstrate",
        z_base=tech_dict["superstrate"][0]["z_base"],
        z_span=tech_dict["superstrate"][0]["z_span"],
        material=get_material(tech_dict["superstrate"][0]),
        layer=[999, 1],  # Use a special layer for superstrate
    )
    device_sub = load_structure_from_bounds(
        bounds,
        name="Subtrate",
        z_base=tech_dict["substrate"][0]["z_base"],
        z_span=tech_dict["substrate"][0]["z_span"],
        material=get_material(tech_dict["substrate"][0]),
        layer=[999, 0],  # Use a special layer for substrate
    )

    # create the device by loading the structures
    return component(
        name=cell.name,
        structures=[device_sub, device_super] + device_wg,
        ports=ports,
        bounds=bounds,
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

    # Convert technology object to dict if needed (for backward compatibility)
    if hasattr(tech, 'to_dict'):
        tech_dict = tech.to_dict()
    else:
        tech_dict = tech

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
                    z_base=tech_dict["device"][idx]["z_base"],
                    z_span=tech_dict["device"][idx]["z_span"],
                    material=get_material(tech_dict["device"][idx]),
                    sidewall_angle=tech_dict["device"][idx]["sidewall_angle"],
                    layer=list(layer),  # Convert layer tuple to list for GDS export
                )
            )

        # get device ports
        for p in c.ports:
            if p.layer == layer:
                z_pos = (
                    tech_dict["device"][idx]["z_base"] + tech_dict["device"][idx]["z_span"] / 2
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
        z_base=tech_dict["superstrate"][0]["z_base"],
        z_span=tech_dict["superstrate"][0]["z_span"],
        material=get_material(tech_dict["superstrate"][0]),
        layer=[999, 1],  # Use a special layer for superstrate
    )
    device_sub = load_structure_from_bounds(
        bounds,
        name="Subtrate",
        z_base=tech_dict["substrate"][0]["z_base"],
        z_span=tech_dict["substrate"][0]["z_span"],
        material=get_material(tech_dict["substrate"][0]),
        layer=[999, 0],  # Use a special layer for substrate
    )

    # create the device by loading the structures
    return component(
        name=c.name,
        structures=[device_sub, device_super] + [device_wg],
        ports=ports,
        bounds=bounds,
    )
