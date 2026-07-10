"""
gds_fdtd simulation toolbox.

Simulation processing module.
@author: Mustafa Hammood, 2025
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from .geometry import Component

if TYPE_CHECKING:
    pass
from .lyprocessor import (
    load_ports,
    load_region,
    load_structure,
    load_structure_from_bounds,
)

logger = logging.getLogger(__name__)


def get_material(device: dict):
    """
    Load material properties for different solvers.

    Args:
        device: Device dictionary containing material specifications

    Returns:
        dict: Material dictionary with solver-specific materials
    """
    material = {"tidy3d": None, "lum": None}

    mat_spec = device.get("material")
    if not isinstance(mat_spec, dict):
        raise ValueError(
            f"Invalid technology entry: expected a 'material' mapping, got {mat_spec!r} "
            f"in device entry {device!r}"
        )

    if "tidy3d_db" in mat_spec:
        tidy3d_db = mat_spec["tidy3d_db"]
        if not isinstance(tidy3d_db, dict) or not ("nk" in tidy3d_db or "model" in tidy3d_db):
            raise ValueError(
                "Invalid technology material: 'tidy3d_db' must be a mapping containing "
                f"'nk' or 'model'; got {tidy3d_db!r}"
            )
        try:
            material["tidy3d"] = _load_tidy3d_material(tidy3d_db)
        except ImportError:
            # engine-agnostic loading (F13): a unified technology carries
            # hints for EVERY engine; a missing tidy3d must not break
            # loading for beamz/lumerical users. The raw hint is preserved
            # below; the tidy3d adapter errors clearly if actually used.
            material["tidy3d"] = None

    if "lum_db" in mat_spec:
        lum_db = mat_spec["lum_db"]
        if not isinstance(lum_db, dict) or "model" not in lum_db:
            raise ValueError(
                "Invalid technology material: 'lum_db' must be a mapping containing "
                f"'model'; got {lum_db!r}"
            )
        material["lum"] = lum_db["model"]

    # preserve the raw neutral + per-engine hints so offline consumers
    # (grid.resolve_index, beamz, mode solving) can resolve indices without
    # engine imports — this is what makes ONE tech file serve every solver
    for key in ("nk", "rii", "tidy3d_db", "lum_db"):
        if key in mat_spec:
            material[key] = mat_spec[key]

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
        raise ImportError(
            "tidy3d is required for Tidy3D material loading. Install with: pip install tidy3d"
        )

    # Handle simple refractive index specification
    if "nk" in material_spec:
        n_value = material_spec["nk"]
        if isinstance(n_value, (int, float)):
            # Simple real refractive index
            return td.Medium(permittivity=n_value**2)
        elif isinstance(n_value, list) and len(n_value) == 2:
            # Complex refractive index [n, k]
            n, k = n_value
            return td.Medium(permittivity=(n + 1j * k) ** 2)

    # Handle material database model specification
    if "model" in material_spec:
        model_spec = material_spec["model"]
        if isinstance(model_spec, list) and len(model_spec) == 2:
            material_name, variant = model_spec
            try:
                # Try to load from Tidy3D material database
                return td.material_library[material_name][variant]
            except KeyError:
                logger.warning(f"Material {material_name}[{variant}] not found in Tidy3D library")
                logger.warning(
                    f"Available variants for {material_name}: {list(td.material_library[material_name].keys()) if material_name in td.material_library else 'Material not found'}"
                )
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
                logger.warning(f"Material {model_spec} not found in Tidy3D library")
                # Fallback to silicon
                return td.material_library["cSi"]["Li1993_293K"]

    # Fallback: return silicon if nothing else works
    logger.warning("Could not parse material specification, using default silicon")
    return td.material_library["cSi"]["Li1993_293K"]


def load_component_from_tech(cell, tech, z_span=4, z_center=None):
    # Accept either a Technology model or an already-materialized legacy dict.
    tech_dict = tech.to_solver_dict() if hasattr(tech, "to_solver_dict") else tech

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
    # Removing empty lists due to no structures existing in an input layer,
    # then flatten: Component.structures is a flat list with roles
    device_wg = [s for dev in device_wg if dev for s in dev]

    # get z_center based on structures center (minimize symmetry failures);
    # averaged PER LAYER (matching the old per-list semantics), not per polygon
    if not z_center:
        z_by_layer: dict[tuple, float] = {}
        for s in device_wg:
            z_by_layer.setdefault(tuple(s.layer), s.z_base + s.z_span / 2)
        z_center = np.average(list(z_by_layer.values()))

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
        role="superstrate",
    )
    device_sub = load_structure_from_bounds(
        bounds,
        name="Substrate",
        z_base=tech_dict["substrate"][0]["z_base"],
        z_span=tech_dict["substrate"][0]["z_span"],
        material=get_material(tech_dict["substrate"][0]),
        layer=[999, 0],  # Use a special layer for substrate
        role="substrate",
    )

    # create the device by loading the structures (flat, role-tagged)
    return Component(
        name=cell.name,
        structures=[device_sub, device_super, *device_wg],
        ports=ports,
        bounds=bounds,
    )


def from_gdsfactory(c, tech: dict, z_span: float = 4.0) -> "Component":
    """Convert a gdsfactory Component to a gds_fdtd Component.

    This is a thin delegate to gds_fdtd.layout.gdsfactory, which is
    written for the gdsfactory >= 9 API (the previous implementation here
    targeted the pre-gf-8 API and carried bugs B2/B3/B4: hardcoded polygon
    index, ports named after the component, wrong units).
    """
    from .layout.gdsfactory import from_gdsfactory as _impl

    return _impl(c, tech, z_span=z_span)
