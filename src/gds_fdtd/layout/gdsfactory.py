"""
gds_fdtd simulation toolbox.

gdsfactory (>= 9) ingestion. Rewritten from scratch for the gf 9 API
(verified against gdsfactory 9.45, 2026-07-07):

- ``Component.get_polygons_points(by="tuple")`` -> {(layer, datatype): [Nx2 um]}
- ``Port.center``/``width`` are um floats; ``Port.orientation`` is degrees;
  the GDS layer tuple comes from ``Port.layer_info`` (kdb.LayerInfo).
- ``Component.dbbox()`` is the um bounding box.
- gf 9.44+ requires an ACTIVE PDK to build components — that is the caller's
  responsibility (e.g. ``gf.gpdk.PDK.activate()``); this module never touches
  PDK state.

The pre-rewrite converter (bugs B2/B3/B4) used the pre-gf-8 API: it indexed a
hardcoded polygon, named every port after the component, and misread units.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from ..errors import LayoutError
from ..geometry import Component, Port, Region, Structure
from ..simprocessor import get_material

logger = logging.getLogger(__name__)

_ORTHOGONAL = (0.0, 90.0, 180.0, 270.0)


def _snap_orientation(orientation: float, port_name: str) -> int:
    """Snap a gf orientation (deg) to the package's 0/90/180/270 convention."""
    o = float(orientation) % 360.0
    for target in _ORTHOGONAL:
        if abs(o - target) < 1e-6 or abs(o - target - 360.0) < 1e-6:
            return int(target)
    raise NotImplementedError(
        f"Port {port_name!r} has non-orthogonal orientation {orientation}°; "
        "only 0/90/180/270 are supported."
    )


def _port_layer_tuple(port: Any) -> tuple[int, int]:
    """GDS (layer, datatype) of a gf 9 port via its kdb.LayerInfo."""
    info = port.layer_info
    return (int(info.layer), int(info.datatype))


def _ensure_trailing_digits(name: str, fallback_index: int) -> str:
    if re.search(r"\d+$", name):
        return name
    new = f"{name}{fallback_index}"
    logger.warning("gdsfactory port %r has no trailing digits; renamed to %r", name, new)
    return new


_MERGE_DBU_UM = 0.001  # 1 nm merge grid, matching the usual GDS dbu


def _merge_layer_polygons(polygons: list[Any]) -> list[list[list[float]]]:
    """Union the polygons of one layer into merged simple polygons (um floats).

    Abutting polygons (a hierarchical gdsfactory component returns one polygon
    per placed reference) must extrude as ONE body: with a sidewall angle each
    polygon tapers from its own footprint, so every internal junction opens a
    wedge-shaped gap in the sloped faces (issue #1). The KLayout GDS loader
    already merges per layer (``pya.Region.merge()``); this applies the same
    proven pipeline — including hole handling via ``to_simple_polygon`` — to
    the gdsfactory frontend. Coordinates snap to a 1 nm grid, the standard
    GDS dbu.
    """
    import pya

    region = pya.Region()
    for pts in polygons:
        arr = np.asarray(pts, dtype=float) / _MERGE_DBU_UM
        region.insert(pya.Polygon([pya.Point(int(round(x)), int(round(y))) for x, y in arr]))
    region.merge()
    merged: list[list[list[float]]] = []
    for poly in region.each_merged():
        simple = poly.to_simple_polygon()
        merged.append([[pt.x * _MERGE_DBU_UM, pt.y * _MERGE_DBU_UM] for pt in simple.each_point()])
    return merged


def from_gdsfactory(c: Any, tech: Any, z_span: float = 4.0) -> Component:
    """Convert a gdsfactory (>=9) Component into a gds_fdtd Component.

    Args:
        c: gdsfactory component (requires an active PDK to have been built).
        tech: technology (gds_fdtd.technology.Technology, or the legacy dict).
        z_span: simulation z-extent captured in the bounds [um].

    Returns:
        Component: flat, role-tagged structures + ports + bounds.
    """
    try:
        import gdsfactory  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "gdsfactory is not installed. Install with: pip install gds_fdtd[gdsfactory]"
        ) from e

    tech_dict = tech.to_solver_dict() if hasattr(tech, "to_solver_dict") else tech
    device_layers = {tuple(d["layer"]): d for d in tech_dict["device"]}

    # ---- structures: one per MERGED polygon per tech-declared device layer ----
    # (abutting per-reference polygons union into one body so angled sidewalls
    # cannot open gaps at internal junctions — issue #1)
    structures: list[Structure] = []
    polygons_by_layer = c.get_polygons_points(by="tuple")
    for layer_tuple, polygons in polygons_by_layer.items():
        d = device_layers.get(tuple(layer_tuple))
        if d is None:
            logger.info("layer %s present in component but not in technology; skipped", layer_tuple)
            continue
        for i, pts in enumerate(_merge_layer_polygons(polygons)):
            structures.append(
                Structure(
                    name=f"poly_{layer_tuple[0]}_{layer_tuple[1]}_{i}",
                    polygon=pts,
                    z_base=d["z_base"],
                    z_span=d["z_span"],
                    material=get_material(d),
                    sidewall_angle=d["sidewall_angle"],
                    layer=list(layer_tuple),
                )
            )
    if not structures:
        raise LayoutError(
            f"component {c.name!r} has no polygons on any technology device layer "
            f"(tech layers: {sorted(device_layers)}; component layers: "
            f"{sorted(map(tuple, polygons_by_layer))})"
        )

    # ---- ports (um floats in gf 9; z from the port's device layer) ----
    ports: list[Port] = []
    for i, p in enumerate(c.ports):
        layer_tuple = _port_layer_tuple(p)
        d = device_layers.get(layer_tuple)
        if d is None:
            logger.info("port %r on non-device layer %s; skipped", p.name, layer_tuple)
            continue
        z = d["z_base"] + d["z_span"] / 2
        ports.append(
            Port(
                name=_ensure_trailing_digits(str(p.name), i + 1),
                center=[float(p.center[0]), float(p.center[1]), z],
                width=float(p.width),
                direction=_snap_orientation(p.orientation, str(p.name)),
            )
        )

    # ---- bounds: um bbox + evanescent margin, EXCEPT on sides with ports ----
    # Ports must lie ON the bounds edge (the devrec convention): the solvers
    # place the domain edge one buffer beyond the bounds and extend waveguide
    # stubs 2*buffer through the PML. Dilating past a port plane leaves the
    # stub ending inside the domain — an abrupt facet that reflects (found by
    # the live gf->Lumerical validation: S11 -7 dB on a straight).
    bb = c.dbbox()
    margin = 1.9
    tol = 1e-6

    def _side_margin(side_coord: float, axis: int, outward: int) -> float:
        for p_ in ports:
            on_side = abs(p_.center[axis] - side_coord) < tol
            faces_out = (axis == 0 and p_.direction == (0 if outward > 0 else 180)) or (
                axis == 1 and p_.direction == (90 if outward > 0 else 270)
            )
            if on_side and faces_out:
                return 0.0
        return margin

    x_lo = bb.left - _side_margin(bb.left, 0, -1)
    x_hi = bb.right + _side_margin(bb.right, 0, +1)
    y_lo = bb.bottom - _side_margin(bb.bottom, 1, -1)
    y_hi = bb.top + _side_margin(bb.top, 1, +1)
    vertices = [[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]]
    z_by_layer: dict[tuple[int, ...], float] = {}
    for s in structures:
        z_by_layer.setdefault(tuple(s.layer), s.z_base + s.z_span / 2)
    z_center = float(np.average(list(z_by_layer.values())))
    bounds = Region(vertices=vertices, z_center=z_center, z_span=z_span)

    # ---- substrate / superstrate from the technology ----
    background = []
    for key, role, layer in (
        ("substrate", "substrate", [999, 0]),
        ("superstrate", "superstrate", [999, 1]),
    ):
        b = tech_dict[key][0]
        background.append(
            Structure(
                name=key.capitalize(),
                polygon=vertices,
                z_base=b["z_base"],
                z_span=b["z_span"],
                material=get_material(b),
                layer=layer,
                role=role,
            )
        )

    component = Component(
        name=c.name,
        structures=[*background, *structures],
        ports=ports,
        bounds=bounds,
    )
    # keep the source component around: engines with native gdsfactory
    # pipelines (beamz) pick it up automatically, so solver setup stays
    # identical across engines — get_solver(name)(component, tech, spec)
    component.gf_component = c  # type: ignore[attr-defined]  # dynamic engine hint (see beamz)
    return component
