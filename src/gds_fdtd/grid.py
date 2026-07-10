"""
gds_fdtd simulation toolbox.

Permittivity rasterizer — the first Tier-B enabler. Kernel-level
engines (fdtdz, fdtdx) take a raw permittivity grid instead of a scene
graph; ``rasterize`` turns a ``Component`` into that grid offline:

- per z-slice shapely cross-sections, sidewall angle applied as a polygon
  offset growing with height (per the technology's angle convention);
- sub-pixel averaging via supersampling (default 4x4 per cell), so material
  boundaries land on the grid as area-fraction-blended permittivity;
- materials resolved to a (complex) refractive index at the given
  wavelength by ``resolve_index`` — offline only: constant ``nk`` entries,
  refractiveindex.info references, and already-built tidy3d media work;
  engine-database names (Lumerical) raise with guidance.

All lengths are um (package-wide convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import radians, tan
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .geometry import Component, Structure

logger = logging.getLogger(__name__)


@dataclass
class PermittivityGrid:
    """Relative permittivity sampled at cell centers.

    ``eps[i, j, k]`` is the cell centered at
    ``origin + ((i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz)``.
    """

    eps: np.ndarray  # (Nx, Ny, Nz), complex
    origin: tuple[float, float, float]  # min corner [um]
    spacing: tuple[float, float, float]  # (dx, dy, dz) [um]

    @property
    def x(self) -> np.ndarray:
        return self.origin[0] + (np.arange(self.eps.shape[0]) + 0.5) * self.spacing[0]

    @property
    def y(self) -> np.ndarray:
        return self.origin[1] + (np.arange(self.eps.shape[1]) + 0.5) * self.spacing[1]

    @property
    def z(self) -> np.ndarray:
        return self.origin[2] + (np.arange(self.eps.shape[2]) + 0.5) * self.spacing[2]


def resolve_index(material: Any, wavelength_um: float) -> complex:
    """Refractive index of a structure material at one wavelength, OFFLINE.

    Accepts every material shape that reaches ``Structure.material``:
    numbers; ``{"nk": n}`` mappings (raw-YAML flow); ``{"rii": {...}}``
    (refractiveindex.info, resolved from the local database); pydantic
    ``MaterialSpec``; the loaded-component dict ``{"tidy3d": medium,
    "lum": name}`` (tidy3d media are evaluated locally; Lumerical names
    live in the engine's database and raise).
    """
    if isinstance(material, (int, float, complex)) and not isinstance(material, bool):
        return complex(material)

    if hasattr(material, "to_solver_dict"):  # pydantic MaterialSpec
        material = material.to_solver_dict()

    if isinstance(material, dict):
        if "nk" in material:
            nk = material["nk"]
            if isinstance(nk, (list, tuple)):
                return complex(nk[0], nk[1] if len(nk) > 1 else 0.0)
            return complex(nk)
        for key in ("tidy3d_db", "tidy3d"):
            hint = material.get(key)
            if isinstance(hint, dict) and "nk" in hint:
                return resolve_index({"nk": hint["nk"]}, wavelength_um)
            if hint is not None and hasattr(hint, "eps_model"):
                # an already-built tidy3d medium: pure local math
                freq_hz = 299792458.0 / (wavelength_um * 1e-6)
                return complex(np.sqrt(complex(hint.eps_model(freq_hz))))
        if "rii" in material and material["rii"] is not None:
            from .materials.rii import load_rii_material

            ref = material["rii"]
            if hasattr(ref, "model_dump"):
                ref = ref.model_dump()
            mat = load_rii_material(ref["shelf"], ref["book"], ref["page"])
            return complex(mat.nk_at(wavelength_um))
        if isinstance(material.get("lum"), str) or "lum_db" in material:
            raise ValueError(
                f"material {material!r} only names an entry in Lumerical's material "
                "database, which cannot be resolved offline. Use a constant 'nk' or a "
                "refractiveindex.info 'rii' reference for rasterization."
            )

    raise ValueError(
        f"cannot resolve a refractive index from material {material!r}; supported: a "
        "number, {'nk': n}, an 'rii' reference, a MaterialSpec, or a tidy3d medium."
    )


def _slice_polygon(structure: Structure, z: float) -> shapely.Polygon | None:
    """Cross-section of a structure at height z, sidewall angle applied."""
    z_lo, z_hi = sorted((structure.z_base, structure.z_base + structure.z_span))
    if not (z_lo <= z <= z_hi):
        return None
    poly = shapely.Polygon(structure.polygon)
    angle = float(structure.sidewall_angle)
    if angle == 90.0:
        return poly
    # height above the structure's base along its growth direction
    h = (z - structure.z_base) if structure.z_span >= 0 else (structure.z_base - z)
    offset = -h * tan(radians(90.0 - angle))
    if offset == 0.0:
        return poly
    shrunk = poly.buffer(offset, join_style="mitre")
    return None if shrunk.is_empty else shrunk


def rasterize(
    component: Component,
    dx: float,
    dy: float | None = None,
    dz: float | None = None,
    *,
    wavelength: float = 1.55,
    z_min: float | None = None,
    z_max: float | None = None,
    buffer: float = 1.0,
    background_index: float = 1.0,
    supersample: int = 4,
) -> PermittivityGrid:
    """Rasterize a Component into a permittivity grid at cell centers.

    Painting order is background roles (substrate, superstrate) first, then
    device structures in list order — later structures overwrite earlier
    ones where they overlap, blended by sub-pixel area fraction at material
    boundaries. The xy domain is the component bounds plus ``buffer``; the
    z domain defaults to the structure stack's full extent.
    """
    dy = dx if dy is None else dy
    dz = dx if dz is None else dz
    if min(dx, dy, dz) <= 0:
        raise ValueError(f"grid spacings must be positive; got {(dx, dy, dz)}")
    if supersample < 1:
        raise ValueError(f"supersample must be >= 1; got {supersample}")

    b = component.bounds
    x0, x1 = b.x_center - b.x_span / 2 - buffer, b.x_center + b.x_span / 2 + buffer
    y0, y1 = b.y_center - b.y_span / 2 - buffer, b.y_center + b.y_span / 2 + buffer
    z_extents = [e for s in component.structures for e in (s.z_base, s.z_base + s.z_span)]
    if not z_extents:
        raise ValueError("component has no structures to rasterize")
    z0 = min(z_extents) if z_min is None else z_min
    z1 = max(z_extents) if z_max is None else z_max
    if z0 >= z1:
        raise ValueError(f"empty z range ({z0}, {z1})")

    nx = max(int(round((x1 - x0) / dx)), 1)
    ny = max(int(round((y1 - y0) / dy)), 1)
    nz = max(int(round((z1 - z0) / dz)), 1)

    # supersampled xy sample points, shared by every slice
    ss = supersample
    xs = x0 + (np.arange(nx * ss) + 0.5) * (dx / ss)
    ys = y0 + (np.arange(ny * ss) + 0.5) * (dy / ss)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xf, yf = xx.ravel(), yy.ravel()

    ordered = [s for s in component.structures if s.role != "device"] + [
        s for s in component.structures if s.role == "device"
    ]
    indices = {id(s): resolve_index(s.material, wavelength) for s in ordered}

    eps = np.full((nx, ny, nz), complex(background_index) ** 2, dtype=complex)
    frac_cache: dict[tuple[int, bytes], np.ndarray] = {}
    for k in range(nz):
        zc = z0 + (k + 0.5) * dz
        for s in ordered:
            poly = _slice_polygon(s, zc)
            if poly is None:
                continue
            key = (id(s), shapely.to_wkb(poly))
            frac = frac_cache.get(key)
            if frac is None:
                shapely.prepare(poly)
                inside = shapely.contains_xy(poly, xf, yf).reshape(nx * ss, ny * ss)
                # average ss x ss blocks -> area fraction per cell
                frac = inside.reshape(nx, ss, ny, ss).mean(axis=(1, 3))
                frac_cache[key] = frac
            eps_s = indices[id(s)] ** 2
            eps[:, :, k] = eps[:, :, k] * (1 - frac) + eps_s * frac

    logger.info(
        "rasterized '%s': %d x %d x %d cells at (%g, %g, %g) um",
        component.name,
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
    )
    return PermittivityGrid(eps=eps, origin=(x0, y0, z0), spacing=(dx, dy, dz))
