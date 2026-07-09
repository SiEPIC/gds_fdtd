"""WP5.2a: permittivity rasterizer — analytic accuracy + index resolution."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from gds_fdtd.geometry import Component, Port, Region, Structure
from gds_fdtd.grid import PermittivityGrid, rasterize, resolve_index

TESTS_DIR = pathlib.Path(__file__).parent


def _component(structures, name="fixture"):
    ports = [
        Port(name="opt1", center=[0.0, 0.5, 0.11], width=0.5, direction=180),
        Port(name="opt2", center=[2.0, 0.5, 0.11], width=0.5, direction=0),
    ]
    bounds = Region(vertices=[[0, 0], [2, 0], [2, 1], [0, 1]], z_center=0.11, z_span=0.22)
    return Component(name=name, structures=structures, ports=ports, bounds=bounds)


def _rect(width=2.0, height=1.0, n=3.47, angle=90, z_base=0.0, z_span=0.22, role="device"):
    return Structure(
        name="rect",
        polygon=[[0, 0], [width, 0], [width, height], [0, height]],
        z_base=z_base,
        z_span=z_span,
        material={"nk": n},
        sidewall_angle=angle,
        role=role,
    )


# ---------------------------------------------------------------------------
# resolve_index
# ---------------------------------------------------------------------------


def test_resolve_index_shapes():
    assert resolve_index(1.44, 1.55) == 1.44 + 0j
    assert resolve_index({"nk": 3.47}, 1.55) == 3.47 + 0j
    assert resolve_index({"nk": [3.47, 0.01]}, 1.55) == 3.47 + 0.01j
    assert resolve_index({"tidy3d_db": {"nk": 1.48}}, 1.55) == 1.48 + 0j
    assert resolve_index({"tidy3d": {"nk": 1.48}}, 1.55) == 1.48 + 0j


def test_resolve_index_rii(monkeypatch):
    monkeypatch.setenv("GDS_FDTD_RII_DB", str(TESTS_DIR / "rii_db"))
    n = resolve_index({"rii": {"shelf": "main", "book": "Si", "page": "Li-293"}}, 1.55)
    assert n.real == pytest.approx(3.476, abs=0.01)


def test_resolve_index_rejects_engine_db_names():
    with pytest.raises(ValueError, match="Lumerical"):
        resolve_index({"tidy3d": None, "lum": "Si (Silicon) - Palik"}, 1.55)
    with pytest.raises(ValueError, match="cannot resolve"):
        resolve_index(object(), 1.55)


# ---------------------------------------------------------------------------
# rasterizer accuracy
# ---------------------------------------------------------------------------


def test_rectangle_area_fraction_analytic():
    """Card acceptance: rectangle area from eps fractions exact to 1e-3 at 10 nm."""
    n_core, n_bg = 3.47, 1.0
    comp = _component([_rect(width=2.0, height=1.0, n=n_core)])
    g = rasterize(comp, dx=0.01, dz=0.22, buffer=0.5, background_index=n_bg)
    k = g.eps.shape[2] // 2
    frac = (g.eps[:, :, k].real - n_bg**2) / (n_core**2 - n_bg**2)
    area = float(frac.sum()) * g.spacing[0] * g.spacing[1]
    assert area == pytest.approx(2.0 * 1.0, rel=1e-3)


def test_grid_geometry_and_coords():
    comp = _component([_rect()])
    g = rasterize(comp, dx=0.05, buffer=0.5)
    assert isinstance(g, PermittivityGrid)
    assert g.eps.shape == (len(g.x), len(g.y), len(g.z))
    # domain = bounds + buffer on xy; z = structure stack extent
    assert g.x[0] == pytest.approx(g.origin[0] + 0.025)
    assert g.z.min() >= 0.0 and g.z.max() <= 0.22
    # cell centered inside the rectangle is pure core
    i = int((1.0 - g.origin[0]) / g.spacing[0])
    j = int((0.5 - g.origin[1]) / g.spacing[1])
    assert g.eps[i, j, 0].real == pytest.approx(3.47**2, rel=1e-12)


def test_sidewall_angle_tapers_volume():
    """An 80-degree sidewall shrinks the cross-section with height; the
    rasterized volume must match the analytic prismatoid within 1%."""
    w, h, t, angle = 2.0, 1.0, 0.2, 80.0
    comp = _component([_rect(width=w, height=h, angle=angle, z_span=t)])
    g = rasterize(comp, dx=0.01, dz=0.01, buffer=0.3)
    vol = (
        float(((g.eps.real - 1.0) / (3.47**2 - 1.0)).sum())
        * g.spacing[0]
        * g.spacing[1]
        * g.spacing[2]
    )
    # analytic: at height z the rectangle insets by z*tan(10 deg) per side
    from math import radians, tan

    zs = g.z
    slabs = [
        (w - 2 * z * tan(radians(90 - angle))) * (h - 2 * z * tan(radians(90 - angle))) for z in zs
    ]
    expected = float(np.sum(slabs)) * g.spacing[2]
    assert vol == pytest.approx(expected, rel=1e-2)


def test_background_roles_paint_first():
    """A device sitting inside a substrate must override it, not blend."""
    substrate = Structure(
        name="sub",
        polygon=[[-1, -1], [3, -1], [3, 2], [-1, 2]],
        z_base=0.0,
        z_span=0.22,
        material={"nk": 1.44},
        sidewall_angle=90,
        role="substrate",
    )
    comp = _component([_rect(n=3.47), substrate])
    g = rasterize(comp, dx=0.05, buffer=0.2)
    i = int((1.0 - g.origin[0]) / g.spacing[0])
    j = int((0.5 - g.origin[1]) / g.spacing[1])
    assert g.eps[i, j, 0].real == pytest.approx(3.47**2, rel=1e-12)  # device wins
    jj = int((-0.1 - g.origin[1]) / g.spacing[1])
    assert g.eps[i, jj, 0].real == pytest.approx(1.44**2, rel=1e-12)  # substrate elsewhere


def test_rasterize_input_validation():
    comp = _component([_rect()])
    with pytest.raises(ValueError, match="positive"):
        rasterize(comp, dx=0.0)
    with pytest.raises(ValueError, match="supersample"):
        rasterize(comp, dx=0.05, supersample=0)
    with pytest.raises(ValueError, match="empty z range"):
        rasterize(comp, dx=0.05, z_min=1.0, z_max=0.0)
