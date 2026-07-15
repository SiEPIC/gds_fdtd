"""Offline tests for the visualization helpers added for the examples revamp.

Assert structure (a figure comes back, the right shape, expected metadata) —
not pixels. Uses the resolvable unified tech so rasterization works without an
engine.
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.modes import Mode
from gds_fdtd.plotting import (
    plot_field,
    plot_mode,
    plot_permittivity,
    plot_tech_stack,
    smatrix_summary,
)
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.smatrix import SMatrix
from gds_fdtd.technology import Technology

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def tech():
    return Technology.from_yaml(str(TESTS_DIR / "tech_unified.yaml"))


@pytest.fixture(scope="module")
def component(tech):
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp
    del layout


def _reciprocal_2port() -> SMatrix:
    f = np.linspace(1.9e14, 2.0e14, 5)
    thru = np.full(f.size, 0.98 + 0j)
    refl = np.full(f.size, 0.02 + 0j)
    return SMatrix.from_entries(
        [
            ("o1", "o2", 1, 1, f, thru),
            ("o2", "o1", 1, 1, f, thru),
            ("o1", "o1", 1, 1, f, refl),
            ("o2", "o2", 1, 1, f, refl),
        ],
        name="dut",
    )


def test_plot_tech_stack_returns_figure(tech):
    fig, ax = plot_tech_stack(tech, wavelength_um=1.55)
    # one rectangle per stack band: substrate + superstrate + device layers
    assert len(ax.patches) == 2 + len(tech.device)
    assert "layer stack" in ax.get_title()


def test_plot_tech_stack_saves(tech, tmp_path):
    out = tmp_path / "stack.png"
    plot_tech_stack(tech, savefig=str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_plot_permittivity_all_axes(component, axis):
    fig, ax = plot_permittivity(component, dx=0.05, axis=axis)
    assert len(ax.images) == 1
    img = ax.images[0].get_array()
    # refractive index map: real, spans low (cladding/air) to high (Si core)
    assert np.nanmin(img) >= 1.0 - 1e-9
    assert np.nanmax(img) > 2.0  # the silicon core is n ~ 3.5


def test_plot_permittivity_rejects_bad_axis(component):
    with pytest.raises(ValueError, match="axis must be"):
        plot_permittivity(component, axis="w")


def test_smatrix_summary_metrics():
    summary = smatrix_summary(_reciprocal_2port())
    assert summary["reciprocal"] is True
    assert summary["passive"] is True
    assert 1.5 <= summary["wavelength_um"] <= 1.6
    kinds = {(p["in"], p["out"]): p["kind"] for p in summary["paths"]}
    assert kinds[("o1", "o2")] == "transmission"
    assert kinds[("o1", "o1")] == "reflection"
    # through path is ~ -0.18 dB (|0.98|^2), reflection ~ -34 dB
    thru = next(p for p in summary["paths"] if (p["in"], p["out"]) == ("o1", "o2"))
    assert thru["db"] == pytest.approx(10 * np.log10(0.98**2), abs=0.1)


def test_smatrix_summary_wavelength_selection():
    summary = smatrix_summary(_reciprocal_2port(), wavelength_um=1.5)
    assert summary["wavelength_um"] == pytest.approx(1.5, abs=0.02)


def _synthetic_mode() -> Mode:
    u = np.linspace(-1.0, 1.0, 24)
    v = np.linspace(-0.8, 0.8, 18)
    prof = np.exp(-(u[:, None] ** 2) - (v[None, :] ** 2)).astype(complex)
    fields = {"Ex": prof, "Ey": 0.1 * prof, "Ez": 0.05 * prof, "Hx": prof, "Hy": prof, "Hz": prof}
    return Mode(n_eff=2.44 + 0j, fields=fields, u=u, v=v, wavelength_um=1.55)


def test_plot_mode_total_and_component():
    fig, ax = plot_mode(_synthetic_mode())  # |E|
    assert len(ax.images) == 1
    assert "n_eff = 2.44" in ax.get_title()
    _, ax2 = plot_mode(_synthetic_mode(), field="Ey")
    assert len(ax2.images) == 1


def test_plot_mode_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown field"):
        plot_mode(_synthetic_mode(), field="Q")


def test_plot_field_linear_and_db():
    mag2 = (np.random.default_rng(0).random((30, 40))) ** 2
    # linear: normalized to peak, imshow via extent, clim 0..1
    _, ax = plot_field(mag2, extent=(0, 4, 0, 3), scale="linear")
    assert len(ax.images) == 1
    assert ax.images[0].get_clim() == (0.0, 1.0)
    # db: pcolormesh on true coords, clim floor_db..0
    _, ax2 = plot_field(mag2, x=np.arange(40), y=np.arange(30), scale="db", floor_db=-25)
    assert len(ax2.collections) == 1 and len(ax2.images) == 0
    assert ax2.collections[0].get_clim() == (-25.0, 0.0)


def test_plot_field_rejects_unknown_scale():
    with pytest.raises(ValueError, match="scale must be one of"):
        plot_field(np.ones((4, 4)), scale="log10")


def test_waveguide_mode_canonical_soi_strip():
    pytest.importorskip("tidy3d")
    from gds_fdtd.modes import waveguide_mode

    modes = waveguide_mode(0.5, 0.22, 3.48, 1.44, 1.55, n_modes=1)
    # the canonical 500x220 nm SOI strip TE0 sits near n_eff ~ 2.44
    assert 2.30 < modes[0].n_eff.real < 2.60
