"""Field-monitor placement and visualization: spec knobs, engine threading,
outlines, and the monitor-plane viewer."""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology

TESTS_DIR = pathlib.Path(__file__).parent


# ------------------------------------------------------------- spec knobs


def test_spec_positions_accepted_and_bad_axis_rejected():
    spec = SimulationSpec(field_monitor_positions={"y": 0.5, "z": 0.11})
    assert spec.field_monitor_positions == {"y": 0.5, "z": 0.11}
    with pytest.raises(ValueError):
        SimulationSpec(field_monitor_positions={"w": 1.0})


def test_spec_monitor_wavelengths_validated():
    spec = SimulationSpec(field_monitor_wavelengths=(1.55, 1.52))
    assert spec.field_monitor_wavelengths == (1.55, 1.52)
    with pytest.raises(ValueError, match="outside the simulated band"):
        SimulationSpec(field_monitor_wavelengths=(1.3,))
    with pytest.raises(ValueError, match="positive"):
        SimulationSpec(field_monitor_wavelengths=(-1.55,))


# ------------------------------------------------- shared offline component


def _escalator(tech_file: str):
    tech = Technology.from_yaml(str(TESTS_DIR / tech_file))
    d = copy.deepcopy(tech.to_solver_dict())
    for layer in d["device"]:
        layer["material"] = {"nk": 3.0}
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=d)
    del layout
    return comp, d


# ------------------------------------------------------------- tidy3d engine


def test_t3d_monitor_honors_position_and_wavelengths():
    pytest.importorskip("tidy3d")
    comp, d = _escalator("tech_tidy3d.yaml")
    spec = SimulationSpec(
        wavelength_points=3,
        mesh=6,
        z_min=-1.0,
        z_max=1.11,
        field_monitors=("y", "z"),
        field_monitor_positions={"y": 0.5, "z": 0.11},
        field_monitor_wavelengths=(1.55,),
    )
    art = get_solver("tidy3d")(comp, technology=d, spec=spec).build()
    monitors = {m.name: m for m in art.native.simulation.monitors}
    assert monitors["y_field"].center[1] == pytest.approx(0.5)
    assert monitors["z_field"].center[2] == pytest.approx(0.11)
    # the chosen wavelength is the ONLY recorded frequency
    for name in ("y_field", "z_field"):
        freqs = np.asarray(monitors[name].freqs)
        assert freqs.shape == (1,)
        assert freqs[0] == pytest.approx(2.99792458e14 / 1.55, rel=1e-9)


def test_t3d_monitor_defaults_unchanged():
    pytest.importorskip("tidy3d")
    comp, d = _escalator("tech_tidy3d.yaml")
    spec = SimulationSpec(wavelength_points=3, mesh=6, z_min=-1.0, z_max=1.11)
    art = get_solver("tidy3d")(comp, technology=d, spec=spec).build()
    monitors = {m.name: m for m in art.native.simulation.monitors}
    # default z plane: average of the device layers' mid-planes; broadband
    assert monitors["z_field"].center[0] == pytest.approx(comp.bounds.x_center)
    assert len(np.asarray(monitors["z_field"].freqs)) == 3


# ---------------------------------------------------------------- lumerical


def test_lsf_profile_monitor_honors_position():
    comp, d = _escalator("tech_lumerical.yaml")
    spec = SimulationSpec(
        wavelength_points=3,
        z_min=-1.0,
        z_max=1.11,
        field_monitors=("z",),
        field_monitor_positions={"z": 0.11},
    )
    art = get_solver("lumerical")(comp, technology=d, spec=spec).build()
    script = art.native
    assert 'set("name", "profile_z")' in script
    assert f'set("z", {0.11 * 1e-6});' in script


# ----------------------------------------------------------------- plotting


def test_component_outlines_per_axis():
    from gds_fdtd.plotting import component_outlines

    comp, _ = _escalator("tech_tidy3d.yaml")
    top = component_outlines(comp, axis="z")
    n_dev = sum(1 for s in comp.structures if s.role == "device")
    assert len(top) == n_dev
    side = component_outlines(comp, axis="y")
    assert len(side) == n_dev
    for rect in side:
        assert np.asarray(rect).shape == (4, 2)
    # z-extents of the rectangles are the layers' z bands
    z_bands = sorted({(round(r[0][1], 3), round(r[2][1], 3)) for r in side})
    layers = sorted(
        {
            (
                round(min(s.z_base, s.z_base + s.z_span), 3),
                round(max(s.z_base, s.z_base + s.z_span), 3),
            )
            for s in comp.structures
            if s.role == "device"
        }
    )
    assert z_bands == layers


def test_plot_field_outline_overlay():
    import matplotlib

    matplotlib.use("Agg")
    from gds_fdtd.plotting import plot_field

    fig, ax = plot_field(
        np.ones((4, 4)),
        extent=(0, 1, 0, 1),
        outline=[np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]])],
    )
    assert len(ax.lines) == 1  # the closed outline
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_monitor_planes_labels_default_vs_custom():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gds_fdtd.plotting import plot_monitor_planes

    comp, d = _escalator("tech_lumerical.yaml")
    spec = SimulationSpec(
        z_min=-1.0,
        z_max=1.11,
        field_monitors=("y", "z"),
        field_monitor_positions={"y": 0.75},
    )
    solver = get_solver("lumerical")(comp, technology=d, spec=spec)
    fig, (ax_top, ax_side) = plot_monitor_planes(solver)
    labels = [t.get_text() for t in ax_top.get_legend().get_texts()]
    assert any("y=0.75" in la and "custom" in la for la in labels)
    assert any("z_field" in la and "default" in la for la in labels)
    plt.close(fig)
