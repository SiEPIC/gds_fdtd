"""The 3D viewer: scene building, HTML generation, and the static renderer."""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import pytest

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.solvers import get_solver
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology
from gds_fdtd.viewer3d import build_scene, render_static, save_3d, scene_html, show_3d

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def escalator_solver():
    tech = Technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))
    d = copy.deepcopy(tech.to_solver_dict())
    for layer in d["device"]:
        layer["material"] = {"nk": 3.0}
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=d)
    spec = SimulationSpec(
        z_min=-1.0,
        z_max=1.11,
        field_monitors=("y", "z"),
        field_monitor_positions={"z": 0.11},
    )
    solver = get_solver("lumerical")(comp, technology=d, spec=spec)
    del layout
    return solver


def test_scene_from_solver_carries_everything(escalator_solver):
    scene = build_scene(escalator_solver)
    kinds = {o["kind"] for o in scene["objects"]}
    assert kinds == {"structure", "port", "monitor"}
    assert "domain" in scene
    center, span = escalator_solver.domain()
    assert scene["domain"]["span"] == pytest.approx(span)
    # every device layer got a legend chip; background is translucent
    assert any(la.startswith("layer ") for la in scene["legend"])
    bg = [o for o in scene["objects"] if o["kind"] == "structure" and o["group"] == "background"]
    assert bg and all(o["opacity"] < 1 for o in bg)
    # the pinned z monitor reports its custom position, y keeps the default
    monitors = {o["name"]: o for o in scene["objects"] if o["kind"] == "monitor"}
    assert monitors["z_field"]["position"] == pytest.approx(0.11)
    assert "custom" in monitors["z_field"]["info"]
    assert "default" in monitors["y_field"]["info"]
    # ports carry direction and width for the cone glyphs
    ports = [o for o in scene["objects"] if o["kind"] == "port"]
    assert len(ports) == len(escalator_solver.component.ports)
    assert all("direction" in p and p["width"] > 0 for p in ports)


def test_scene_from_bare_component_has_no_domain(escalator_solver):
    scene = build_scene(escalator_solver.component)
    assert "domain" not in scene
    assert {o["kind"] for o in scene["objects"]} == {"structure", "port"}


def test_scene_html_embeds_scene_and_viewer(escalator_solver):
    scene = build_scene(escalator_solver)
    html = scene_html(scene, height=430)
    assert "three@0." in html and "OrbitControls" in html
    assert "height:430px" in html
    # the scene JSON survives the template substitution intact
    start = html.index("const SCENE = ") + len("const SCENE = ")
    end = html.index(";\n", start)
    assert json.loads(html[start:end]) == scene


def test_save_3d_writes_standalone_page(escalator_solver, tmp_path):
    path = save_3d(escalator_solver, str(tmp_path / "esc.html"))
    text = pathlib.Path(path).read_text()
    assert text.startswith("<!doctype html>")
    assert "si_sin_escalator" in text


def test_show_3d_wraps_in_iframe(escalator_solver):
    pytest.importorskip("IPython")
    out = show_3d(escalator_solver, height=300)
    html = out.data
    assert html.startswith("<iframe srcdoc=")
    assert "height:320px" in html  # height + 20 chrome
    # the srcdoc content is escaped: no raw double quotes from the inner html
    assert "&quot;" in html


def test_render_static_solver_and_component(escalator_solver, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = tmp_path / "esc3d.png"
    fig, ax = render_static(escalator_solver, savefig=str(out))
    assert out.exists() and out.stat().st_size > 10_000
    plt.close(fig)

    fig2, ax2 = render_static(escalator_solver.component)
    assert ax2.get_zlim() is not None
    plt.close(fig2)


def test_scene_skips_degenerate_polygons(escalator_solver):
    from gds_fdtd.viewer3d import _structure_objects

    comp = escalator_solver.component
    # a structure with a 2-point "polygon" must be ignored, not crash
    bad = copy.copy(comp.structures[0])
    bad.polygon = [[0.0, 0.0], [1.0, 1.0]]
    objects, _ = _structure_objects(comp)
    n_before = len(objects)
    comp2 = copy.copy(comp)
    comp2.structures = list(comp.structures) + [bad]
    objects2, _ = _structure_objects(comp2)
    assert len(objects2) == n_before + 1 or len(objects2) == n_before  # bad one skipped
    assert all(len(np.asarray(o["contour"])) >= 3 for o in objects2)
