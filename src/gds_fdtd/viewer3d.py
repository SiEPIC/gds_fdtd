"""
gds_fdtd simulation toolbox.

Interactive 3D viewer: the component's extruded layer stack, its ports, the
field-monitor planes, and the simulation domain in one orbitable scene.

The scene is built in Python as plain JSON (polygon contours + z extents; no
new dependencies) and rendered by three.js in a self-contained HTML snippet.
The same snippet works everywhere an HTML output renders: Jupyter (via
``show_3d``), the Sphinx docs gallery (myst-nb passes it through), or a
standalone file (``save_3d``). Clicking an object shows its name, material,
layer, and z-extent; the legend toggles each group. ``render_static`` draws
the identical scene with matplotlib for contexts that cannot run JavaScript
(the README, PDF exports).

Geometry note: structures render as vertical extrusions of their footprint
(sidewall angles are not shown; engines still simulate them).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .geometry import Component
    from .spec import SimulationSpec

#: pinned three.js build served from the jsDelivr CDN (docs + notebooks are
#: online contexts; offline users get render_static). 0.137.x is the last
#: line shipping the classic (non-module) examples/js OrbitControls that
#: notebook webviews can run — see the loader comment in the template.
_THREE_VERSION = "0.137.5"

# object palette: layers cycle through the RdBu-family chips used across the
# package; infrastructure keeps the plot_monitor_planes color language.
_LAYER_COLORS = ["#b2182b", "#2166ac", "#d6604d", "#4393c3", "#f4a582", "#92c5de"]
_BACKGROUND_COLOR = "#7fa8c9"
_PORT_COLOR = "#67001f"
_MONITOR_COLORS = {"x": "#d62728", "y": "#1f77b4", "z": "#2ca02c"}


def _structure_objects(
    component: Component,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[tuple[int, ...], str]]:
    """Device + background structures as contour-extrusion scene objects."""
    objects: list[dict[str, Any]] = []
    layer_colors: dict[tuple[int, ...], str] = {}
    legend: dict[str, str] = {}
    for s in component.structures:
        pts = np.asarray(s.polygon, dtype=float)
        if pts.ndim != 2 or len(pts) < 3:
            continue
        z0, z1 = sorted((float(s.z_base), float(s.z_base + s.z_span)))
        if s.role == "device":
            key = tuple(s.layer) if s.layer else ()
            color = layer_colors.setdefault(
                key, _LAYER_COLORS[len(layer_colors) % len(_LAYER_COLORS)]
            )
            label = f"layer {key[0]}/{key[1]}" if key else "device"
            legend.setdefault(label, color)
            opacity, group = 1.0, "layers"
        else:
            color = _BACKGROUND_COLOR
            label = s.role or "background"
            legend.setdefault("substrate / superstrate", color)
            opacity, group = 0.10, "background"
        material = s.material if isinstance(s.material, str) else type(s.material).__name__
        if isinstance(s.material, dict):
            material = ", ".join(sorted(s.material))
        objects.append(
            {
                "kind": "structure",
                "group": group,
                "name": s.name,
                "info": f"{label} · material: {material} · z {z0:g}..{z1:g} µm",
                "contour": [[float(x), float(y)] for x, y in pts],
                "z0": z0,
                "z1": z1,
                "color": color,
                "opacity": opacity,
            }
        )
    return objects, legend, layer_colors


def _port_plane_objects(component: Component, spec: SimulationSpec) -> list[dict[str, Any]]:
    """The mode source/monitor plane of each port, at its real dimensions."""
    objects = []
    w, d = float(spec.width_ports), float(spec.depth_ports)
    for p in component.ports:
        z = float(p.center[2]) if len(p.center) > 2 and p.center[2] is not None else 0.11
        objects.append(
            {
                "kind": "portplane",
                "group": "ports",
                "name": f"{p.name}_plane",
                "info": (
                    f"port {p.name} mode plane · {w:g} × {d:g} µm · "
                    f"({p.center[0]:g}, {p.center[1]:g}, {z:g}) µm, normal {p.direction:g}°"
                ),
                "center": [float(p.center[0]), float(p.center[1]), z],
                "direction": float(p.direction),
                "width": w,
                "depth": d,
                "color": _PORT_COLOR,
            }
        )
    return objects


def _port_extension_objects(
    component: Component, spec: SimulationSpec, layer_colors: dict[tuple[int, ...], str]
) -> list[dict[str, Any]]:
    """The port-extension stubs the solvers extrude through the PML."""
    objects = []
    buffer = 2 * float(spec.buffer)
    for p in component.ports:
        try:
            contour = np.asarray(p.polygon_extension(buffer=buffer), dtype=float)
        except (
            AttributeError,
            ValueError,
            TypeError,
            IndexError,
        ):  # a port without geometry info cannot extend
            continue
        if contour.ndim != 2 or len(contour) < 3:
            continue
        h = getattr(p, "height", None)
        height = float(h) if h else 0.22
        z = float(p.center[2]) if len(p.center) > 2 and p.center[2] is not None else 0.11
        la = getattr(p, "layer", None)
        key = tuple(la) if la else ()
        objects.append(
            {
                "kind": "structure",
                "group": "port extensions",
                "name": f"{p.name}_extension",
                "info": (
                    f"port {p.name} extension stub · +{buffer:g} µm through the "
                    f"domain edge · z {z - height / 2:g}..{z + height / 2:g} µm"
                ),
                "contour": [[float(x), float(y)] for x, y in contour],
                "z0": z - height / 2,
                "z1": z + height / 2,
                "color": layer_colors.get(key, _PORT_COLOR),
                "opacity": 0.45,
            }
        )
    return objects


def _port_objects(component: Component) -> list[dict[str, Any]]:
    objects = []
    for p in component.ports:
        width = float(p.width or 0.5)
        height = float(p.height or 0.22) if getattr(p, "height", None) else 0.22
        z = float(p.center[2]) if len(p.center) > 2 and p.center[2] is not None else 0.11
        objects.append(
            {
                "kind": "port",
                "group": "ports",
                "name": str(p.name),
                "info": f"port {p.name} · faces {p.direction}° · width {width:g} µm",
                "center": [float(p.center[0]), float(p.center[1]), z],
                "direction": float(p.direction),
                "width": width,
                "height": height,
                "color": _PORT_COLOR,
            }
        )
    return objects


def _monitor_objects(
    component: Component, spec: SimulationSpec, center: list[float], span: list[float]
) -> list[dict[str, Any]]:
    # engine-default plane positions (matching the tidy3d adapter)
    z_by_layer: dict[tuple[int, ...], float] = {}
    for s in component.structures:
        if s.role == "device" and s.layer:
            z_by_layer.setdefault(tuple(s.layer), s.z_base + s.z_span / 2)
    z_default = float(np.average(list(z_by_layer.values()))) if z_by_layer else 0.0
    defaults = {"x": center[0], "y": center[1], "z": z_default}
    positions = dict(getattr(spec, "field_monitor_positions", {}) or {})
    objects = []
    for axis in getattr(spec, "field_monitors", ()) or ():
        pos = float(positions.get(axis, defaults[axis]))
        origin = "custom" if axis in positions else "default"
        objects.append(
            {
                "kind": "monitor",
                "group": "monitors",
                "name": f"{axis}_field",
                "info": f"{axis}_field monitor @ {axis}={pos:g} µm ({origin})",
                "axis": axis,
                "position": pos,
                "color": _MONITOR_COLORS[axis],
            }
        )
    return objects


def build_scene(obj: Any) -> dict[str, Any]:
    """Build the JSON scene for a solver or a bare component.

    A solver contributes its component, the simulation domain, and the
    monitor planes; a bare :class:`~gds_fdtd.geometry.Component` yields
    geometry and ports only.
    """
    component = obj.component if hasattr(obj, "component") else obj
    spec = getattr(obj, "spec", None)

    objects, legend, layer_colors = _structure_objects(component)
    objects += _port_objects(component)

    scene: dict[str, Any] = {"name": component.name, "objects": objects, "legend": legend}
    if spec is not None and hasattr(obj, "domain"):
        center, span = obj.domain()
        scene["domain"] = {"center": [float(c) for c in center], "span": [float(s) for s in span]}
        objects += _monitor_objects(component, spec, center, span)
        objects += _port_plane_objects(component, spec)
        objects += _port_extension_objects(component, spec, layer_colors)
        # the tech's substrate/superstrate slabs usually reach beyond the
        # simulated z-window; clamp them to the domain so the scene stays tight
        z_lo, z_hi = center[2] - span[2] / 2, center[2] + span[2] / 2
        for o in objects:
            if o.get("group") == "background":
                o["z0"], o["z1"] = max(o["z0"], z_lo), min(o["z1"], z_hi)

    # camera framing, computed here where the data is (in chip coordinates;
    # the template applies its z exaggeration and axis flip)
    if "domain" in scene:
        frame_center = scene["domain"]["center"]
        frame_span = scene["domain"]["span"]
    else:
        xy = np.vstack(
            [np.asarray(o["contour"]) for o in objects if o["kind"] == "structure"]
            or [np.zeros((1, 2))]
        )
        zs = [z for o in objects if o["kind"] == "structure" for z in (o["z0"], o["z1"])] or [0.0]
        frame_center = [
            float((xy[:, 0].min() + xy[:, 0].max()) / 2),
            float((xy[:, 1].min() + xy[:, 1].max()) / 2),
            float((min(zs) + max(zs)) / 2),
        ]
        frame_span = [
            float(xy[:, 0].max() - xy[:, 0].min()),
            float(xy[:, 1].max() - xy[:, 1].min()),
            float(max(zs) - min(zs)),
        ]
    scene["frame"] = {"center": frame_center, "span": frame_span}
    return scene


_HTML_TEMPLATE = """\
<div id="__ID__" style="position:relative;width:100%;height:__HEIGHT__px;\
background:#101418;border-radius:8px;overflow:hidden;\
font-family:system-ui,sans-serif">
  <div id="__ID___info" style="position:absolute;left:10px;top:10px;z-index:2;\
color:#dde3ea;font-size:12px;background:rgba(16,20,24,.75);padding:6px 10px;\
border-radius:6px;max-width:65%">loading 3D viewer…</div>
  <div id="__ID___legend" style="position:absolute;right:10px;top:10px;z-index:2;\
color:#dde3ea;font-size:12px;background:rgba(16,20,24,.75);padding:6px 10px;\
border-radius:6px"></div>
</div>
<script>
(function () {
"use strict";
// Classic (non-module) scripts: notebook webviews (VSCode, JupyterLab) do not
// reliably run ES-module CDN imports, and import maps are one-per-page so
// multiple embeds would collide. The UMD builds work everywhere; the loader
// is guarded so several viewers on one page fetch three.js once.
var BASE = "https://cdn.jsdelivr.net/npm/three@__THREE__/";
var SCENE = __SCENE_JSON__;

// Notebook renderers mount cell outputs in surprising places: VSCode-style
// webviews may put them behind shadow roots, where document.getElementById
// cannot see our divs. Try the root this script itself lives in first
// (document.currentScript, captured synchronously), then the document, then
// walk every open shadow root; retry briefly in case the output attaches
// after this script runs.
var CS = document.currentScript;
function findEl(id) {
  if (CS && CS.getRootNode) {
    var r = CS.getRootNode();
    if (r && r.querySelector) {
      var near = r.querySelector("#" + id);
      if (near) return near;
    }
  }
  var el = document.getElementById(id);
  if (el) return el;
  function walk(root) {
    var all = root.querySelectorAll("*");
    for (var i = 0; i < all.length; i++) {
      var sr = all[i].shadowRoot;
      if (sr) {
        var hit = sr.querySelector("#" + id) || walk(sr);
        if (hit) return hit;
      }
    }
    return null;
  }
  return walk(document);
}

function start(attempt) {
var host = findEl("__ID__");
var info = findEl("__ID___info");
var legendBox = findEl("__ID___legend");
if (!host || !info || !legendBox) {
  if (attempt < 25) setTimeout(function () { start(attempt + 1); }, 120);
  return;
}

// boot stages are written into the panel: a failure names itself instead of
// leaving a silent black rectangle
// SCENE.name reaches the DOM only via textContent, which never parses HTML
function stage(msg) { info.textContent = (SCENE.name || "component") + " — " + msg; }
function fail(msg) {
  info.textContent = "3D viewer: " + msg +
    " — use viewer3d.render_static for a static view.";
}
stage("container found; loading three.js…");
function load(src, cb) {
  var s = document.createElement("script");
  s.src = src;
  s.onload = cb;
  s.onerror = function () { fail("could not load three.js from the CDN (offline?)"); };
  document.head.appendChild(s);
}
function ensureThree(cb) {
  if (window.THREE && window.THREE.OrbitControls) return cb();
  if (window.__gdsfdtd3d_loading) { window.__gdsfdtd3d_loading.push(cb); return; }
  window.__gdsfdtd3d_loading = [cb];
  load(BASE + "build/three.min.js", function () {
    load(BASE + "examples/js/controls/OrbitControls.js", function () {
      var queue = window.__gdsfdtd3d_loading; window.__gdsfdtd3d_loading = null;
      queue.forEach(function (f) { f(); });
    });
  });
}

ensureThree(function () {
try {
stage("three.js ready; building the scene…");
var THREE = window.THREE;
var OrbitControls = THREE.OrbitControls;

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
} catch (e) {
  fail("WebGL is unavailable in this environment");
  return;
}
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(host.clientWidth, host.clientHeight);
renderer.setClearColor(0x101418);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, host.clientWidth / host.clientHeight, 0.01, 5000);
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(1, -1.5, 2.5);
scene.add(key);
const fill = new THREE.DirectionalLight(0xbcd4ff, 0.35); fill.position.set(-2, 1, -1);
scene.add(fill);

// z is "up" for a chip. zScale = 1 shows TRUE proportions (a 220 nm layer
// really is hair-thin next to a many-µm device); the legend checkbox turns on
// a 4x z exaggeration for inspecting thin stacks.
let zScale = 1.0;
const root = new THREE.Group();
root.rotation.x = -Math.PI / 2;   // lay the chip flat, z up on screen
scene.add(root);

let pickables = [];
let groups = {};
const groupVisibility = {};
const F = SCENE.frame;
let radius = 1;

function inGroup(name) {
  if (!groups[name]) { groups[name] = new THREE.Group(); root.add(groups[name]); }
  return groups[name];
}

// text label as a camera-facing sprite (canvas texture; no DOM/HTML sinks)
function makeLabel(txt) {
  const c = document.createElement("canvas");
  let ctx = c.getContext("2d");
  const fs = 42;
  ctx.font = fs + "px system-ui, sans-serif";
  c.width = Math.ceil(ctx.measureText(txt).width) + 26;
  c.height = fs + 22;
  ctx = c.getContext("2d");
  ctx.font = fs + "px system-ui, sans-serif";
  ctx.fillStyle = "rgba(16,20,24,0.72)";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#ffd9d9";
  ctx.textBaseline = "middle";
  ctx.fillText(txt, 13, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial(
    { map: tex, transparent: true, depthTest: false }));
  const h = 0.055 * radius;  // readable at any device size
  sp.scale.set(h * c.width / c.height, h, 1);
  return sp;
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function disposeTree(obj) {
  // three.js does not free GPU memory on JS GC; release geometries, materials
  // and textures explicitly so repeated z-scale rebuilds do not leak VRAM
  obj.traverse(function (n) {
    if (n.geometry) n.geometry.dispose();
    if (n.material) {
      var mats = Array.isArray(n.material) ? n.material : [n.material];
      mats.forEach(function (m) { if (m.map) m.map.dispose(); m.dispose(); });
    }
  });
}

function buildScene() {
  disposeTree(root);
  while (root.children.length) root.remove(root.children[0]);
  groups = {}; pickables = [];
  const Z = zScale;
  radius = Math.max(F.span[0], F.span[1], F.span[2] * Z, 1);

  for (const o of SCENE.objects) {
    if (o.kind === "structure") {
      const shape = new THREE.Shape(o.contour.map(p => new THREE.Vector2(p[0], p[1])));
      const geo = new THREE.ExtrudeGeometry(shape,
        { depth: (o.z1 - o.z0) * Z, bevelEnabled: false });
      const mat = new THREE.MeshStandardMaterial({
        color: o.color, roughness: 0.55, metalness: 0.05,
        transparent: o.opacity < 1, opacity: o.opacity,
        depthWrite: o.opacity >= 1,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.z = o.z0 * Z;
      mesh.userData = o;
      inGroup(o.group).add(mesh);
      if (o.group !== "background") pickables.push(mesh);
    } else if (o.kind === "portplane") {
      // the port's mode source/monitor plane at its real dimensions, with a
      // crisp border and a floating name + dimensions label
      const geo = new THREE.PlaneGeometry(o.width, o.depth * Z);
      const mat = new THREE.MeshBasicMaterial({ color: o.color, transparent: true,
        opacity: 0.35, side: THREE.DoubleSide, depthWrite: false });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0xff6b6b })));
      if (o.direction % 180 === 0) { mesh.rotation.y = Math.PI / 2; mesh.rotation.z = Math.PI / 2; }
      else { mesh.rotation.x = Math.PI / 2; }
      mesh.position.set(o.center[0], o.center[1], o.center[2] * Z);
      mesh.userData = o;
      inGroup("ports").add(mesh);
      pickables.push(mesh);
      const tag = makeLabel(o.name.replace("_plane", "") + " · " +
                            o.width + "×" + o.depth + " µm");
      tag.position.set(o.center[0], o.center[1],
                       (o.center[2] + o.depth / 2) * Z + 0.06 * radius);
      inGroup("ports").add(tag);
    } else if (o.kind === "port") {
      const geo = new THREE.ConeGeometry(0.35 * o.width, 0.9 * o.width, 20);
      const mat = new THREE.MeshStandardMaterial({ color: o.color, roughness: 0.4 });
      const mesh = new THREE.Mesh(geo, mat);
      const rad = (o.direction * Math.PI) / 180;
      // cone points INTO the device (against the port's facing direction)
      mesh.rotation.z = rad - Math.PI / 2;
      mesh.position.set(o.center[0] + 0.45 * o.width * Math.cos(rad),
                        o.center[1] + 0.45 * o.width * Math.sin(rad),
                        o.center[2] * Z);
      mesh.userData = o;
      inGroup("ports").add(mesh);
      pickables.push(mesh);
    }
  }

  if (SCENE.domain) {
    const d = SCENE.domain;
    const geo = new THREE.BoxGeometry(d.span[0], d.span[1], d.span[2] * Z);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x9aa7b4 }));
    edges.position.set(d.center[0], d.center[1], d.center[2] * Z);
    edges.userData = { name: "simulation domain",
      info: `domain ${d.span[0].toFixed(1)} × ${d.span[1].toFixed(1)} × ` +
        `${d.span[2].toFixed(2)} µm` };
    inGroup("domain").add(edges);

    for (const o of SCENE.objects.filter(o => o.kind === "monitor")) {
      let w, h;
      if (o.axis === "z") { w = d.span[0]; h = d.span[1]; }
      else if (o.axis === "y") { w = d.span[0]; h = d.span[2] * Z; }
      else { w = d.span[1]; h = d.span[2] * Z; }
      const geo = new THREE.PlaneGeometry(w, h);
      const mat = new THREE.MeshBasicMaterial({ color: o.color, transparent: true,
        opacity: 0.18, side: THREE.DoubleSide, depthWrite: false });
      const mesh = new THREE.Mesh(geo, mat);
      if (o.axis === "z") mesh.position.set(d.center[0], d.center[1], o.position * Z);
      else if (o.axis === "y") { mesh.rotation.x = Math.PI / 2;
        mesh.position.set(d.center[0], o.position, d.center[2] * Z); }
      else { mesh.rotation.y = Math.PI / 2; mesh.rotation.z = Math.PI / 2;
        mesh.position.set(o.position, d.center[1], d.center[2] * Z); }
      mesh.userData = o;
      inGroup("monitors").add(mesh);
      pickables.push(mesh);
    }
  }

  // keep the user's group toggles across z-scale rebuilds
  for (const g of Object.keys(groups)) {
    if (g in groupVisibility) groups[g].visible = groupVisibility[g];
  }

  // frame from the Python-computed bounds. Chip coordinates map to world as
  // (x, y, z) -> (x, z * Z, -y) through the root rotation.
  const target = new THREE.Vector3(F.center[0], F.center[2] * Z, -F.center[1]);
  camera.position.set(target.x + 0.55 * radius,
                      target.y + 0.55 * radius,
                      target.z + 0.85 * radius);
  controls.target.copy(target);
}
buildScene();

// legend with visibility toggles. Built with createElement/textContent only:
// HTML-string sinks (innerHTML/insertAdjacentHTML) are restricted by Trusted
// Types in hardened webviews, and text APIs work everywhere.
const CHIP = "display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px";
for (const [label, color] of Object.entries(SCENE.legend)) {
  const row = document.createElement("div");
  const chip = document.createElement("span");
  chip.setAttribute("style", CHIP + ";background:" + color);
  row.appendChild(chip);
  row.appendChild(document.createTextNode(label));
  legendBox.appendChild(row);
}
for (const g of Object.keys(groups)) {
  const row = document.createElement("div");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = true;
  box.setAttribute("style", "margin-right:6px");
  box.addEventListener("change", function () {
    groupVisibility[g] = box.checked;
    if (groups[g]) groups[g].visible = box.checked;
  });
  row.appendChild(box);
  row.appendChild(document.createTextNode(g));
  legendBox.appendChild(row);
}
{
  // true 1:1 proportions by default; opt-in exaggeration for thin stacks
  const row = document.createElement("div");
  row.setAttribute("style", "margin-top:6px;border-top:1px solid #39424d;padding-top:5px");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = false;
  box.setAttribute("style", "margin-right:6px");
  box.addEventListener("change", function () {
    zScale = box.checked ? 4.0 : 1.0;
    try { buildScene(); } catch (e) {
      fail("scene rebuild failed: " + (e && e.message ? e.message : e));
    }
  });
  row.appendChild(box);
  row.appendChild(document.createTextNode("z ×4 (exaggerate thin layers)"));
  legendBox.appendChild(row);
}

// picking
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let selected = null;
renderer.domElement.addEventListener("click", (ev) => {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
  mouse.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects(pickables, false);
  if (selected && selected.material.emissive) selected.material.emissive.setHex(0);
  if (hits.length) {
    selected = hits[0].object;
    if (selected.material.emissive) selected.material.emissive.setHex(0x333333);
    info.textContent = selected.userData.info || selected.userData.name;
  } else {
    selected = null;
    stage("drag to orbit · scroll to zoom · click an object");
  }
});

function animate() {
  requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera);
}
animate();
stage("drag to orbit · scroll to zoom · click an object");
new ResizeObserver(() => {
  camera.aspect = host.clientWidth / host.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(host.clientWidth, host.clientHeight);
}).observe(host);
} catch (e) {
  // surface the exception where the user can see it - a silent black panel
  // is undebuggable from a screenshot
  fail("scene build failed: " + (e && e.message ? e.message : e));
}
});
}
start(0);
})();
</script>
"""


def scene_html(scene: dict[str, Any], height: int = 520) -> str:
    """The interactive viewer as a self-contained HTML snippet (three.js via CDN)."""
    import uuid

    # ids must be unique per EMISSION, not per scene: a notebook holds the
    # committed output and the freshly-run one side by side, and a content-
    # derived id would make the script attach its canvas to the stale copy
    uid = f"gdsfdtd3d_{uuid.uuid4().hex[:10]}"
    # Escape the scene JSON for a <script> context: user-derived strings
    # (component/structure/port names, materials — sourced from arbitrary GDS
    # files) must not be able to close the script element or inject markup.
    # <, >, & become \u.... escapes json.loads reads back unchanged, so the
    # JS still parses the real characters; U+2028/U+2029 are JSON-legal but
    # were JS line terminators before ES2019. The name reaches the DOM only
    # through SCENE.name via textContent (never parsed as HTML), so there is
    # no separate __NAME__ substitution to escape.
    scene_json = (
        json.dumps(scene)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return (
        _HTML_TEMPLATE.replace("__ID__", uid)
        .replace("__HEIGHT__", str(int(height)))
        .replace("__THREE__", _THREE_VERSION)
        .replace("__SCENE_JSON__", scene_json)
    )


def show_3d(obj: Any, height: int = 520) -> Any:
    """Display the interactive 3D scene in a notebook (and in the docs gallery).

    Accepts a solver (geometry + ports + monitor planes + domain) or a bare
    component (geometry + ports). The output is an ``<iframe srcdoc>`` — the
    one embed that executes its scripts in every renderer (the folium/pydeck
    pattern): JupyterLab inserts outputs via innerHTML, where bare ``<script>``
    tags are inert by spec, but an iframe's document always runs; VSCode and
    static docs pages load it the same way. Inside the iframe the page is a
    plain document, so no shadow-DOM or sanitizer concerns apply. Requires
    internet for the three.js CDN; use :func:`render_static` where JavaScript
    cannot run.
    """
    import warnings
    from typing import cast

    from IPython import display as _display

    # IPython's typing varies across versions; call through Any so the strict
    # gate holds regardless of which release is installed
    html_cls = cast("Any", _display).HTML

    html = scene_html(build_scene(obj), height=height)
    escaped = html.replace("&", "&amp;").replace('"', "&quot;")
    iframe = (
        f'<iframe srcdoc="{escaped}" style="width:100%;height:{height + 20}px;'
        f'border:none" loading="lazy" title="gds_fdtd 3D viewer"></iframe>'
    )
    with warnings.catch_warnings():
        # IPython suggests IPython.display.IFrame on seeing an iframe string,
        # but IFrame needs a served src URL; srcdoc is the point here
        warnings.filterwarnings("ignore", message=".*IFrame.*")
        return html_cls(iframe)


def save_3d(obj: Any, path: str, height: int = 640, title: str | None = None) -> str:
    """Write the viewer as a standalone HTML page (for docs or sharing)."""
    scene = build_scene(obj)
    body = scene_html(scene, height=height)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title or scene['name']} — gds_fdtd 3D</title>"
        "<style>body{margin:0;background:#101418}</style></head>"
        f"<body>{body}</body></html>"
    )
    with open(path, "w") as f:
        f.write(page)
    return path


def render_static(
    obj: Any,
    ax: Any = None,
    elev: float = 28.0,
    azim: float = -65.0,
    savefig: str | None = None,
) -> tuple[Any, Any]:
    """Matplotlib 3D rendering of the same scene (no JavaScript required).

    Used for the README and any static context. Layers keep their viewer
    colors; substrate/superstrate are omitted for clarity; the domain box and
    monitor planes are drawn when a solver is given.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    scene = build_scene(obj)
    if ax is None:
        fig = plt.figure(figsize=(9, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    for o in scene["objects"]:
        if o["kind"] != "structure" or o["group"] == "background":
            continue
        pts = np.asarray(o["contour"])
        z0, z1 = o["z0"], o["z1"]
        top = [(x, y, z1) for x, y in pts]
        bot = [(x, y, z0) for x, y in pts]
        faces = [top, bot]
        n = len(pts)
        faces += [[bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]] for i in range(n)]
        alpha = 0.95 if o["opacity"] >= 1 else 0.35  # port-extension stubs stay see-through
        ax.add_collection3d(
            Poly3DCollection(faces, facecolor=o["color"], edgecolor="none", alpha=alpha),
            autolim=False,  # explicit limits below; mpl's 3D autoscale can overflow
        )
    for o in scene["objects"]:
        if o["kind"] == "port":
            x, y, z = o["center"]
            ax.scatter([x], [y], [z], color=o["color"], s=45, depthshade=False)
            ax.text(x, y, z + 0.25, o["name"], fontsize=8, ha="center")
        elif o["kind"] == "portplane":
            x, y, z = o["center"]
            w, d = o["width"], o["depth"]
            if o["direction"] % 180 == 0:  # x-normal plane
                verts = [
                    [
                        (x, y - w / 2, z - d / 2),
                        (x, y + w / 2, z - d / 2),
                        (x, y + w / 2, z + d / 2),
                        (x, y - w / 2, z + d / 2),
                    ]
                ]
            else:  # y-normal plane
                verts = [
                    [
                        (x - w / 2, y, z - d / 2),
                        (x + w / 2, y, z - d / 2),
                        (x + w / 2, y, z + d / 2),
                        (x - w / 2, y, z + d / 2),
                    ]
                ]
            ax.add_collection3d(
                Poly3DCollection(verts, facecolor=o["color"], alpha=0.25, edgecolor=o["color"]),
                autolim=False,  # explicit limits below; mpl's 3D autoscale can overflow
            )
            ax.text(
                x,
                y,
                z + d / 2 + 0.15,
                f"{w:g}×{d:g} µm",
                fontsize=7,
                ha="center",
                color=o["color"],
            )

    if "domain" in scene:
        d = scene["domain"]
        cx, cy, cz = d["center"]
        sx, sy, sz = d["span"]
        x0, x1, y0, y1, z0, z1 = (
            cx - sx / 2,
            cx + sx / 2,
            cy - sy / 2,
            cy + sy / 2,
            cz - sz / 2,
            cz + sz / 2,
        )
        for za in (z0, z1):
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [za] * 5, color="0.55", lw=0.8)
        for xa in (x0, x1):
            for ya in (y0, y1):
                ax.plot([xa, xa], [ya, ya], [z0, z1], color="0.55", lw=0.8)
        for o in scene["objects"]:
            if o["kind"] != "monitor":
                continue
            p = o["position"]
            if o["axis"] == "z":
                verts = [[(x0, y0, p), (x1, y0, p), (x1, y1, p), (x0, y1, p)]]
            elif o["axis"] == "y":
                verts = [[(x0, p, z0), (x1, p, z0), (x1, p, z1), (x0, p, z1)]]
            else:
                verts = [[(p, y0, z0), (p, y1, z0), (p, y1, z1), (p, y0, z1)]]
            ax.add_collection3d(
                Poly3DCollection(verts, facecolor=o["color"], alpha=0.15, edgecolor=o["color"]),
                autolim=False,  # explicit limits below; mpl's 3D autoscale can overflow
            )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0 - 0.3, z1 + 0.3)
    else:
        # add_collection3d does not feed autoscale, so without a domain box
        # the limits must come from the structures themselves
        structs = [o for o in scene["objects"] if o["kind"] == "structure"]
        if structs:
            xy = np.vstack([np.asarray(o["contour"]) for o in structs])
            zs = [z for o in structs for z in (o["z0"], o["z1"])]
            ax.set_xlim(float(xy[:, 0].min()), float(xy[:, 0].max()))
            ax.set_ylim(float(xy[:, 1].min()), float(xy[:, 1].max()))
            ax.set_zlim(min(zs) - 0.3, max(zs) + 0.3)

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_zlabel("z [µm]")
    ax.set_title(scene["name"])
    try:  # true 1:1:1 proportions, matching the interactive viewer's default
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        z0, z1 = ax.get_zlim()
        ax.set_box_aspect((abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)))
    except (AttributeError, NotImplementedError, ValueError):
        pass  # older 3D axes cannot fix the aspect; the view stays usable
    ax.view_init(elev=elev, azim=azim)
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax
