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
#: online contexts; offline users get render_static)
_THREE_VERSION = "0.160.0"

# object palette: layers cycle through the RdBu-family chips used across the
# package; infrastructure keeps the plot_monitor_planes color language.
_LAYER_COLORS = ["#b2182b", "#2166ac", "#d6604d", "#4393c3", "#f4a582", "#92c5de"]
_BACKGROUND_COLOR = "#7fa8c9"
_PORT_COLOR = "#67001f"
_MONITOR_COLORS = {"x": "#d62728", "y": "#1f77b4", "z": "#2ca02c"}


def _structure_objects(component: Component) -> tuple[list[dict[str, Any]], dict[str, str]]:
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
    return objects, legend


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

    objects, legend = _structure_objects(component)
    objects += _port_objects(component)

    scene: dict[str, Any] = {"name": component.name, "objects": objects, "legend": legend}
    if spec is not None and hasattr(obj, "domain"):
        center, span = obj.domain()
        scene["domain"] = {"center": [float(c) for c in center], "span": [float(s) for s in span]}
        objects += _monitor_objects(component, spec, center, span)
    return scene


_HTML_TEMPLATE = """\
<div id="__ID__" style="position:relative;width:100%;height:__HEIGHT__px;\
background:#101418;border-radius:8px;overflow:hidden;\
font-family:system-ui,sans-serif">
  <div id="__ID___info" style="position:absolute;left:10px;top:10px;z-index:2;\
color:#dde3ea;font-size:12px;background:rgba(16,20,24,.75);padding:6px 10px;\
border-radius:6px;max-width:65%">__NAME__ — drag to orbit · scroll to zoom · \
click an object</div>
  <div id="__ID___legend" style="position:absolute;right:10px;top:10px;z-index:2;\
color:#dde3ea;font-size:12px;background:rgba(16,20,24,.75);padding:6px 10px;\
border-radius:6px"></div>
</div>
<script type="module">
import * as THREE from "https://cdn.jsdelivr.net/npm/three@__THREE__/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/controls/OrbitControls.js";

const SCENE = __SCENE_JSON__;
const host = document.getElementById("__ID__");
const info = document.getElementById("__ID___info");
const legendBox = document.getElementById("__ID___legend");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
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

// z is "up" for a chip; scale z so thin layers stay visible
const Z_SCALE = 4.0;
const root = new THREE.Group();
root.rotation.x = -Math.PI / 2;   // lay the chip flat, z up on screen
scene.add(root);

const pickables = [];
const groups = {};
function inGroup(name) {
  if (!groups[name]) { groups[name] = new THREE.Group(); root.add(groups[name]); }
  return groups[name];
}

let bbox = new THREE.Box3();
function track(mesh) { bbox.expandByObject(mesh); }

for (const o of SCENE.objects) {
  if (o.kind === "structure") {
    const shape = new THREE.Shape(o.contour.map(p => new THREE.Vector2(p[0], p[1])));
    const geo = new THREE.ExtrudeGeometry(shape,
      { depth: (o.z1 - o.z0) * Z_SCALE, bevelEnabled: false });
    const mat = new THREE.MeshStandardMaterial({
      color: o.color, roughness: 0.55, metalness: 0.05,
      transparent: o.opacity < 1, opacity: o.opacity,
      depthWrite: o.opacity >= 1,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.z = o.z0 * Z_SCALE;
    mesh.userData = o;
    inGroup(o.group).add(mesh);
    if (o.opacity >= 1) { pickables.push(mesh); track(mesh); }
  } else if (o.kind === "port") {
    const geo = new THREE.ConeGeometry(0.35 * o.width, 0.9 * o.width, 20);
    const mat = new THREE.MeshStandardMaterial({ color: o.color, roughness: 0.4 });
    const mesh = new THREE.Mesh(geo, mat);
    const rad = (o.direction * Math.PI) / 180;
    // cone points INTO the device (against the port's facing direction)
    mesh.rotation.z = rad - Math.PI / 2;
    mesh.position.set(o.center[0] + 0.45 * o.width * Math.cos(rad),
                      o.center[1] + 0.45 * o.width * Math.sin(rad),
                      o.center[2] * Z_SCALE);
    mesh.userData = o;
    inGroup("ports").add(mesh);
    pickables.push(mesh); track(mesh);
  }
}

if (SCENE.domain) {
  const d = SCENE.domain;
  const geo = new THREE.BoxGeometry(d.span[0], d.span[1], d.span[2] * Z_SCALE);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0x9aa7b4 }));
  edges.position.set(d.center[0], d.center[1], d.center[2] * Z_SCALE);
  edges.userData = { name: "simulation domain",
    info: `domain ${d.span[0].toFixed(1)} × ${d.span[1].toFixed(1)} × ${d.span[2].toFixed(2)} µm` };
  inGroup("domain").add(edges);
  track(edges);

  for (const o of SCENE.objects.filter(o => o.kind === "monitor")) {
    let w, h;
    if (o.axis === "z") { w = d.span[0]; h = d.span[1]; }
    else if (o.axis === "y") { w = d.span[0]; h = d.span[2] * Z_SCALE; }
    else { w = d.span[1]; h = d.span[2] * Z_SCALE; }
    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({ color: o.color, transparent: true,
      opacity: 0.18, side: THREE.DoubleSide, depthWrite: false });
    const mesh = new THREE.Mesh(geo, mat);
    if (o.axis === "z") mesh.position.set(d.center[0], d.center[1], o.position * Z_SCALE);
    else if (o.axis === "y") { mesh.rotation.x = Math.PI / 2;
      mesh.position.set(d.center[0], o.position, d.center[2] * Z_SCALE); }
    else { mesh.rotation.y = Math.PI / 2; mesh.rotation.z = Math.PI / 2;
      mesh.position.set(o.position, d.center[1], d.center[2] * Z_SCALE); }
    mesh.userData = o;
    inGroup("monitors").add(mesh);
    pickables.push(mesh);
  }
}

// frame the scene
const c = new THREE.Vector3(); bbox.getCenter(c);
const size = new THREE.Vector3(); bbox.getSize(size);
const radius = Math.max(size.x, size.y, size.z, 1);
camera.position.set(c.x + radius * 0.9, c.y + radius * 1.1, c.z + radius * 0.8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(c.x, c.z, -c.y);  // account for root rotation
controls.enableDamping = true;

// legend with visibility toggles
const CHIP = "display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px";
for (const [label, color] of Object.entries(SCENE.legend)) {
  legendBox.insertAdjacentHTML("beforeend",
    `<div><span style="${CHIP};background:${color}"></span>${label}</div>`);
}
for (const g of Object.keys(groups)) {
  const id = "__ID___" + g;
  legendBox.insertAdjacentHTML("beforeend",
    `<div><input type="checkbox" id="${id}" checked style="margin-right:6px">${g}</div>`);
  legendBox.querySelector("#" + CSS.escape(id)).addEventListener("change",
    e => { groups[g].visible = e.target.checked; });
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
    info.textContent = "__NAME__";
  }
});

function animate() {
  requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera);
}
animate();
new ResizeObserver(() => {
  camera.aspect = host.clientWidth / host.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(host.clientWidth, host.clientHeight);
}).observe(host);
</script>
"""


def scene_html(scene: dict[str, Any], height: int = 520) -> str:
    """The interactive viewer as a self-contained HTML snippet (three.js via CDN)."""
    uid = f"gdsfdtd3d_{abs(hash(json.dumps(scene, sort_keys=True))) % 10**8}"
    return (
        _HTML_TEMPLATE.replace("__ID__", uid)
        .replace("__HEIGHT__", str(int(height)))
        .replace("__NAME__", str(scene.get("name", "component")))
        .replace("__THREE__", _THREE_VERSION)
        .replace("__SCENE_JSON__", json.dumps(scene))
    )


def show_3d(obj: Any, height: int = 520) -> Any:
    """Display the interactive 3D scene in a notebook (and in the docs gallery).

    Accepts a solver (geometry + ports + monitor planes + domain) or a bare
    component (geometry + ports). The output is a plain HTML snippet, so the
    committed notebook stays lightweight and myst-nb renders it interactive
    on the documentation site. Requires internet for the three.js CDN; use
    :func:`render_static` where JavaScript cannot run.
    """
    from typing import cast

    from IPython import display as _display

    # IPython's typing varies across versions; call through Any so the strict
    # gate holds regardless of which release is installed
    html_cls = cast("Any", _display).HTML

    # srcdoc-iframe isolates the viewer from the notebook's own DOM/CSS
    html = scene_html(build_scene(obj), height=height)
    escaped = html.replace("&", "&amp;").replace('"', "&quot;")
    return html_cls(
        f'<iframe srcdoc="{escaped}" style="width:100%;height:{height + 20}px;'
        f'border:none" loading="lazy" title="gds_fdtd 3D viewer"></iframe>'
    )


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
        if o["kind"] != "structure" or o["opacity"] < 1:
            continue
        pts = np.asarray(o["contour"])
        z0, z1 = o["z0"], o["z1"]
        top = [(x, y, z1) for x, y in pts]
        bot = [(x, y, z0) for x, y in pts]
        faces = [top, bot]
        n = len(pts)
        faces += [[bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]] for i in range(n)]
        ax.add_collection3d(
            Poly3DCollection(faces, facecolor=o["color"], edgecolor="none", alpha=0.95)
        )
    for o in scene["objects"]:
        if o["kind"] == "port":
            x, y, z = o["center"]
            ax.scatter([x], [y], [z], color=o["color"], s=45, depthshade=False)
            ax.text(x, y, z + 0.25, o["name"], fontsize=8, ha="center")

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
                Poly3DCollection(verts, facecolor=o["color"], alpha=0.15, edgecolor=o["color"])
            )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0 - 0.3, z1 + 0.3)

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_zlabel("z [µm]")
    ax.set_title(scene["name"])
    try:  # matplotlib >= 3.6
        ax.set_aspect("equalxy")
    except (NotImplementedError, ValueError):
        pass  # older 3D axes cannot fix the aspect; the view stays usable
    ax.view_init(elev=elev, azim=azim)
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax
