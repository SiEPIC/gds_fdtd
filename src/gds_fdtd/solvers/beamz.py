"""
gds_fdtd simulation toolbox.

BeamzSolver: the beamz (>= 0.5.3, < 0.6) adapter on the Phase-3 Solver contract. beamz
is an open-source JAX FDTD engine (Apache-2.0, pip-installable, CPU or GPU) —
the first zero-cost engine in the registry.

Geometry is constructed from the solver-neutral component polygons. Preparation
uses the native rasterizer on the CPU; only run() constructs a BeamZ simulation and executes it.
One simulation per excited port produces the full single-mode TE S-matrix using
BeamZ's immutable ports, monitors, and detached result analysis API.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ..errors import JobValidationError, SolverError
from ..smatrix import SMatrix
from .base import (
    ResourceEstimate,
    SetupArtifacts,
    Solver,
    SolverCapabilities,
    register_solver,
)

UM = 1e-6
C_M_S = 299792458.0

# geometry margins mirroring beamz's reference compact-model example
_PML_XY = 1.0 * UM
_PML_Z = 1.0 * UM
_MONITOR_CLEARANCE = 1.0 * UM
_Z_PADDING = 0.5 * UM
_PORT_MARGIN = 0.5 * UM
_MODE_PLANE_SCALE = 1.8
# Put sources in the uniform port extensions, before a fixed measurement
# plane. Moving a monitor between excitation columns changes the phase datum
# and violates reciprocity, especially on short/coarse-grid waveguides.
_SOURCE_OFFSET = -0.35 * UM
_OUTPUT_MONITOR_OFFSET = 0.05 * UM
_DECAY_RATIO = 1e-4
_RUN_AFTER_SOURCES_UOC = 90.0


def probe_beamz() -> str | None:
    try:
        import beamz

        version = tuple(int(part) for part in beamz.__version__.split(".")[:3])
        if not ((0, 5, 3) <= version < (0, 6)):
            return f"BeamzSolver requires beamz>=0.5.3,<0.6; found {beamz.__version__}"

        return None
    except Exception as e:  # pragma: no cover - env dependent
        return f"beamz not importable: {e}"


def _move_along(
    center: tuple[float, float], direction: str, distance: float
) -> tuple[float, float]:
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[str(direction)]


def _port_plane(
    port: dict[str, Any], *, span: float, z_span: float, z_center: float, offset: float = 0.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(start, end) corner points of a port-normal plane (beamz convention)."""
    cx, cy = _move_along(port["center"], port["direction"], offset)
    z0, z1 = float(z_center) - 0.5 * float(z_span), float(z_center) + 0.5 * float(z_span)
    if str(port["direction"]).endswith("x"):
        return (cx, cy - 0.5 * float(span), z0), (cx, cy + 0.5 * float(span), z1)
    return (cx - 0.5 * float(span), cy, z0), (cx + 0.5 * float(span), cy, z1)


@register_solver
class BeamzSolver(Solver):
    """beamz JAX FDTD adapter (tier: full-service, execution: local, free)."""

    name = "beamz"
    capabilities = SolverCapabilities(
        tier="full",
        execution="local",
        supports_dispersion=False,  # constant index per layer
        supports_sidewall_angle=False,  # v1: vertical extrusion
        supports_multimode=False,  # v1: TE mode 1
        supports_gpu=True,  # jax backend selects automatically
        cost_model="free",
    )

    def __init__(
        self,
        *args: Any,
        gf_component: Any = None,
        n_core: float | None = None,
        n_clad: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Retain this legacy argument for call-site compatibility. Geometry
        # now always comes from the canonical component's device polygons.
        self.gf_component = (
            gf_component
            if gf_component is not None
            else getattr(self.component, "gf_component", None)
        )
        self._n_core_kwarg = n_core
        self._n_clad_kwarg = n_clad

    @staticmethod
    def probe_available() -> str | None:
        return probe_beamz()

    # ---------------- helpers ----------------

    def _tech_dict(self) -> dict[str, Any] | None:
        t = self.technology
        if t is None:
            return None
        out = t.to_solver_dict() if hasattr(t, "to_solver_dict") else t
        return cast("dict[str, Any]", out)

    def _device_layers(self) -> list[dict[str, Any]]:
        """Every tech device layer actually present in the component, ordered by
        the tech's ``device`` list.

        Each device layer is extruded at its own z, with port extensions
        confined to the layer that contains that port.
        """
        tech = self._tech_dict()
        if tech is None:
            return []
        present = {tuple(s.layer) for s in self.component.structures if s.role == "device"}
        return [d for d in tech["device"] if tuple(d["layer"]) in present]

    def _device_layer(self) -> dict[str, Any] | None:
        """The primary (first-present) device layer; see :meth:`_device_layers`."""
        layers = self._device_layers()
        return layers[0] if layers else None

    def _resolve_index(
        self, material: dict[str, Any], kwarg: float | None, label: str
    ) -> float | None:
        """kwarg override > any offline-resolvable material shape (neutral nk,
        rii @ center wavelength, tidy3d nk/medium — via grid.resolve_index)."""
        if kwarg is not None:
            return float(kwarg)
        from ..errors import MaterialSourceError
        from ..materials.select import select_source, source_index

        try:
            # beamz has no vendor database, so its source precedence is rii -> nk
            src = select_source(material, "beamz", name=label)
            return float(source_index(material, src, self.spec.wavelength_center_um).real)
        except (MaterialSourceError, ValueError, FileNotFoundError, KeyError):
            return None

    def _indices(self) -> tuple[float | None, float | None]:
        tech = self._tech_dict()
        d = self._device_layer()
        n_core = self._resolve_index(d["material"] if d else {}, self._n_core_kwarg, "core")
        clad_mat = tech["superstrate"][0]["material"] if tech else {}
        n_clad = self._resolve_index(clad_mat, self._n_clad_kwarg, "clad")
        return n_core, n_clad

    # ---------------- lifecycle ----------------

    def validate(self) -> list[str]:
        problems = []
        reason = self.probe_available()
        if reason:
            problems.append(reason)
        if not self.component.ports:
            problems.append("component has no ports")
        for port in self.component.ports:
            if port.direction in (90, 270):
                problems.append(
                    f"port {port.name!r} faces {port.direction} deg: this adapter supports "
                    "x-oriented ports only. The F14 guard remains until y-normal "
                    "modal extraction is revalidated on BeamZ 0.5. Orient the device "
                    "along x, or use tidy3d/lumerical for devices with y-facing ports."
                )
        if not any(s.role == "device" for s in self.component.structures):
            problems.append(
                "BeamzSolver needs a component carrying device-layer polygons "
                "(gdsfactory/GDS/KLayout/SiEPIC sources are supported)"
            )
        if self.technology is None:
            problems.append("BeamzSolver requires a technology")
            return problems
        layers = self._device_layers()
        if not layers:
            problems.append(
                "BeamzSolver needs at least one technology device layer present in the component"
            )
        if tuple(self.spec.modes) != (1,):
            problems.append(f"BeamzSolver v1 supports modes=(1,) (TE); got {self.spec.modes}")
        # Resolve a refractive index for EVERY device layer — multi-layer stacks
        # (e.g. the Si→SiN escalator) are supported. An n_core= override only
        # makes sense for a single-layer device.
        n_core_kwarg = self._n_core_kwarg if len(layers) == 1 else None
        for d in layers:
            if (
                self._resolve_index(d["material"], n_core_kwarg, f"core {tuple(d['layer'])}")
                is None
            ):
                problems.append(
                    f"cannot resolve refractive index for device layer {tuple(d['layer'])}: "
                    "pass n_core= (single-layer only), or give its material an 'rii' reference "
                    "or an 'nk' constant (beamz has no vendor DB)"
                )
        _, n_clad = self._indices()
        if n_clad is None:
            problems.append(
                "cannot resolve cladding refractive index: pass n_clad=, or give the "
                "superstrate material an 'rii' reference or an 'nk' constant"
            )
        return problems

    def build(self) -> SetupArtifacts:
        """Prepare the extruded design, grid, frequencies and pulse (offline)."""
        problems = self.validate()
        if problems:
            raise JobValidationError("cannot build: " + "; ".join(problems))

        import beamz
        from beamz.devices.sources.time import gaussian_band_pulse

        s = self.spec
        d = self._device_layer()
        assert d is not None  # validate() guarantees a device layer
        n_core, n_clad = self._indices()
        assert n_core is not None and n_clad is not None
        wl0 = s.wavelength_center_um * UM
        core_t = abs(d["z_span"]) * UM
        # xy guard band = PML + monitor clearance + a safety margin. beamz needs
        # at least this much for correct PML/monitor placement, so it is a hard
        # floor; but honor a LARGER spec.buffer if the user asked for one (the
        # documented meaning of SimulationSpec.buffer). buffer <= the floor
        # (incl. the default 1.0) leaves the domain exactly as before.
        extension = max(_PML_XY + _MONITOR_CLEARANCE + 1.0 * UM, s.buffer * UM)

        # Honor a larger requested z window while retaining the cladding and
        # PML guard bands around every layer in the stack.
        _zfloor_um = (_PORT_MARGIN + _Z_PADDING) / UM  # 1.0 um of cladding
        _all = self._device_layers()
        z_lo_all = min(min(dd["z_base"], dd["z_base"] + dd["z_span"]) for dd in _all)
        z_hi_all = max(max(dd["z_base"], dd["z_base"] + dd["z_span"]) for dd in _all)
        z_lo_prim = min(d["z_base"], d["z_base"] + d["z_span"])
        z_hi_prim = max(d["z_base"], d["z_base"] + d["z_span"])
        want_lo = min(z_lo_all - _zfloor_um, s.z_min)
        want_hi = max(z_hi_all + _zfloor_um, s.z_max)

        layers = self._device_layers()
        indices = [
            self._resolve_index(
                layer["material"], self._n_core_kwarg if len(layers) == 1 else None, "core"
            )
            for layer in layers
        ]
        dx, dt = cast(Any, beamz.dxdt)(
            wl0,
            n_max=max(float(n) for n in indices if n is not None),
            dims=3,
            safety_factor=0.999,
            points_per_wavelength=s.mesh,
        )
        # Keep the physical guard bands used by the 0.4 adapter. Use the
        # canonical geometry for every frontend, including multilayer stacks.
        polygons = [st for st in self.component.structures if st.role == "device"]
        xy = np.concatenate([np.asarray(st.polygon)[:, :2] for st in polygons])
        lower, upper = xy.min(axis=0) * UM, xy.max(axis=0) * UM
        xy_off = extension - lower
        z_off = -want_lo * UM + _Z_PADDING + _PML_Z
        structures = []
        for layer, index in zip(layers, indices, strict=True):
            assert index is not None
            lo, hi = sorted((layer["z_base"], layer["z_base"] + layer["z_span"]))
            footprints = [st.polygon for st in polygons if tuple(st.layer) == tuple(layer["layer"])]
            for layer_port in self.component.ports:
                if (
                    layer_port.center[2] is None
                    or lo - 1e-9 <= float(layer_port.center[2]) <= hi + 1e-9
                ):
                    # Raster domains round up to whole cells. Extend one extra
                    # cell so the waveguide also fills the final PML voxel.
                    footprints.append(layer_port.polygon_extension(buffer=(extension + dx) / UM))
            for poly in footprints:
                structures.append(
                    beamz.Polygon(
                        vertices=[
                            (float(x) * UM + xy_off[0], float(y) * UM + xy_off[1]) for x, y in poly
                        ],
                        z=lo * UM + z_off,
                        depth=(hi - lo) * UM,
                        material=beamz.Material(permittivity=float(index) ** 2),
                    )
                )
        design = beamz.Design(
            width=float(upper[0] - lower[0] + 2 * extension),
            height=float(upper[1] - lower[1] + 2 * extension),
            depth=(want_hi - want_lo) * UM + 2 * (_Z_PADDING + _PML_Z),
            material=beamz.Material(permittivity=float(n_clad) ** 2),
            structures=structures,
        )
        directions = {0: "-x", 180: "+x", 90: "-y", 270: "+y"}
        ports = {
            p.name: {
                "center": (
                    float(p.center[0]) * UM + xy_off[0],
                    float(p.center[1]) * UM + xy_off[1],
                ),
                "direction": directions[int(p.direction)],
                "width": float(p.width) * UM,
                "z_center": float(
                    p.center[2] if p.center[2] is not None else (z_lo_prim + z_hi_prim) / 2
                )
                * UM
                + z_off,
            }
            for p in self.component.ports
        }
        # Native rasterization is CPU based in BeamZ 0.5.
        grid = design.rasterize(resolution=dx)

        freqs = np.linspace(
            C_M_S / (s.wavelength_end * UM),
            C_M_S / (s.wavelength_start * UM),
            s.wavelength_points,
            dtype=np.float32,
        )

        # Each mode plane is centered on its own core in a multilayer stack.
        gds_ports = {p.name: p for p in self.component.ports}
        planes = {}
        for name, port in ports.items():
            width = float(port.get("width", 0.5 * UM))
            span = _MODE_PLANE_SCALE * (width + 2 * _PORT_MARGIN)
            core_h = core_t
            z_center = float(port.get("z_center", design.depth / 2))
            gp = gds_ports.get(name)
            if len(layers) > 1 and gp is not None and gp.center[2] is not None:
                z_center = float(gp.center[2]) * UM + z_off
                if gp.height is not None:
                    core_h = float(gp.height) * UM
            z_span = _MODE_PLANE_SCALE * (core_h + 2 * _PORT_MARGIN)
            planes[name] = {
                "span": span,
                "z_span": z_span,
                "z_center": z_center,
                "monitor": _port_plane(
                    port,
                    span=span,
                    z_span=z_span,
                    z_center=z_center,
                    offset=_OUTPUT_MONITOR_OFFSET,
                ),
            }

        max_dist_um = 0.0
        centers = {
            n: (
                0.5 * (p["monitor"][0][0] + p["monitor"][1][0]),
                0.5 * (p["monitor"][0][1] + p["monitor"][1][1]),
            )
            for n, p in planes.items()
        }
        for a in centers.values():
            for b in centers.values():
                max_dist_um = max(max_dist_um, float(np.hypot(a[0] - b[0], a[1] - b[1])) / UM)

        pulse = cast(Any, gaussian_band_pulse)(
            freqs,
            carrier_frequency=C_M_S / wl0,
            dt=dt,
            run_after_sources_uoc=_RUN_AFTER_SOURCES_UOC,
            max_output_distance_um=max_dist_um,
        )

        self._artifacts = SetupArtifacts(
            native={
                "design": design,
                "grid": grid,
                "ports": ports,
                "planes": planes,
                "pulse": pulse,
                "dx": dx,
                "dt": dt,
                "freqs": freqs,
                "wl0": wl0,
            },
            summary={
                "n_ports": len(ports),
                "grid_shape": tuple(np.asarray(grid.permittivity).shape),
                "dx_nm": dx / 1e-9,
                "n_core": n_core,
                "n_clad": n_clad,
                "n_simulations": len(ports),
            },
        )
        return self._artifacts

    def estimate(self) -> ResourceEstimate:
        artifacts = self._artifacts if self._artifacts is not None else self.build()
        shape = artifacts.summary["grid_shape"]
        cells = int(np.prod(shape))
        return ResourceEstimate(
            grid_cells=cells,
            memory_gb=cells * 4 * 12 / 1e9,  # ~12 float32 field/eps arrays
            n_simulations=artifacts.summary["n_simulations"],
            cost_hint="free local compute (JAX; CPU works, GPU if available)",
        )

    def plot_fields(
        self, axis: str = "z", scale: str = "linear", savefig: str | None = None
    ) -> tuple[Any, Any]:
        """``|E|²`` profile at the core-center z-plane (first excitation).

        ``scale="db"`` renders a log view that reveals weak radiation; see
        :func:`gds_fdtd.plotting.plot_field`.
        """
        from ..plotting import plot_field

        if axis != "z":
            raise JobValidationError("BeamzSolver v1 records the z-plane profile only")
        fields = getattr(self, "_field_z", None)
        if fields is None:
            raise SolverError(
                "no field data: include 'z' in spec.field_monitors and call run() first"
            )
        mag2 = sum(np.abs(np.squeeze(v)) ** 2 for v in fields.values())  # rows already y
        meta = self._field_z_meta
        return plot_field(
            np.asarray(mag2),
            extent=(0, meta["width_um"], 0, meta["height_um"]),
            scale=scale,
            title=f"|E|² (z-plane), excitation {meta['source']}, center frequency",
            savefig=savefig,
        )

    def run(self) -> SMatrix:
        """Execute each input port using BeamZ 0.5's immutable results API."""
        artifacts = self._artifacts if self._artifacts is not None else self.build()
        import beamz
        from beamz.analysis import s_parameters

        nat = artifacts.native
        design, ports, planes = nat["design"], nat["ports"], nat["planes"]
        pulse, dx, dt, freqs = nat["pulse"], nat["dx"], nat["dt"], nat["freqs"]
        source_time = beamz.SampledSignal(
            values=pulse.signal,
            quadrature=pulse.signal_quadrature,
            dt=dt,
            freq0=C_M_S / nat["wl0"],
        )
        mode_spec = beamz.ModeSpec(polarization="te")
        entries = []
        port_names = sorted(ports)
        self.run_diagnostics = []

        def make_port(name: str, offset: float) -> Any:
            g = planes[name]
            start, end = _port_plane(
                ports[name],
                span=g["span"],
                z_span=g["z_span"],
                z_center=g["z_center"],
                offset=offset,
            )
            return beamz.Port(
                center=cast(
                    tuple[float, float, float],
                    tuple((a + b) / 2 for a, b in zip(start, end, strict=True)),
                ),
                size=cast(
                    tuple[float, float, float],
                    tuple(b - a for a, b in zip(start, end, strict=True)),
                ),
                name=name,
                direction=ports[name]["direction"][0],
                mode_spec=mode_spec,
            )

        for src_name in port_names:
            source_port = make_port(src_name, _SOURCE_OFFSET)
            source = beamz.ModeSource(
                center=source_port.center,
                size=source_port.size,
                direction=source_port.direction,
                mode_spec=mode_spec,
                source_time=source_time,
            )
            specs = [make_port(name, _OUTPUT_MONITOR_OFFSET) for name in port_names]
            monitors = [port.to_monitor(freqs) for port in specs]
            record_field = "z" in self.spec.field_monitors and src_name == port_names[0]
            if record_field:
                monitors.append(
                    beamz.FieldMonitor(
                        center=(design.width / 2, design.height / 2, planes[src_name]["z_center"]),
                        size=(design.width, design.height, 0.0),
                        freqs=np.asarray([np.median(freqs)]),
                        name="field_z",
                        fields=("Ex", "Ey", "Ez"),
                    )
                )
            sim = beamz.Simulation(
                design=design,
                sources=[source],
                monitors=monitors,
                boundaries=[
                    beamz.PML(edges=("left", "right", "top", "bottom"), thickness=_PML_XY),
                    beamz.PML(edges=("front", "back"), thickness=_PML_Z),
                ],
                time=pulse.time,
                resolution=dx,
                setup_device="cpu",
            )
            result = sim.run(
                progress=False,
                termination=beamz.AutoTermination(
                    min_steps=int(np.ceil((pulse.source_end_time + pulse.tail_time) / dt)),
                    field_decay=_DECAY_RATIO,
                ),
            )
            self.run_diagnostics.append({"source": src_name, "termination": result.termination})
            if record_field:
                data = result["field_z"]
                region = data.sample_region
                if region is None:
                    raise SolverError("BeamZ field monitor returned no sampling region")
                ix, iy = region.axis_interval("x"), region.axis_interval("y")
                if ix is None or iy is None:
                    raise SolverError("BeamZ field monitor returned an incomplete XY region")
                shape = (iy.stop - iy.start, ix.stop - ix.start)
                self._field_z = {
                    c: np.asarray(data.dft_fields[c])[0].reshape(shape) for c in ("Ex", "Ey", "Ez")
                }
                self._field_z_meta = {
                    "width_um": float(design.width) / UM,
                    "height_um": float(design.height) / UM,
                    "source": src_name,
                }
            extracted = cast(Any, s_parameters)(
                result, source_port=src_name, ports=specs, frequencies=freqs
            )
            f_asc = np.asarray(freqs, dtype=float)
            order = np.argsort(f_asc)
            for out_name in port_names:
                col = np.asarray(extracted.s_matrix[(out_name, src_name)], dtype=complex)
                entries.append((src_name, out_name, 1, 1, f_asc[order], col[order]))
        return SMatrix.from_entries(entries, name=self.component.name, port_names=port_names)
