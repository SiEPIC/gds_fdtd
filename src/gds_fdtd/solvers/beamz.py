"""
gds_fdtd simulation toolbox.

BeamzSolver: the beamz (>= 0.4) adapter on the Phase-3 Solver contract. beamz is an open-source JAX FDTD engine
(Apache-2.0, pip-installable, CPU or GPU) — the first zero-cost engine in the
registry.

Design decisions (rule 8 — everything below verified against beamz 0.4.3 and
its own compact-model reference example, which is the UBC SiEPIC crossing):
- geometry rides beamz's ``gdsf.prepare_component`` pipeline (extrusion, port
  extension, padding) rather than re-deriving its delicate port-plane/PML
  tuning. It is fed the source **gdsfactory component** when one is present, or
  else a shim over the polygons gds_fdtd already extruded from ANY source
  (KLayout/SiEPIC, raw GDS) — so beamz is solver-agnostic like the other
  adapters, not limited to gdsfactory-object inputs.
- refractive indices resolve from, in order: explicit ``n_core=``/``n_clad=``
  kwargs → the technology material's ``rii`` reference (refractiveindex.info,
  evaluated at the center wavelength) → ``tidy3d_db.nk``.
- v1 scope (validated, not aspirational): ONE device layer, TE, mode 1.
  Multi-layer stacks / TM / higher modes are future work.
- units: beamz is SI (meters, seconds); gds_fdtd is um — converted here.
- run() executes one FDTD run per excited port (full S-matrix = N runs), with
  adaptive decay stopping; extraction via ``get_S_matrix_modal_dft``.
"""

from __future__ import annotations

import numpy as np

from ..errors import JobValidationError
from ..plotting import DEFAULT_CMAP
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
_SOURCE_OFFSET = 0.10 * UM
_SOURCE_TO_MONITOR = 0.40 * UM
_OUTPUT_MONITOR_OFFSET = 0.05 * UM
_DECAY_RATIO = 1e-4
_LOOKBACK = 20
_RUN_AFTER_SOURCES_UOC = 90.0


def probe_beamz() -> str | None:
    try:
        import beamz  # noqa: F401

        return None
    except Exception as e:  # pragma: no cover - env dependent
        return f"beamz not importable: {e}"


def _move_along(center, direction: str, distance: float):
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[str(direction)]


def _port_plane(port: dict, *, span: float, z_span: float, z_center: float, offset: float = 0.0):
    """(start, end) corner points of a port-normal plane (beamz convention)."""
    cx, cy = _move_along(port["center"], port["direction"], offset)
    z0, z1 = float(z_center) - 0.5 * float(z_span), float(z_center) + 0.5 * float(z_span)
    if str(port["direction"]).endswith("x"):
        return (cx, cy - 0.5 * float(span), z0), (cx, cy + 0.5 * float(span), z1)
    return (cx - 0.5 * float(span), cy, z0), (cx + 0.5 * float(span), cy, z1)


class _ShimPort:
    """Minimal port view that beamz's ``gdsf.load`` accepts (name/orientation/
    center/width), sourced from a gds_fdtd :class:`~gds_fdtd.geometry.Port`."""

    def __init__(self, port):
        self.name = port.name
        self.orientation = float(port.direction)
        self.center = (float(port.center[0]), float(port.center[1]))
        self.width = float(port.width)


class _ComponentImportShim:
    """Adapt a solver-agnostic gds_fdtd ``Component`` to the tiny surface beamz's
    ``gdsf.prepare_component``/``load`` pipeline consumes from a gdsfactory
    component: ``name``, ``get_polygons_points(by=...)`` and ``ports``.

    This is the whole point of the toolbox — the polygons gds_fdtd already
    extruded from *any* source (KLayout/SiEPIC, gdsfactory, raw GDS) feed beamz
    through its own tuned extrusion/port pipeline, so beamz is no longer limited
    to gdsfactory-object inputs. Coordinates are microns on both sides.
    """

    def __init__(self, component):
        self._component = component
        self.name = component.name

    def get_polygons_points(self, by: str = "tuple", **_kw) -> dict:
        polys: dict[tuple[int, int], list] = {}
        for s in self._component.structures:
            if s.role != "device":
                continue
            key = tuple(s.layer)
            polys.setdefault(key, []).append([[float(x), float(y)] for x, y in s.polygon])
        return polys

    @property
    def ports(self):
        return [_ShimPort(p) for p in self._component.ports]


@register_solver
class BeamzSolver(Solver):
    """beamz JAX FDTD adapter (tier: full-service, execution: local, free)."""

    name = "beamz"
    capabilities = SolverCapabilities(
        tier="full",
        execution="local",
        supports_dispersion=False,  # v1: single (n_core, n_clad) pair
        supports_sidewall_angle=False,  # v1: vertical extrusion
        supports_multimode=False,  # v1: TE mode 1
        supports_gpu=True,  # jax backend selects automatically
        cost_model="free",
    )

    def __init__(
        self,
        *args,
        gf_component=None,
        n_core: float | None = None,
        n_clad: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # agnostic setup: components converted via layout.gdsfactory carry
        # their source component; the kwarg remains as an explicit override
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

    def _tech_dict(self) -> dict | None:
        t = self.technology
        if t is None:
            return None
        return t.to_solver_dict() if hasattr(t, "to_solver_dict") else t

    def _device_layer(self) -> dict | None:
        """The single device layer present in the component (v1 restriction)."""
        tech = self._tech_dict()
        if tech is None:
            return None
        present = {tuple(s.layer) for s in self.component.structures if s.role == "device"}
        matches = [d for d in tech["device"] if tuple(d["layer"]) in present]
        return matches[0] if len(matches) == 1 else None

    def _resolve_index(self, material: dict, kwarg: float | None, label: str):
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
                    f"port {port.name!r} faces {port.direction} deg: beamz v1 handles "
                    "x-oriented ports only (F14: modal wave separation on y-normal "
                    "monitors mis-normalizes by tens of dB - measured S11 +40 dB on a "
                    "vertical straight). Orient the device along x, or use tidy3d/"
                    "lumerical for devices with y-facing ports."
                )
        if self.gf_component is None and not any(
            s.role == "device" for s in self.component.structures
        ):
            problems.append(
                "BeamzSolver needs geometry: pass a gdsfactory component (gf_component=) "
                "or a component carrying device-layer polygons (any GDS/KLayout/SiEPIC source)"
            )
        if self.technology is None:
            problems.append("BeamzSolver requires a technology")
            return problems
        if self._device_layer() is None:
            problems.append(
                "BeamzSolver v1 supports exactly one device layer present in the component"
            )
        if tuple(self.spec.modes) != (1,):
            problems.append(f"BeamzSolver v1 supports modes=(1,) (TE); got {self.spec.modes}")
        n_core, n_clad = self._indices()
        if n_core is None:
            problems.append(
                "cannot resolve core refractive index: pass n_core=, or give the device "
                "material an 'rii' reference or an 'nk' constant (beamz has no vendor DB)"
            )
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
        from beamz.design.io import gdsf
        from beamz.devices.sources.signals import gaussian_band_pulse

        s = self.spec
        d = self._device_layer()
        n_core, n_clad = self._indices()
        wl0 = s.wavelength_center_um * UM
        core_t = abs(d["z_span"]) * UM
        extension = _PML_XY + _MONITOR_CLEARANCE + 1.0 * UM

        dx, dt = beamz.dxdt(
            wl0, n_max=n_core, dims=3, safety_factor=0.999, points_per_wavelength=s.mesh
        )

        # gdsfactory object when present, else a shim over the polygons gds_fdtd
        # already extruded from any source (KLayout/SiEPIC, raw GDS). Both drive
        # beamz's own tuned extrusion/port pipeline identically.
        import_source = (
            self.gf_component
            if self.gf_component is not None
            else _ComponentImportShim(self.component)
        )
        prepared = gdsf.prepare_component(
            import_source,
            layer=tuple(d["layer"]),
            n_core=n_core,
            n_clad=n_clad,
            core_thickness=core_t,
            clad_below=_PORT_MARGIN + _Z_PADDING,
            clad_above=_PORT_MARGIN + _Z_PADDING,
            xy_padding=extension,
            z_padding=_Z_PADDING + _PML_Z,
            extension=extension,
            port_overlap=0.0,
            use_pdk_layer_stack=False,
        )
        design, ports = prepared["design"], prepared["ports"]
        grid = design.rasterize(resolution=dx)

        freqs = np.linspace(
            C_M_S / (s.wavelength_end * UM),
            C_M_S / (s.wavelength_start * UM),
            s.wavelength_points,
            dtype=np.float32,
        )

        # port mode-plane geometry (mirrors the beamz reference example)
        planes = {}
        for name, port in ports.items():
            width = float(port.get("width", 0.5 * UM))
            span = _MODE_PLANE_SCALE * (width + 2 * _PORT_MARGIN)
            z_span = _MODE_PLANE_SCALE * (core_t + 2 * _PORT_MARGIN)
            z_center = float(port.get("z_center", design.depth / 2))
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

        pulse = gaussian_band_pulse(
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
        if self._artifacts is None:
            self.build()
        shape = self._artifacts.summary["grid_shape"]
        cells = int(np.prod(shape))
        return ResourceEstimate(
            grid_cells=cells,
            memory_gb=cells * 4 * 12 / 1e9,  # ~12 float32 field/eps arrays
            n_simulations=self._artifacts.summary["n_simulations"],
            cost_hint="free local compute (JAX; CPU works, GPU if available)",
        )

    def plot_fields(self, axis: str = "z", savefig: str | None = None):
        """|E|² profile at the core-center z-plane (first excitation)."""
        import matplotlib.pyplot as plt

        if axis != "z":
            raise JobValidationError("BeamzSolver v1 records the z-plane profile only")
        fields = getattr(self, "_field_z", None)
        if fields is None:
            raise RuntimeError(
                "no field data: include 'z' in spec.field_monitors and call run() first"
            )
        mag2 = sum(np.abs(np.squeeze(v)) ** 2 for v in fields.values())
        meta = self._field_z_meta
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(
            np.asarray(mag2),  # rows are already y (see reshape note in run())
            origin="lower",
            extent=(0, meta["width_um"], 0, meta["height_um"]),
            cmap=DEFAULT_CMAP,
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, label="|E|²")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.set_title(f"|E|² (z-plane) — excitation {meta['source']}, center frequency")
        if savefig:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")
        return fig, ax

    def run(self) -> SMatrix:
        """One FDTD run per excited port; full S-matrix via modal DFT extraction."""
        if self._artifacts is None:
            self.build()
        import beamz
        from beamz import ModeMonitor, ModeSource, PortSpec, Simulation

        nat = self._artifacts.native
        design, grid, ports, planes = nat["design"], nat["grid"], nat["ports"], nat["planes"]
        pulse, dx, dt, freqs = nat["pulse"], nat["dx"], nat["dt"], nat["freqs"]

        monitor_cfg = {
            "record_fields": False,
            "dft_enabled": True,
            "dft_frequencies": freqs,
            "dft_components": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            "dft_window": "none",
            "dft_record_every_step": True,
        }

        entries = []
        port_names = sorted(ports)
        for src_name in port_names:
            src = ports[src_name]
            geo = planes[src_name]
            source_plane = _port_plane(
                src,
                span=geo["span"],
                z_span=geo["z_span"],
                z_center=geo["z_center"],
                offset=_SOURCE_OFFSET,
            )
            src_center = tuple(0.5 * (source_plane[0][i] + source_plane[1][i]) for i in range(3))
            source = ModeSource(
                grid=grid,
                center=src_center,
                width=geo["span"],
                height=geo["z_span"],
                wavelength=nat["wl0"],
                pol="te",
                signal=pulse.signal,
                direction=src["direction"],
            )
            source.initialize(grid.permittivity, dx, dt=dt)

            monitors = []
            for name in port_names:
                offset = (
                    _SOURCE_OFFSET + _SOURCE_TO_MONITOR
                    if name == src_name
                    else _OUTPUT_MONITOR_OFFSET
                )
                g = planes[name]
                plane = _port_plane(
                    ports[name],
                    span=g["span"],
                    z_span=g["z_span"],
                    z_center=g["z_center"],
                    offset=offset,
                )
                monitors.append(
                    ModeMonitor(
                        start=plane[0],
                        end=plane[1],
                        name=name,
                        direction=ports[name]["direction"],
                        polarization="te",
                        # NO reference monitor: incident power comes from the
                        # source port's own ModeMonitor via modal wave
                        # separation, which is symmetric by construction. The
                        # reference-monitor normalization used by the beamz
                        # example mis-scales '-'-directed sources by ~+2 dB
                        # (found by live validation; the example only excites
                        # a '+x' port so never hits it).
                        reference_monitor=None,
                        **monitor_cfg,
                    )
                )

            specs = {}
            for m in monitors:
                port_obj = m.to_port()
                specs[m.name] = PortSpec(
                    name=m.name,
                    monitor_name=m.name,
                    direction=str(ports[m.name]["direction"]),
                    polarization="te",
                    mode_index=0,
                    reference_monitor=None,  # see note on the monitors above
                    incident_wave=port_obj.incident_wave,
                    scattered_wave=port_obj.scattered_wave,
                )

            field_monitor = None
            if "z" in self.spec.field_monitors and src_name == port_names[0]:
                # standardized field profile: full-XY plane at the core center,
                # DFT at the center frequency (first excitation only)
                z_mid = planes[port_names[0]]["z_center"]
                f0 = float(np.median(np.asarray(freqs, dtype=float)))
                field_monitor = beamz.Monitor(
                    start=(0.0, 0.0, z_mid),
                    end=(float(design.width), float(design.height), z_mid),
                    name="field_z",
                    record_fields=False,
                    dft_enabled=True,
                    dft_frequencies=np.asarray([f0], dtype=np.float32),
                    dft_components=("Ex", "Ey", "Ez"),
                    dft_window="none",
                    dft_record_every_step=True,
                )

            sim = Simulation(
                design=design,
                sources=[source],
                monitors=monitors if field_monitor is None else [*monitors, field_monitor],
                boundaries=[
                    beamz.PML(
                        edges=["left", "right", "top", "bottom"],
                        thickness=_PML_XY,
                        formulation="sigma",
                    ),
                    beamz.PML(edges=["front", "back"], thickness=_PML_Z, formulation="sigma"),
                ],
                time=pulse.time,
                resolution=dx,
            )
            sim.run_compiled_until_decay(
                monitors,
                min_time_s=pulse.source_end_time + pulse.tail_time,
                lookback_records=_LOOKBACK,
                decay_ratio=_DECAY_RATIO,
                progress=False,
            )
            if field_monitor is not None:
                # get_dft_component returns (nfreq, ncells) with the plane
                # FLATTENED; recover (Nx, Ny) from the known plane extents,
                # searching ±3 cells for grid snapping
                raw = {
                    c: np.asarray(field_monitor.get_dft_component(c)) for c in ("Ex", "Ey", "Ez")
                }
                ncells = int(np.asarray(raw["Ex"]).reshape(-1).shape[0])
                nx0 = round(float(design.width) / dx)
                nx = next((n for n in range(max(nx0 - 3, 1), nx0 + 4) if ncells % n == 0), None)
                if nx is None:
                    raise RuntimeError(f"cannot factor field plane: {ncells} cells vs Nx~{nx0}")
                # beamz grids are (z, y, x)-ordered: the flattened plane is
                # y-major with x fastest -> reshape to (Ny, Nx)
                self._field_z = {
                    c: np.asarray(v).reshape(-1)[:ncells].reshape(ncells // nx, nx)
                    for c, v in raw.items()
                }
                self._field_z_meta = {
                    "width_um": float(design.width) / UM,
                    "height_um": float(design.height) / UM,
                    "source": src_name,
                }
            result = sim.get_S_matrix_modal_dft(
                source_port=specs[src_name],
                ports=list(specs.values()),
                output_ports=list(specs.values()),
                frequencies=freqs,
                as_sax=False,
                return_diagnostics=False,
            )
            # beamz returns {"s_matrix": {...}, "diagnostics": ...} with
            # diagnostics enabled, and the bare matrix dict without
            s_map = result["s_matrix"] if "s_matrix" in result else result
            f_asc = np.asarray(freqs, dtype=float)
            order = np.argsort(f_asc)
            for out_name in port_names:
                col = np.asarray(s_map[(out_name, src_name)], dtype=complex)
                entries.append((src_name, out_name, 1, 1, f_asc[order], col[order]))

        return SMatrix.from_entries(entries, name=self.component.name, port_names=port_names)
