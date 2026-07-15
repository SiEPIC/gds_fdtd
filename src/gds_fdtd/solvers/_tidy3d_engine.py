"""
gds_fdtd simulation toolbox.

Internal Tidy3D scene-building engine, the implementation behind the supported
``gds_fdtd.solvers.tidy3d.Tidy3DSolver`` adapter. The tidy3d-independent
geometry/port/domain setup lives in the sibling ``_tidy3d_base`` module.
Not a public module.
"""

import numpy as np
import tidy3d as td
from tidy3d.plugins.smatrix import ModalComponentModeler, Port

from gds_fdtd.errors import JobValidationError
from gds_fdtd.solvers._tidy3d_base import _TidyEngineBase


class _TidyEngine(_TidyEngineBase):
    """Tidy3D ComponentModeler scene builder + runner (internal engine)."""

    def __init__(self, *args, visualize: bool = True, **kwargs):
        """Initialize the Tidy3D solver by calling the parent constructor."""
        super().__init__(*args, **kwargs)
        self.visualize = visualize
        self.simulation = None
        self.setup()

    def setup(self) -> None:
        """Setup the Tidy3D simulation using ComponentModeler for S-matrix calculation."""
        self.logger.info("Starting Tidy3D solver setup")

        # Validate simulation parameters
        self._validate_simulation_parameters()

        # Export GDS with port extensions to working directory
        self._export_gds()
        self.logger.info(f"GDS exported to: {self._gds_filepath}")

        # Calculate frequencies for S-matrix calculation
        self.freqs = td.C_0 / np.linspace(
            self.wavelength_start, self.wavelength_end, self.wavelength_points
        )
        self.lda0 = (self.wavelength_end + self.wavelength_start) / 2
        self.freq0 = td.C_0 / self.lda0
        self.logger.debug(
            f"Frequency calculation: {len(self.freqs)} points from {self.freqs[0]:.2e} to {self.freqs[-1]:.2e} Hz"
        )

        # Create base simulation and ports for S-matrix calculation
        self.base_simulation = self._create_base_simulation()
        self.smatrix_ports = self._create_smatrix_ports()
        self.logger.info(f"Created base simulation with {len(self.smatrix_ports)} ports")

        # Create the modeler for S-matrix calculation.
        # tidy3d >=2.9 renamed ComponentModeler -> ModalComponentModeler and
        # moved web options (verbose/path_dir) to tidy3d.web.run.
        self.component_modeler = ModalComponentModeler(
            simulation=self.base_simulation,
            ports=self.smatrix_ports,
            freqs=self.freqs,
        )
        self.logger.info("ModalComponentModeler created successfully")

        # Print setup summary
        self._print_simulation_summary()
        total_mode_combinations = len(self.smatrix_ports) * len(self.modes)
        setup_info = f"Tidy3D solver setup complete with ComponentModeler: {len(self.smatrix_ports)} ports × {len(self.modes)} modes = {total_mode_combinations} mode combinations"
        self.logger.info(setup_info)

        self.logger.info("Tidy3D solver setup complete with ComponentModeler:")
        self.logger.info(
            f"  • {len(self.smatrix_ports)} ports × {len(self.modes)} modes = {total_mode_combinations} mode combinations"
        )
        self.logger.info("  • Multi-modal S-matrix calculation ready")
        self.logger.info(
            f"  • ComponentModeler will auto-generate task names for {self.component.name}"
        )

    def _create_base_simulation(self):
        """Create base simulation without sources/monitors for ComponentModeler."""
        device = self.component

        # Create structures
        structures = self._create_structures()

        # Add field monitor if requested
        monitors = []
        if self.field_monitors:
            for field_monitor_axis in self.field_monitors:
                field_monitor = self._create_field_monitor(
                    device, freqs=self.freqs, axis=field_monitor_axis
                )
                monitors.append(field_monitor)
                self.logger.debug(f"Created Tidy3D field monitor: {field_monitor.name}")

        # simulation domain size (in microns)
        sim_size = [self.span[0], self.span[1], self.span[2]]

        # Run time from the shared base-class helper (includes the group-index
        # slowdown factor) so run_time_factor means the same physics as in the
        # Lumerical solver (bug B11). max dimension converted um -> m.
        run_time = self._calculate_simulation_time(max(sim_size) * 1e-6)

        # Honor the user's boundary settings (previously hardcoded to PML on
        # all sides while the Lumerical solver obeyed them; bug B10).
        boundary_spec = self._create_boundary_spec()

        # Create base simulation (no sources - ComponentModeler adds them)
        base_sim = td.Simulation(
            size=sim_size,
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=self.mesh, wavelength=self.lda0),
            structures=structures,
            sources=[],  # No sources - ComponentModeler will add them
            monitors=monitors,
            run_time=run_time,
            boundary_spec=boundary_spec,
            center=(self.center[0], self.center[1], self.center[2]),
            symmetry=tuple(self.symmetry),
        )

        return base_sim

    def _create_boundary_spec(self) -> "td.BoundarySpec":
        """Map the solver's boundary strings to a tidy3d BoundarySpec.

        Supported names (case-insensitive, matching the Lumerical adapter):
        "PML", "Metal" (perfect electric conductor), "Periodic".
        """
        mapping = {
            "pml": td.PML,
            "metal": td.PECBoundary,
            "periodic": td.Periodic,
        }
        axis_boundaries = []
        for axis_name, boundary_name in zip("xyz", self.boundary, strict=True):
            key = str(boundary_name).lower()
            if key not in mapping:
                raise JobValidationError(
                    f"Unsupported boundary {boundary_name!r} for axis {axis_name}. "
                    f"Supported: {sorted(mapping)}"
                )
            b = mapping[key]()
            axis_boundaries.append(td.Boundary(minus=b, plus=b))
        return td.BoundarySpec(x=axis_boundaries[0], y=axis_boundaries[1], z=axis_boundaries[2])

    def _create_smatrix_ports(self):
        """Create Tidy3D Port objects for S-matrix calculation with multi-modal support."""
        ports = []

        for tp in self.tidy_ports:
            # Determine port direction and size based on the _TidyPort configuration
            if tp.span[0] is None:  # x-axis injection
                direction = "+" if tp.direction == "forward" else "-"
                size = [0, self.width_ports, self.depth_ports]
            elif tp.span[1] is None:  # y-axis injection
                direction = "+" if tp.direction == "forward" else "-"
                size = [self.width_ports, 0, self.depth_ports]
            else:
                raise JobValidationError(f"Invalid span configuration for port {tp.name}")

            # Create Tidy3D Port object with multi-modal support
            port = Port(
                center=tp.position,
                size=size,
                direction=direction,
                name=tp.name,
                mode_spec=td.ModeSpec(
                    num_modes=max(self.modes)
                ),  # Ensure enough modes are calculated
            )
            ports.append(port)

        return ports

    def _background_polygon(self):
        """Rectangle flush with the port extensions, CENTERED ON THE COMPONENT.

        Previously this square was centered at the origin (sized via
        abs(center)+span/2), so off-origin components got an oversized,
        origin-anchored substrate/superstrate (bug B9).
        """
        b = self.component.bounds
        half_x = b.x_span / 2 + 2 * self.buffer
        half_y = b.y_span / 2 + 2 * self.buffer
        cx, cy = b.x_center, b.y_center
        return [
            (cx - half_x, cy - half_y),
            (cx + half_x, cy - half_y),
            (cx + half_x, cy + half_y),
            (cx - half_x, cy + half_y),
        ]

    @staticmethod
    def _is_background(structure) -> bool:
        """Background = substrate/superstrate, by ROLE, with a
        name-sniffing fallback only for td.Structure objects in tests."""
        role = getattr(structure, "role", None)
        if role is not None:
            return role in ("substrate", "superstrate")
        n = structure.name.lower()
        # name fallback (td.Structure has no role); tolerate the historical
        # "Subtrate" typo present in components built by older versions
        return "substrate" in n or "superstrate" in n or "subtrate" in n

    def _medium(self, material, name: str = "material"):
        """Build the tidy3d medium for one material, honoring its source
        selection (see gds_fdtd.materials.select): the engine's own database
        model (``eda``), a dispersive refractiveindex.info fit (``rii``), or a
        constant ``nk``. A pre-built medium object passes straight through.
        """
        if not isinstance(material, dict):
            return material
        from gds_fdtd.materials.select import select_source

        src = select_source(material, "tidy3d", name=name)
        if src == "eda":
            from gds_fdtd.simprocessor import _load_tidy3d_material

            return _load_tidy3d_material(material["tidy3d_db"])
        if src == "rii":
            from gds_fdtd.materials.rii import load_rii_material

            ref = material["rii"]
            ref = ref.model_dump() if hasattr(ref, "model_dump") else ref
            wl = np.linspace(
                self.wavelength_start, self.wavelength_end, max(self.wavelength_points, 2)
            )
            mat = load_rii_material(ref["shelf"], ref["book"], ref["page"])
            return mat.to_tidy3d_medium(wavelength_um=wl)
        nk = material["nk"]
        n, k = (nk[0], nk[1]) if isinstance(nk, (list, tuple)) else (nk, 0.0)
        if k:  # lossy constant: td.Medium takes REAL permittivity + conductivity
            return td.Medium.from_nk(n=float(n), k=float(k), freq=td.C_0 / self.lda0)
        return td.Medium(permittivity=float(n) ** 2)

    def _to_td_structure(self, s):
        """Convert one geometry.Structure into a td.Structure."""
        if s.z_span < 0:
            bounds = (s.z_base + s.z_span, s.z_base)
        else:
            bounds = (s.z_base, s.z_base + s.z_span)

        polygon = self._background_polygon() if self._is_background(s) else s.polygon

        return td.Structure(
            geometry=td.PolySlab(
                vertices=polygon,
                slab_bounds=bounds,
                axis=2,
                sidewall_angle=(90 - s.sidewall_angle) * (np.pi / 180),
            ),
            medium=self._medium(s.material, name=s.name),
            name=s.name,
        )

    def _create_structures(self):
        """Create Tidy3D structure objects from the component."""
        device = self.component

        structures = [self._to_td_structure(s) for s in device.structures]

        # extend ports beyond sim region with 2*buffer
        for p in device.ports:
            structures.append(
                td.Structure(
                    geometry=td.PolySlab(
                        vertices=p.polygon_extension(buffer=2 * self.buffer),
                        slab_bounds=(
                            p.center[2] - p.height / 2,
                            p.center[2] + p.height / 2,
                        ),
                        axis=2,
                        sidewall_angle=(90 - device.structures[0].sidewall_angle) * (np.pi / 180),
                    ),
                    medium=self._medium(p.material, name=f"port_{p.name}"),
                    name=f"port_{p.name}",
                )
            )
        return structures

    def _create_field_monitor(self, device, freqs=2e14, axis="z", z_center=None):
        """Create a field monitor for the specified axis."""
        # identify a device field z_center if None
        if z_center is None:
            # per-LAYER average (matching the old per-list semantics)
            z_by_layer = {}
            for s in device.structures:
                if s.role == "device":
                    z_by_layer.setdefault(tuple(s.layer), s.z_base + s.z_span / 2)
            z_center = np.average(list(z_by_layer.values()))
        # center the monitor on the component, not the origin (bug B9)
        cx, cy = device.bounds.x_center, device.bounds.y_center
        if axis == "z":
            center = [cx, cy, z_center]
            size = [td.inf, td.inf, 0]
        elif axis == "y":
            center = [cx, cy, z_center]
            size = [td.inf, 0, td.inf]
        elif axis == "x":
            center = [cx, cy, z_center]
            size = [0, td.inf, td.inf]
        else:
            raise JobValidationError(
                f"Invalid axis {axis!r} for field monitor. Valid selections are 'x', 'y', 'z'."
            )
        return td.FieldMonitor(
            center=center,
            size=size,
            freqs=freqs,
            name=f"{axis}_field",
        )
