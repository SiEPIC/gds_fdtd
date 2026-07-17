"""
gds_fdtd simulation toolbox.

Tidy3DSolver: the tidy3d (>=2.11) adapter on the Phase-3 Solver contract.
Scene construction runs inside ``build()`` via the internal
``_tidy3d_engine`` module; the constructor stays pure per the contract, and
``run()`` is the only method that talks to the tidy3d cloud.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
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


def probe_tidy3d() -> str | None:
    """None if tidy3d is importable, else the reason it isn't."""
    try:
        import tidy3d  # noqa: F401  (availability probe)

        return None
    except Exception as e:  # pragma: no cover - env dependent
        return f"tidy3d not importable: {e}"


@register_solver
class Tidy3DSolver(Solver):
    """Tidy3D cloud FDTD adapter (tier: full-service, execution: cloud)."""

    name = "tidy3d"
    capabilities = SolverCapabilities(
        tier="full",
        execution="cloud",
        supports_dispersion=True,
        supports_sidewall_angle=True,
        supports_multimode=True,
        supports_gpu=True,  # cloud hardware, not user-selectable
        cost_model="credits",
    )

    @staticmethod
    def probe_available() -> str | None:
        return probe_tidy3d()

    # ---------------- lifecycle ----------------

    def validate(self) -> list[str]:
        problems = []
        reason = self.probe_available()
        if reason:
            problems.append(reason)
        if not self.component.ports:
            problems.append("component has no ports")
        if self.technology is None:
            problems.append("Tidy3DSolver requires a technology (materials)")
        else:
            from ..materials.select import check_materials

            tech_dict = (
                self.technology.to_solver_dict()
                if hasattr(self.technology, "to_solver_dict")
                else self.technology
            )
            problems += check_materials(cast("dict[str, Any]", tech_dict), "tidy3d")
        return problems

    def build(self) -> SetupArtifacts:
        """Construct the ModalComponentModeler (offline; no cloud access)."""
        problems = self.validate()
        if problems:
            raise JobValidationError("cannot build: " + "; ".join(problems))

        from ._tidy3d_engine import _TidyEngine

        workdir = (
            str(self.workdir)
            if self.workdir is not None
            # Default to a fresh subdir under the current working directory.
            # Avoid the system temp dir: it is often a small RAM-backed tmpfs,
            # and large result downloads (multi-GB with field monitors) can
            # blow past its free space (OSError 28: No space left on device).
            else tempfile.mkdtemp(prefix="gds_fdtd_t3d_", dir=os.getcwd())
        )
        s = self.spec
        engine = _TidyEngine(
            component=self.component,
            tech=self.technology,
            wavelength_start=s.wavelength_start,
            wavelength_end=s.wavelength_end,
            wavelength_points=s.wavelength_points,
            mesh=s.mesh,
            boundary=list(s.boundary),
            symmetry=list(s.symmetry),
            z_min=s.z_min,
            z_max=s.z_max,
            width_ports=s.width_ports,
            depth_ports=s.depth_ports,
            buffer=s.buffer,
            modes=list(s.modes),
            mode_freq_pts=s.mode_freq_pts,
            run_time_factor=s.run_time_factor,
            field_monitors=list(s.field_monitors),
            field_monitor_positions=dict(s.field_monitor_positions),
            field_monitor_wavelengths=list(s.field_monitor_wavelengths),
            working_dir=workdir,
        )
        self._engine = engine
        self._artifacts = SetupArtifacts(
            native=engine.component_modeler,
            files={"gds": Path(engine._gds_filepath)},
            summary={
                "n_ports": len(engine.smatrix_ports),
                "n_modes": len(s.modes),
                "n_simulations": len(engine.component_modeler.sim_dict),
                "run_time_s": float(engine.base_simulation.run_time),
            },
        )
        return self._artifacts

    def estimate(self) -> ResourceEstimate:
        artifacts = self._artifacts if self._artifacts is not None else self.build()
        modeler = artifacts.native
        sim = modeler.simulation
        cells = int(np.prod(sim.grid.num_cells))
        return ResourceEstimate(
            grid_cells=cells,
            n_simulations=len(modeler.sim_dict),
            cost_hint="cloud FlexCredits; use tidy3d.web.estimate_cost per task before running",
        )

    def run(self) -> SMatrix:
        """Run on the tidy3d cloud (SPENDS FlexCredits) and return the SMatrix."""
        artifacts = self._artifacts if self._artifacts is not None else self.build()
        import tidy3d.web as web

        workdir = (
            str(self.workdir)
            if self.workdir is not None
            else os.path.dirname(str(artifacts.files["gds"]))
        )
        # Route tidy3d's own download temp files into the workdir too. Its
        # compressed-download tempfile (s3utils.download_gz_file) uses the
        # process temp dir with no override, which is often a small RAM-backed
        # tmpfs; multi-GB results then fail with OSError 28 (No space left).
        prev_tempdir = tempfile.tempdir
        tempfile.tempdir = workdir
        try:
            modeler_data = web.run(
                artifacts.native,
                task_name=f"gdsfdtd_{self.component.name}",
                path=os.path.join(workdir, f"{self.component.name}_modeler.hdf5"),
                verbose=True,
            )
        finally:
            tempfile.tempdir = prev_tempdir
        self._modeler_data = modeler_data
        da = modeler_data.smatrix()
        return self._dataarray_to_smatrix(da)

    def plot_fields(
        self,
        axis: str = "z",
        scale: str = "linear",
        savefig: str | None = None,
        wavelength_um: float | None = None,
        task: str | int | None = None,
        outline: bool = True,
    ) -> tuple[Any, Any]:
        """Field profile from one excitation's SimulationData.

        ``scale="db"`` gives a log view, ``wavelength_um`` picks the recorded
        frequency nearest that wavelength, ``task`` selects the excitation (an
        index or a port-name substring), and ``outline=True`` overlays the
        device geometry (see :func:`gds_fdtd.plotting.plot_field`).
        """
        data = getattr(self, "_modeler_data", None)
        if data is None:
            raise SolverError("run() has not completed; no field data available")
        from ..plotting import component_outlines

        lines = component_outlines(self.component, axis=axis) if outline else None
        return plot_tidy3d_fields(
            data,
            axis=axis,
            scale=scale,
            savefig=savefig,
            wavelength_um=wavelength_um,
            task=task,
            outline=lines,
        )

    # ---------------- conversion ----------------

    def _dataarray_to_smatrix(self, da: Any) -> SMatrix:
        """ModalPortDataArray -> canonical SMatrix (1-based mode ids)."""
        freqs = np.asarray(da.coords["f"].values, dtype=float)
        entries = []
        wanted = set(self.spec.modes)
        for p_in in da.coords["port_in"].values:
            for p_out in da.coords["port_out"].values:
                for m_in in da.coords["mode_index_in"].values:
                    for m_out in da.coords["mode_index_out"].values:
                        if (m_in + 1) not in wanted or (m_out + 1) not in wanted:
                            continue
                        s_c = da.sel(
                            port_in=p_in,
                            port_out=p_out,
                            mode_index_in=m_in,
                            mode_index_out=m_out,
                        ).values
                        entries.append(
                            (str(p_in), str(p_out), int(m_in) + 1, int(m_out) + 1, freqs, s_c)
                        )
        port_names = [str(p) for p in da.coords["port_in"].values]
        return SMatrix.from_entries(entries, name=self.component.name, port_names=port_names)


def plot_tidy3d_fields(
    modeler_data: Any,
    axis: str = "z",
    scale: str = "linear",
    savefig: str | None = None,
    wavelength_um: float | None = None,
    task: str | int | None = None,
    outline: Any = None,
) -> tuple[Any, Any]:
    """Plot ``|E|²`` of the '{axis}_field' monitor from a ModalComponentModelerData.

    Used by Tidy3DSolver.plot_fields. Extracts the field onto its true grid and
    renders via :func:`gds_fdtd.plotting.plot_field`, so ``scale="db"`` gives a
    log view. Drawn on the engine's own (non-uniform) mesh coordinates.

    Args:
        wavelength_um: pick the recorded frequency nearest this wavelength
            (the monitor stores every simulated point, or the ones chosen via
            ``spec.field_monitor_wavelengths``). None takes the band center.
        task: which excitation to show — an index, or a substring of the task
            name (e.g. a port name like ``"o1"``). None takes the first.
        outline: geometry outlines (list of Nx2 polygons in plot coordinates)
            passed through to :func:`~gds_fdtd.plotting.plot_field`.
    """
    from ..plotting import plot_field

    sim_data_map = modeler_data.data
    items = (
        list(sim_data_map.items())
        if hasattr(sim_data_map, "items")
        else list(enumerate(sim_data_map))
    )
    if not items:
        raise SolverError("modeler data contains no per-task simulation data")
    if task is None:
        task_name, sim_data = items[0]
    elif isinstance(task, int):
        task_name, sim_data = items[task]
    else:
        matches = [(n, d) for n, d in items if task in str(n)]
        if not matches:
            raise SolverError(
                f"no excitation task matching {task!r}; tasks: {[str(n) for n, _ in items]}"
            )
        task_name, sim_data = matches[0]
    monitor_name = f"{axis}_field"
    fd = sim_data[monitor_name]
    # the monitor is broadband: pick the requested wavelength, else band center
    c_um_s = 2.99792458e14  # speed of light [um/s] (tidy3d works in um units)
    freqs = np.asarray(fd.Ex.coords["f"].values)
    if wavelength_um is not None:
        f0 = float(freqs[np.argmin(np.abs(freqs - c_um_s / wavelength_um))])
    else:
        f0 = float(freqs[len(freqs) // 2])
    lda_um = c_um_s / f0
    mag2 = sum(
        np.abs(np.asarray(getattr(fd, c).sel(f=f0).values).squeeze()) ** 2
        for c in ("Ex", "Ey", "Ez")
    )
    h_name, v_name = {"z": ("x", "y"), "y": ("x", "z"), "x": ("y", "z")}[axis]
    h = np.asarray(fd.Ex.coords[h_name].values)
    v = np.asarray(fd.Ex.coords[v_name].values)
    return plot_field(
        mag2.T,  # (h, v) -> (v, h) for row=v, col=h
        x=h,
        y=v,
        scale=scale,
        title=f"|E|² ({monitor_name}) at {lda_um:.4f} um, excitation {task_name}",
        outline=outline,
        savefig=savefig,
    )
