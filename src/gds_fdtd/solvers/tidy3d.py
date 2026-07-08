"""
gds_fdtd simulation toolbox.

Tidy3DSolver: the tidy3d (>=2.11) adapter on the Phase-3 Solver contract
(WP3.1c). Composition strategy: the scene construction reuses the validated
legacy ``fdtd_solver_tidy3d`` machinery, but ONLY inside ``build()`` — the
constructor stays pure per the contract, and ``run()`` is the only method
that talks to the tidy3d cloud.

The legacy class remains fully supported; it and this adapter share one
implementation, so the WP1.5 offline tests and the live cloud validation
cover both.
"""

from __future__ import annotations

import os
import tempfile

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
        import tidy3d  # noqa: F401

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
            tech_dict = (
                self.technology.to_legacy_dict()
                if hasattr(self.technology, "to_legacy_dict")
                else self.technology
            )
            for i, d in enumerate(tech_dict.get("device", [])):
                mat = d.get("material", {})
                if "tidy3d_db" not in mat:
                    problems.append(
                        f"device layer {i} has no 'tidy3d_db' material entry "
                        "(rii/other sources are wired to tidy3d in WP5.x)"
                    )
        return problems

    def build(self) -> SetupArtifacts:
        """Construct the ModalComponentModeler (offline; no cloud access)."""
        problems = self.validate()
        if problems:
            raise JobValidationError("cannot build: " + "; ".join(problems))

        from ..solver_tidy3d import fdtd_solver_tidy3d

        workdir = (
            str(self.workdir)
            if self.workdir is not None
            else tempfile.mkdtemp(prefix="gds_fdtd_t3d_")
        )
        s = self.spec
        legacy = fdtd_solver_tidy3d(
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
            working_dir=workdir,
        )
        self._legacy = legacy
        self._artifacts = SetupArtifacts(
            native=legacy.component_modeler,
            files={"gds": legacy._gds_filepath},
            summary={
                "n_ports": len(legacy.smatrix_ports),
                "n_modes": len(s.modes),
                "n_simulations": len(legacy.component_modeler.sim_dict),
                "run_time_s": float(legacy.base_simulation.run_time),
            },
        )
        return self._artifacts

    def estimate(self) -> ResourceEstimate:
        if self._artifacts is None:
            self.build()
        modeler = self._artifacts.native
        sim = modeler.simulation
        cells = int(np.prod(sim.grid.num_cells))
        return ResourceEstimate(
            grid_cells=cells,
            n_simulations=len(modeler.sim_dict),
            cost_hint="cloud FlexCredits; use tidy3d.web.estimate_cost per task before running",
        )

    def run(self) -> SMatrix:
        """Run on the tidy3d cloud (SPENDS FlexCredits) and return the SMatrix."""
        if self._artifacts is None:
            self.build()
        import tidy3d.web as web

        workdir = (
            str(self.workdir)
            if self.workdir is not None
            else os.path.dirname(str(self._artifacts.files["gds"]))
        )
        modeler_data = web.run(
            self._artifacts.native,
            task_name=f"gdsfdtd_{self.component.name}",
            path=os.path.join(workdir, f"{self.component.name}_modeler.hdf5"),
            verbose=True,
        )
        self._modeler_data = modeler_data
        da = modeler_data.smatrix()
        return self._dataarray_to_smatrix(da)

    def plot_fields(self, axis: str = "z", savefig: str | None = None):
        """Field profile from the first excitation's SimulationData."""
        data = getattr(self, "_modeler_data", None)
        if data is None:
            raise SolverError("run() has not completed; no field data available")
        return plot_tidy3d_fields(data, axis=axis, savefig=savefig)

    # ---------------- conversion ----------------

    def _dataarray_to_smatrix(self, da) -> SMatrix:
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


def plot_tidy3d_fields(modeler_data, axis: str = "z", savefig: str | None = None):
    """Plot |E| of the '{axis}_field' monitor from a ModalComponentModelerData.

    Shared by Tidy3DSolver.plot_fields and the legacy solver's
    visualize_field_monitors (the WP1.9/WP4.1 promise, redeemed here).
    """
    import matplotlib.pyplot as plt

    sim_data_map = modeler_data.data
    items = (
        list(sim_data_map.items())
        if hasattr(sim_data_map, "items")
        else list(enumerate(sim_data_map))
    )
    if not items:
        raise SolverError("modeler data contains no per-task simulation data")
    task_name, sim_data = items[0]
    monitor_name = f"{axis}_field"
    # the monitor is broadband: select the center frequency for a 2D plot
    freqs = sim_data[monitor_name].Ex.coords["f"].values
    f0 = float(freqs[len(freqs) // 2])
    fig, ax = plt.subplots(figsize=(9, 5))
    sim_data.plot_field(monitor_name, "E", val="abs", f=f0, ax=ax)
    ax.set_title(f"|E| ({monitor_name}) — excitation {task_name}")
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig, ax
