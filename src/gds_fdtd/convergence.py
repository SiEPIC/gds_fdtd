"""
gds_fdtd simulation toolbox.

Convergence sweeps (WP5.5 item 1): rerun one job while stepping a single
SimulationSpec field (mesh, run_time_factor, buffer, ...) and measure how
much the S-matrix still moves between successive values. Generalizes the
hand-rolled mesh sweeps of the legacy examples.

    report = sweep(Tidy3DSolver, component, tech, spec,
                   field="mesh", values=[6, 10, 14, 18],
                   cache_dir=".cache")   # reruns are free
    report.recommend(tol_db=0.05)        # first converged value
    report.plot(savefig="convergence.png")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .smatrix import SMatrix
from .spec import SimulationSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .geometry import Component
    from .solvers.base import Solver
    from .technology import Technology

logger = logging.getLogger(__name__)


def _align_ports(a: SMatrix, b: SMatrix) -> list[tuple[str, str]]:
    """Pair up ports across two S-matrices: by exact name, else by the
    package-wide trailing-digit id convention ('opt1' <-> '1')."""
    if set(a.port_names) == set(b.port_names):
        return [(n, n) for n in a.port_names]

    def digits(name: str) -> str:
        return "".join(ch for ch in name if ch.isdigit())

    da = {digits(n): n for n in a.port_names}
    db = {digits(n): n for n in b.port_names}
    if "" in da or "" in db or set(da) != set(db) or len(da) != a.n_ports:
        raise ValueError(
            f"cannot align ports {a.port_names} vs {b.port_names}: names differ "
            "and trailing-digit ids do not match one-to-one"
        )
    return [(da[k], db[k]) for k in sorted(da)]


def max_delta_db(a: SMatrix, b: SMatrix, floor_db: float = -30.0) -> float:
    """Worst-case |Δ|S|²| in dB between two S-matrices of the same job.

    Ports are aligned by name, or by trailing-digit id when the two solvers
    named them differently ('opt1' <-> '1'); ``b`` is interpolated onto the
    overlapping part of ``a``'s frequency grid. Entries are compared only
    where both matrices measured them and at least one sits above
    ``floor_db``: entries near the noise floor (reflections of a well-matched
    device, deep crosstalk) swing tens of dB between mesh settings while
    being equally "zero" — measured live on tidy3d, mesh 6→8 moved S11-type
    entries 17 dB while the thru path moved 0.07 dB. The −30 dB default keeps
    the metric on paths that carry meaningful power.
    """
    pairs = _align_ports(a, b)
    if a.n_modes != b.n_modes:
        raise ValueError(f"mode counts differ: {a.n_modes} vs {b.n_modes}")

    f_lo, f_hi = max(a.f.min(), b.f.min()), min(a.f.max(), b.f.max())
    if f_lo > f_hi:
        raise ValueError("frequency grids do not overlap")
    mask = (a.f >= f_lo) & (a.f <= f_hi)
    f = a.f[mask]

    worst = 0.0
    floor_lin = 10 ** (floor_db / 20)
    for out_a, out_b in pairs:
        for in_a, in_b in pairs:
            for m_out in range(1, a.n_modes + 1):
                for m_in in range(1, a.n_modes + 1):
                    sa = np.abs(a.sel(out_a, in_a, m_out, m_in))[mask]
                    sb_full = np.abs(b.sel(out_b, in_b, m_out, m_in))
                    sb = np.interp(f, b.f, sb_full)
                    ok = np.isfinite(sa) & np.isfinite(sb) & (np.maximum(sa, sb) > floor_lin)
                    if not np.any(ok):
                        continue
                    with np.errstate(divide="ignore"):
                        d = np.abs(20 * np.log10(sa[ok] / np.maximum(sb[ok], 1e-300)))
                    worst = max(worst, float(np.max(d)))
    return worst


@dataclass
class ConvergenceReport:
    """Result of a convergence sweep over one SimulationSpec field."""

    field: str
    values: list
    smatrices: list[SMatrix]
    deltas_db: list[float] = dc_field(default_factory=list)  # len(values) - 1

    def recommend(self, tol_db: float = 0.05):
        """Smallest swept value whose S-matrix moved < tol_db from the
        previous step; None if the sweep never converged."""
        for i, d in enumerate(self.deltas_db):
            if d < tol_db:
                return self.values[i + 1]
        return None

    def summary(self) -> str:
        lines = [f"convergence sweep over spec.{self.field}:"]
        for i, d in enumerate(self.deltas_db):
            lines.append(
                f"  {self.field}={self.values[i]} -> {self.values[i + 1]}: max |ΔS| = {d:.4f} dB"
            )
        rec = self.recommend()
        lines.append(f"  recommended: {self.field}={rec}" if rec is not None else "  NOT converged")
        return "\n".join(lines)

    def plot(self, tol_db: float = 0.05, savefig: str | None = None):
        """Delta-vs-value plot (semilogy) with the tolerance line."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(self.values[1:], self.deltas_db, "o-", label="max |ΔS| vs previous")
        ax.axhline(tol_db, color="tab:red", linestyle="--", label=f"tolerance {tol_db} dB")
        ax.set_xlabel(f"spec.{self.field}")
        ax.set_ylabel("max |ΔS| [dB]")
        ax.set_title(f"S-matrix convergence vs {self.field}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if savefig:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")
        return fig, ax


def sweep(
    solver_cls: type[Solver],
    component: Component,
    technology: Technology | None = None,
    spec: SimulationSpec | None = None,
    *,
    field: str = "mesh",
    values: Sequence,
    cache_dir: str | Path | None = None,
    workdir: str | Path | None = None,
    floor_db: float = -30.0,
) -> ConvergenceReport:
    """Run the same job at each ``spec.<field> = value`` and compare results.

    WARNING: each distinct value costs one full run() (money/licenses/time);
    pass ``cache_dir`` so repeated sweeps only pay for genuinely new points.
    """
    base = spec if spec is not None else SimulationSpec()
    if field not in type(base).model_fields:
        raise ValueError(f"SimulationSpec has no field {field!r}")

    smatrices: list[SMatrix] = []
    for v in values:
        s = base.model_copy(update={field: v})
        solver = solver_cls(component, technology, s, workdir=workdir)
        problems = solver.validate()
        if problems:
            raise ValueError(f"{field}={v}: job invalid: {problems}")
        logger.info("convergence sweep: %s=%s", field, v)
        sm = solver.run_cached(cache_dir) if cache_dir is not None else solver.run()
        smatrices.append(sm)

    deltas = [
        max_delta_db(smatrices[i + 1], smatrices[i], floor_db=floor_db)
        for i in range(len(smatrices) - 1)
    ]
    return ConvergenceReport(
        field=field, values=list(values), smatrices=smatrices, deltas_db=deltas
    )
