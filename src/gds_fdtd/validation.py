"""
gds_fdtd simulation toolbox.

Cross-solver validation: run the SAME component/technology/spec through several
engines and quantify how well their S-matrices agree.

    report = validate_across([Tidy3DSolver, LumericalSolver], comp, tech, spec,
                             cache_dir=".cache")
    print(report.summary())
    report.plot(out=2, in_=1, savefig="agreement.png")

``compare_smatrices`` is the pure core (no runs, no cost) — feed it recorded
results to reproduce a comparison offline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .convergence import max_delta_db
from .smatrix import SMatrix
from .spec import SimulationSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .geometry import Component
    from .solvers.base import Solver
    from .technology import Technology

logger = logging.getLogger(__name__)


@dataclass
class CrossSolverReport:
    """Pairwise S-matrix agreement between engines on one job."""

    smatrices: dict[str, SMatrix]
    pairwise_db: dict[tuple[str, str], float]  # max |ΔS| per solver pair

    @property
    def worst_db(self) -> float:
        return max(self.pairwise_db.values()) if self.pairwise_db else 0.0

    def passed(self, tol_db: float = 0.5) -> bool:
        return self.worst_db <= tol_db

    def summary(self) -> str:
        lines = [f"cross-solver agreement ({len(self.smatrices)} engines):"]
        for (a, b), d in sorted(self.pairwise_db.items()):
            lines.append(f"  {a} vs {b}: max |ΔS| = {d:.4f} dB")
        lines.append(f"  worst pair: {self.worst_db:.4f} dB")
        return "\n".join(lines)

    def plot(
        self, out: Any, in_: Any, mode_out: int = 1, mode_in: int = 1, savefig: str | None = None
    ) -> tuple[Any, Any]:
        """Overlay one S-parameter magnitude from every engine."""
        import matplotlib.pyplot as plt

        from .plotting import rdbu_colors

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (name, sm), color in zip(
            self.smatrices.items(), rdbu_colors(len(self.smatrices)), strict=True
        ):
            ax.plot(
                sm.wavelength_um,
                sm.magnitude_db(out, in_, mode_out, mode_in),
                "o-",
                markersize=3,
                color=color,
                label=name,
            )
        ax.set_xlabel("wavelength [µm]")
        ax.set_ylabel(f"|S({out}←{in_})|² [dB]")
        ax.set_title(f"cross-solver agreement (worst pair {self.worst_db:.3f} dB)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if savefig:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")
        return fig, ax


def compare_smatrices(smatrices: dict[str, SMatrix], floor_db: float = -30.0) -> CrossSolverReport:
    """Pairwise comparison of already-computed S-matrices (offline, free).

    ``floor_db`` ignores entries where both engines are below it — deep
    stopband/crosstalk values disagree wildly in dB while being equally
    "zero"; the default −30 dB focuses the metric on meaningful paths.
    """
    if len(smatrices) < 2:
        raise ValueError("need at least two S-matrices to compare")
    names = list(smatrices)
    pairwise: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairwise[(a, b)] = max_delta_db(smatrices[a], smatrices[b], floor_db=floor_db)
    return CrossSolverReport(smatrices=dict(smatrices), pairwise_db=pairwise)


def validate_across(
    solvers: Sequence[type[Solver] | Solver],
    component: Component,
    technology: Technology | None = None,
    spec: SimulationSpec | None = None,
    *,
    cache_dir: str | Path | None = None,
    workdir: str | Path | None = None,
    floor_db: float = -30.0,
) -> CrossSolverReport:
    """Run one job on every engine and report pairwise agreement.

    Accepts solver classes (instantiated with the shared job) or
    pre-configured instances. WARNING: each engine pays one full run();
    pass ``cache_dir`` to make reruns free.
    """
    spec = spec if spec is not None else SimulationSpec()
    results: dict[str, SMatrix] = {}
    for s in solvers:
        solver = s if not isinstance(s, type) else s(component, technology, spec, workdir=workdir)
        problems = solver.validate()
        if problems:
            raise ValueError(f"{solver.name}: job invalid: {problems}")
        logger.info("cross-solver validation: running %s", solver.name)
        results[solver.name] = (
            solver.run_cached(cache_dir) if cache_dir is not None else solver.run()
        )
    return compare_smatrices(results, floor_db=floor_db)
