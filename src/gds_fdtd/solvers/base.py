"""
gds_fdtd simulation toolbox.

The Phase-3 solver contract (WP3.1b). Every engine adapter implements:

    validate() -> list[str]        # human-readable problems; [] = ok
    build()    -> SetupArtifacts   # native scene, OFFLINE, serializable
    estimate() -> ResourceEstimate # cells/memory/cost hints, offline
    run()      -> SMatrix          # the ONLY method allowed to spend
                                   # money / licenses / GPU time

Constructors MUST be cheap and pure: no disk writes, no network, no license
checks (rule 10 / the remote-compute invariant in MODERNIZATION_PLAN.md).
The legacy ``fdtd_solver`` hierarchy remains in gds_fdtd.solver until the
adapters are ported (WP3.1c/d); nothing imports this module yet besides the
conformance tests and FakeSolver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..geometry import Component
from ..smatrix import SMatrix
from ..spec import SimulationSpec
from ..technology import Technology


class SolverCapabilities(BaseModel):
    """What an engine adapter can and cannot do (declared, not probed)."""

    model_config = ConfigDict(frozen=True)

    tier: Literal["full", "kernel"]
    execution: Literal["local", "cloud"]
    supports_dispersion: bool
    supports_sidewall_angle: bool
    supports_multimode: bool
    supports_gpu: bool
    cost_model: Literal["free", "licensed", "credits"]


@dataclass
class SetupArtifacts:
    """Everything build() produced: native scene + any files written."""

    native: Any = None
    files: dict[str, Path] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceEstimate:
    """Offline resource estimate; fields are hints, None = unknown."""

    grid_cells: int | None = None
    memory_gb: float | None = None
    n_simulations: int | None = None
    cost_hint: str | None = None


class Solver(ABC):
    """Abstract engine adapter. Subclasses set ``name`` and ``capabilities``."""

    name: ClassVar[str]
    capabilities: ClassVar[SolverCapabilities]

    def __init__(
        self,
        component: Component,
        technology: Technology | None,
        spec: SimulationSpec | None = None,
        workdir: str | Path | None = None,
    ):
        # cheap and pure — validated here, exercised by the conformance suite
        self.component = component
        self.technology = technology
        self.spec = spec if spec is not None else SimulationSpec()
        self.workdir = Path(workdir) if workdir is not None else None
        self._artifacts: SetupArtifacts | None = None

    # ---------------- lifecycle ----------------

    @abstractmethod
    def validate(self) -> list[str]:
        """Return human-readable problems with this job; [] means runnable."""

    @abstractmethod
    def build(self) -> SetupArtifacts:
        """Produce the native scene OFFLINE. Deterministic; no network/licenses."""

    @abstractmethod
    def estimate(self) -> ResourceEstimate:
        """Offline resource estimate (may call build())."""

    @abstractmethod
    def run(self) -> SMatrix:
        """Execute the simulation and return the canonical S-matrix."""

    # ---------------- shared helpers ----------------

    def describe(self) -> str:
        """One-paragraph human summary of the configured job."""
        c, s = self.component, self.spec
        ports = ", ".join(p.name for p in c.ports)
        return (
            f"{type(self).__name__}('{self.name}') on component '{c.name}' "
            f"({len(c.ports)} ports: {ports}); "
            f"wavelength {s.wavelength_start}-{s.wavelength_end} um "
            f"({s.wavelength_points} pts), mesh {s.mesh} cells/wvl, "
            f"boundary {list(s.boundary)}, symmetry {list(s.symmetry)}, "
            f"modes {list(s.modes)}."
        )

    def frequencies_hz(self) -> np.ndarray:
        """The simulation frequency grid in Hz (ascending)."""
        c_um_s = 299792458.0 * 1e6
        wavelengths = np.linspace(
            self.spec.wavelength_start, self.spec.wavelength_end, self.spec.wavelength_points
        )
        return np.sort(c_um_s / wavelengths)

    def injection_plan(self) -> list[dict]:
        """Solver-agnostic port injection descriptors, sorted by port index.

        Each entry: name, position [x,y,z], axis ('x'|'y'), direction
        ('forward'|'backward'), size [sx,sy,sz] with 0 on the injection axis.
        """
        plan = []
        for p in sorted(self.component.ports, key=lambda p: p.idx):
            if p.direction in (90, 270):
                axis, direction = "y", ("backward" if p.direction == 90 else "forward")
                size = [self.spec.width_ports, 0.0, self.spec.depth_ports]
            elif p.direction in (0, 180):
                axis, direction = "x", ("forward" if p.direction == 180 else "backward")
                size = [0.0, self.spec.width_ports, self.spec.depth_ports]
            else:
                raise ValueError(
                    f"Port direction {p.direction} not supported (0/90/180/270 only)."
                )
            plan.append(
                {
                    "name": p.name,
                    "position": [p.center[0], p.center[1], p.center[2]],
                    "axis": axis,
                    "direction": direction,
                    "size": size,
                }
            )
        return plan

    def domain(self) -> tuple[list[float], list[float]]:
        """(center, span) of the simulation domain in um."""
        b, s = self.component.bounds, self.spec
        center = [b.x_center, b.y_center, (s.z_max + s.z_min) / 2]
        span = [b.x_span + 2 * s.buffer, b.y_span + 2 * s.buffer, s.z_max - s.z_min]
        return center, span


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Solver]] = {}
_EP_SCANNED = False


def _scan_entry_points() -> None:
    """Merge adapters advertised via the 'gds_fdtd.solvers' entry-point group.

    External packages can ship solvers by declaring
    [project.entry-points."gds_fdtd.solvers"] name = "pkg.mod:Class".
    In-package adapters also declare entry points, but they self-register on
    import first; loading is idempotent. Broken plugins are skipped (their
    load error is reported by available_solvers, not raised at import).
    """
    global _EP_SCANNED
    if _EP_SCANNED:
        return
    _EP_SCANNED = True
    from importlib.metadata import entry_points

    for ep in entry_points(group="gds_fdtd.solvers"):
        if ep.name in _REGISTRY:
            continue
        try:
            cls = ep.load()
            _REGISTRY[ep.name] = cls
        except Exception as e:
            _REGISTRY_ERRORS[ep.name] = f"entry point failed to load: {e}"


_REGISTRY_ERRORS: dict[str, str] = {}


def register_solver(cls: type[Solver]) -> type[Solver]:
    """Class decorator: register an adapter under its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


def available_solvers() -> dict[str, str]:
    """Registered solver names -> availability ('ok' or the import problem).

    Entry-point discovery (external plugins) is wired in WP3.1e; for now this
    reports the in-package registrations.
    """
    _scan_entry_points()
    out = dict(_REGISTRY_ERRORS)
    for name, cls in _REGISTRY.items():
        try:
            probe = getattr(cls, "probe_available", None)
            out[name] = "ok" if probe is None else (probe() or "ok")
        except Exception as e:  # pragma: no cover - defensive
            out[name] = f"unavailable: {e}"
    return out


def get_solver(name: str) -> type[Solver]:
    """Fetch a registered solver class by name (registry + entry points)."""
    _scan_entry_points()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No solver named {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None
