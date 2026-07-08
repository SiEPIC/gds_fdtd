"""
gds_fdtd simulation toolbox.

SimulationSpec (WP3.1a): one validated pydantic model for every numeric
simulation setting that was previously ~15 loose solver kwargs. Defaults are
IDENTICAL to the historical fdtd_solver defaults. All lengths are um, angles
degrees, frequencies Hz (package-wide convention).

The legacy fdtd_solver consumes this via a thin adapter (its keyword surface
is unchanged); the Phase-3 Solver ABC takes a SimulationSpec directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BoundaryKind = Literal["PML", "Metal", "Periodic"]


class SimulationSpec(BaseModel):
    """Solver-agnostic FDTD simulation settings (validated)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    wavelength_start: float = Field(1.5, gt=0, description="start wavelength [um]")
    wavelength_end: float = Field(1.6, gt=0, description="end wavelength [um]")
    wavelength_points: int = Field(100, ge=2)
    mesh: int = Field(10, gt=0, description="grid cells per wavelength")
    boundary: tuple[BoundaryKind, BoundaryKind, BoundaryKind] = ("PML", "PML", "PML")
    symmetry: tuple[int, int, int] = (0, 0, 0)
    z_min: float = -1.0
    z_max: float = 1.0
    width_ports: float = Field(2.0, gt=0, description="port width [um]")
    depth_ports: float = Field(1.5, gt=0, description="port depth [um]")
    buffer: float = Field(1.0, gt=0, description="domain buffer beyond ports [um]")
    modes: tuple[int, ...] = (1,)
    mode_freq_pts: int = Field(3, ge=1)
    run_time_factor: float = Field(3.0, gt=0)
    field_monitors: tuple[Literal["x", "y", "z"], ...] = ("z",)

    @field_validator("boundary", mode="before")
    @classmethod
    def _boundary_case(cls, v):
        # accept any case, normalize to the canonical spelling
        canon = {"pml": "PML", "metal": "Metal", "periodic": "Periodic"}
        if isinstance(v, (list, tuple)):
            out = []
            for b in v:
                key = str(b).lower()
                if key not in canon:
                    raise ValueError(
                        f"Unsupported boundary {b!r}; supported: {sorted(canon.values())}"
                    )
                out.append(canon[key])
            return tuple(out)
        return v

    @field_validator("symmetry")
    @classmethod
    def _symmetry_values(cls, v):
        if any(s not in (-1, 0, 1) for s in v):
            raise ValueError(f"symmetry values must be -1, 0 or 1; got {v}")
        return v

    @field_validator("modes")
    @classmethod
    def _modes_positive(cls, v):
        if len(v) == 0 or any(int(m) <= 0 for m in v):
            raise ValueError(f"modes must be a non-empty list of positive 1-based ids; got {v}")
        return tuple(int(m) for m in v)

    @model_validator(mode="after")
    def _cross_checks(self):
        if self.wavelength_start >= self.wavelength_end:
            raise ValueError(
                f"wavelength_start ({self.wavelength_start}) must be < "
                f"wavelength_end ({self.wavelength_end})"
            )
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be < z_max ({self.z_max})")
        return self

    # ---------------- convenience ----------------

    @property
    def wavelength_center_um(self) -> float:
        return (self.wavelength_start + self.wavelength_end) / 2

    @property
    def z_span(self) -> float:
        return self.z_max - self.z_min
