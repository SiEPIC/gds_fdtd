"""
gds_fdtd simulation toolbox.

Technology definition as validated pydantic models (WP2.2).

The YAML format is UNCHANGED from the legacy parser (schema v1; the existing
key names — including per-solver material hints ``tidy3d_db``/``lum_db`` — are
frozen through the 1.x series). Additions in this module are additive-only:

- optional ``schema_version: 1`` key (defaults to 1 when absent);
- optional ``rii:`` material source referencing the refractiveindex.info
  database (see gds_fdtd.materials.rii).

``Technology.to_legacy_dict()`` reproduces exactly the dict shape the rest of
the package consumes today; ``core.parse_yaml_tech`` routes through it, so the
golden fixtures prove equivalence with the legacy parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .materials.rii import RiiMaterial


class RiiRef(BaseModel):
    """A refractiveindex.info database page reference (shelf/book/page)."""

    model_config = ConfigDict(extra="forbid")

    shelf: str
    book: str
    page: str

    def load(self, db_dir: str | Path | None = None) -> RiiMaterial:
        """Resolve this reference to tabulated data. See materials.rii."""
        from .materials.rii import load_rii_material

        return load_rii_material(self.shelf, self.book, self.page, db_dir=db_dir)


class MaterialSpec(BaseModel):
    """Material description for one layer.

    Solver-specific hints (``tidy3d_db``, ``lum_db``) are carried verbatim —
    resolving them into solver objects is the solver adapter's job. ``rii``
    is the solver-neutral refractiveindex.info source (shelf/book/page).
    """

    model_config = ConfigDict(extra="allow")  # forward-compatible: unknown hints pass through

    tidy3d_db: dict[str, Any] | None = None
    lum_db: dict[str, Any] | None = None
    rii: RiiRef | None = None

    @field_validator("lum_db")
    @classmethod
    def _lum_db_needs_model(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None and "model" not in v:
            raise ValueError("'lum_db' must be a mapping containing 'model'")
        return v

    @field_validator("tidy3d_db")
    @classmethod
    def _tidy3d_db_needs_nk_or_model(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None and not ("nk" in v or "model" in v):
            raise ValueError("'tidy3d_db' must be a mapping containing 'nk' or 'model'")
        return v

    def to_legacy(self) -> dict[str, Any]:
        """The raw mapping shape the legacy dict flow carries for materials."""
        out: dict[str, Any] = {}
        if self.tidy3d_db is not None:
            out["tidy3d_db"] = self.tidy3d_db
        if self.lum_db is not None:
            out["lum_db"] = self.lum_db
        if self.rii is not None:
            out["rii"] = self.rii.model_dump()
        if self.model_extra:
            out.update(self.model_extra)
        return out


class BackgroundLayer(BaseModel):
    """Substrate/superstrate: a z-slab with a material, no GDS layer."""

    model_config = ConfigDict(extra="forbid")

    z_base: float
    z_span: float
    material: MaterialSpec

    @field_validator("z_span")
    @classmethod
    def _nonzero_span(cls, v: float) -> float:
        if v == 0:
            raise ValueError("z_span must be nonzero")
        return v


class DeviceLayer(BaseModel):
    """A patterned device layer: GDS layer + z-extent + material + sidewall."""

    model_config = ConfigDict(extra="forbid")

    layer: tuple[int, int]
    z_base: float
    z_span: float
    material: MaterialSpec
    sidewall_angle: float = 90.0

    @field_validator("layer", mode="before")
    @classmethod
    def _layer_pair(cls, v: object) -> tuple[int, int]:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"'layer' must be [layer_number, datatype]; got {v!r}")
        return tuple(v)

    @field_validator("z_span")
    @classmethod
    def _nonzero_span(cls, v: float) -> float:
        if v == 0:
            raise ValueError("z_span must be nonzero")
        return v


class Technology(BaseModel):
    """A validated technology (layer stack) definition. YAML schema v1."""

    model_config = ConfigDict(extra="forbid")

    name: str = "Unknown"
    schema_version: int = 1
    substrate: BackgroundLayer
    superstrate: BackgroundLayer
    pinrec: list[tuple[int, int]] = Field(min_length=1)
    devrec: list[tuple[int, int]] = Field(min_length=1)
    device: list[DeviceLayer] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def _v1_only(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported technology schema_version {v}; this release reads v1")
        return v

    @field_validator("pinrec", "devrec", mode="before")
    @classmethod
    def _layer_list(cls, v: list[object]) -> list[tuple[int, int]]:
        # YAML shape: [{layer: [a, b]}, ...] (legacy) or [[a, b], ...]
        out: list[tuple[int, int]] = []
        for item in v:
            if isinstance(item, dict):
                item = item.get("layer")
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"expected 'layer: [number, datatype]'; got {item!r}")
            out.append((int(item[0]), int(item[1])))
        return out

    @model_validator(mode="before")
    @classmethod
    def _unwrap_background_lists(cls, data: object) -> object:
        # legacy YAML holds substrate/superstrate as a single mapping; the
        # legacy dict flow wraps them into one-element lists — accept both
        if isinstance(data, dict):
            for key in ("substrate", "superstrate"):
                v = data.get(key)
                if isinstance(v, list):
                    if len(v) != 1:
                        raise ValueError(f"'{key}' must contain exactly one entry; got {len(v)}")
                    data[key] = v[0]
        return data

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> Technology:
        """Load and validate a technology YAML file (schema v1)."""
        with open(file_path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "technology" not in data:
            raise ValueError(f"{file_path}: expected a top-level 'technology' mapping")
        try:
            return cls.model_validate(data["technology"])
        except Exception as e:
            raise ValueError(f"Invalid technology file {file_path}: {e}") from e

    def to_legacy_dict(self) -> dict[str, Any]:
        """Exactly the dict shape core.technology.to_dict() produced (schema v1)."""
        return {
            "name": self.name,
            "substrate": [
                {
                    "z_base": self.substrate.z_base,
                    "z_span": self.substrate.z_span,
                    "material": self.substrate.material.to_legacy(),
                }
            ],
            "superstrate": [
                {
                    "z_base": self.superstrate.z_base,
                    "z_span": self.superstrate.z_span,
                    "material": self.superstrate.material.to_legacy(),
                }
            ],
            "pinrec": [{"layer": list(layer)} for layer in self.pinrec],
            "devrec": [{"layer": list(layer)} for layer in self.devrec],
            "device": [
                {
                    "layer": list(d.layer),
                    "z_base": d.z_base,
                    "z_span": d.z_span,
                    "material": d.material.to_legacy(),
                    "sidewall_angle": d.sidewall_angle,
                }
                for d in self.device
            ],
        }
