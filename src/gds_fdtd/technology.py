"""
gds_fdtd simulation toolbox.

Technology definition as validated pydantic models.

The YAML format is UNCHANGED from the legacy parser (schema v1; the existing
key names — including per-solver material hints ``tidy3d_db``/``lum_db`` — are
frozen through the 1.x series). Additions in this module are additive-only:

- optional ``schema_version: 1`` key (defaults to 1 when absent);
- optional ``rii:`` material source referencing the refractiveindex.info
  database (see gds_fdtd.materials.rii).

``Technology.to_solver_dict()`` reproduces exactly the dict shape the rest of
the package consumes today; the golden fixtures prove equivalence with the
original parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

    A material may name up to three sources of optical constants, and each
    engine picks one at build time (see :mod:`gds_fdtd.materials.select`):

    - the engine's own database model — ``tidy3d`` / ``lumerical`` in the tech
      file (stored here as ``tidy3d_db`` / ``lum_db``); this is the ``"eda"``
      source. beamz has no vendor database.
    - ``rii`` — a refractiveindex.info page (engine-independent, dispersive).
    - ``nk`` — a single neutral constant index (carried as an extra field).

    ``source`` optionally forces which one to use; otherwise the precedence is
    **eda → rii → nk** (first defined wins). It is an error if the chosen /
    only source does not apply to the engine being run.
    """

    model_config = ConfigDict(extra="allow")  # forward-compatible: unknown hints pass through

    tidy3d_db: dict[str, Any] | None = None
    lum_db: dict[str, Any] | None = None
    rii: RiiRef | None = None
    source: Literal["eda", "rii", "nk"] | None = None

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

    def to_solver_dict(self) -> dict[str, Any]:
        """The solver-facing dict mapping for one material (per-solver hints + neutral nk/rii)."""
        out: dict[str, Any] = {}
        if self.tidy3d_db is not None:
            out["tidy3d_db"] = self.tidy3d_db
        if self.lum_db is not None:
            out["lum_db"] = self.lum_db
        if self.rii is not None:
            out["rii"] = self.rii.model_dump()
        if self.source is not None:
            out["source"] = self.source
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


def _expand_v2_material(entry: object, name: str) -> dict[str, Any]:
    """One v2 named-material entry -> the v1 MaterialSpec mapping.

    v2 keys: ``nk`` (neutral constant), ``rii`` (neutral dispersive),
    ``tidy3d`` (model list, nk number, or full mapping), ``lumerical``
    (database name string). Unknown keys pass through untouched.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"material {name!r} must be a mapping; got {entry!r}")
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "tidy3d":
            if isinstance(value, dict):
                out["tidy3d_db"] = value
            elif isinstance(value, (list, tuple)):
                out["tidy3d_db"] = {"model": list(value)}
            elif isinstance(value, (int, float)):
                out["tidy3d_db"] = {"nk": value}
            else:
                raise ValueError(
                    f"material {name!r}: 'tidy3d' must be a model list, an nk number, "
                    f"or a mapping; got {value!r}"
                )
        elif key == "lumerical":
            if not isinstance(value, str):
                raise ValueError(
                    f"material {name!r}: 'lumerical' must be a material-database "
                    f"name string; got {value!r}"
                )
            out["lum_db"] = {"model": value}
        else:  # nk, rii, and forward-compatible extras carry over verbatim
            out[key] = value
    return out


class Technology(BaseModel):
    """A validated technology (layer stack) definition.

    Two YAML schemas are read:

    - **v1** (default): every layer carries an inline material mapping with
      per-solver hints (``tidy3d_db``/``lum_db``) and/or neutral ``nk``/``rii``.
    - **v2** (``schema_version: 2``): a top-level ``materials:`` section
      defines NAMED materials once; layers reference them by name — no more
      repeating material blocks, one technology for every solver:

      .. code-block:: yaml

         technology:
           name: EBeam
           schema_version: 2
           materials:
             Si:   {nk: 3.476, tidy3d: [cSi, Li1993_293K], lumerical: "Si (Silicon) - Palik"}
             SiO2: {nk: 1.444, lumerical: "SiO2 (Glass) - Palik"}
           substrate:   {z_base: 0.0, z_span: -2, material: SiO2}
           superstrate: {z_base: 0.0, z_span: 3, material: SiO2}
           pinrec: [{layer: [1, 10]}]
           devrec: [{layer: [68, 0]}]
           device:
             - {layer: [1, 0], z_base: 0.0, z_span: 0.22, material: Si, sidewall_angle: 85}

      v2 expands into the v1 model before validation, so the two schemas are
      equivalent by construction (``gds-fdtd convert-tech`` migrates files).
    """

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
    def _known_schema(cls, v: int) -> int:
        if v not in (1, 2):
            raise ValueError(
                f"Unsupported technology schema_version {v}; this release reads v1 and v2"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _expand_v2(cls, data: object) -> object:
        """schema v2 -> v1: resolve named materials into inline mappings."""
        if not (isinstance(data, dict) and data.get("schema_version") == 2):
            return data
        data = dict(data)
        materials = data.pop("materials", None) or {}
        if not isinstance(materials, dict):
            raise ValueError(f"'materials' must be a mapping of names; got {materials!r}")
        expanded = {n: _expand_v2_material(m, n) for n, m in materials.items()}

        def resolve(mat: object, where: str) -> object:
            if isinstance(mat, str):
                if mat not in expanded:
                    raise ValueError(
                        f"{where}: unknown material {mat!r}; defined materials: {sorted(expanded)}"
                    )
                return dict(expanded[mat])
            if isinstance(mat, dict):  # inline v2 material (escape hatch)
                return _expand_v2_material(mat, where)
            return mat

        for key in ("substrate", "superstrate"):
            entry = data.get(key)
            entries = entry if isinstance(entry, list) else [entry]
            for e in entries:
                if isinstance(e, dict) and "material" in e:
                    e["material"] = resolve(e["material"], key)
        for i, d in enumerate(data.get("device") or []):
            if isinstance(d, dict) and "material" in d:
                d["material"] = resolve(d["material"], f"device[{i}]")
        return data

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

    def to_solver_dict(self) -> dict[str, Any]:
        """The schema-v1 dict shape the solver adapters and simprocessor consume."""
        return {
            "name": self.name,
            "substrate": [
                {
                    "z_base": self.substrate.z_base,
                    "z_span": self.substrate.z_span,
                    "material": self.substrate.material.to_solver_dict(),
                }
            ],
            "superstrate": [
                {
                    "z_base": self.superstrate.z_base,
                    "z_span": self.superstrate.z_span,
                    "material": self.superstrate.material.to_solver_dict(),
                }
            ],
            "pinrec": [{"layer": list(layer)} for layer in self.pinrec],
            "devrec": [{"layer": list(layer)} for layer in self.devrec],
            "device": [
                {
                    "layer": list(d.layer),
                    "z_base": d.z_base,
                    "z_span": d.z_span,
                    "material": d.material.to_solver_dict(),
                    "sidewall_angle": d.sidewall_angle,
                }
                for d in self.device
            ],
        }
