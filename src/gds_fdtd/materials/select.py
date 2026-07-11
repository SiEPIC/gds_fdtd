"""Per-engine material-source selection — how a solver picks a material model.

A technology material may name up to **three** sources of optical constants,
and each engine chooses exactly one at build time:

============  ==============================================  ====================
source        what it is                                      engines
============  ==============================================  ====================
``eda``       the engine's *own* database model               tidy3d, Lumerical
              (``tidy3d`` / ``lumerical`` in the tech file)   (beamz has none)
``rii``       a refractiveindex.info page (``rii:``)           all three
``nk``        a single neutral constant index (``nk:``)        all three
============  ==============================================  ====================

**Selection rule** (per material, per engine):

1. If the material sets ``source:`` explicitly, that source is used — and it is
   an error if that source is not defined for the engine in question.
2. Otherwise the default precedence is **eda → rii → nk**: the first one that is
   defined (and applies to the engine) wins.
3. If *none* applies to the engine, a :class:`~gds_fdtd.errors.MaterialSourceError`
   is raised (e.g. a Lumerical-only material asked to run on tidy3d).

You do **not** need to specify all three — most materials give the engine model
and let dispersion be exact; add ``rii`` for an engine-independent measured
model, or just ``nk`` for a quick constant. ``beamz`` has no vendor database, so
its ``eda`` slot is always empty (it uses ``rii`` or ``nk``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..errors import MaterialSourceError

#: which technology key holds the engine's own ("eda") database model.
#: beamz has no vendor material database, so it is intentionally absent.
_EDA_KEY: dict[str, str] = {"tidy3d": "tidy3d_db", "lumerical": "lum_db"}

#: valid sources, in default-precedence order.
SOURCES: tuple[str, ...] = ("eda", "rii", "nk")


def _defined(material: dict[str, Any], source: str, engine: str) -> bool:
    """Is ``source`` defined on ``material`` and applicable to ``engine``?"""
    if source == "eda":
        key = _EDA_KEY.get(engine)
        return key is not None and material.get(key) is not None
    return material.get(source) is not None


def available_sources(material: dict[str, Any], engine: str) -> list[str]:
    """The sources defined for this material that apply to ``engine``,
    in precedence order (eda, rii, nk)."""
    return [s for s in SOURCES if _defined(material, s, engine)]


def select_source(material: dict[str, Any], engine: str, *, name: str = "material") -> str:
    """Choose the optical-constant source for ``material`` on ``engine``.

    Honors an explicit ``material["source"]`` (validated against what the
    engine can actually use), else applies the eda → rii → nk precedence.

    Args:
        material: the solver-facing material mapping (``to_solver_dict`` form:
            may hold ``tidy3d_db``, ``lum_db``, ``rii``, ``nk``, ``source``).
        engine: ``"tidy3d"``, ``"lumerical"``, or ``"beamz"``.
        name: material name, for error messages.

    Returns:
        one of ``"eda"``, ``"rii"``, ``"nk"``.

    Raises:
        MaterialSourceError: explicit ``source`` unusable, or nothing available.
    """
    available: list[str] = available_sources(material, engine)
    explicit = material.get("source")
    if explicit is not None:
        if explicit not in SOURCES:
            raise MaterialSourceError(f"{name}: source={explicit!r} is not one of {SOURCES}")
        if explicit not in available:
            raise MaterialSourceError(
                f"{name}: source={explicit!r} was requested but is not defined for the "
                f"{engine!r} engine (defined here: {available or 'none'}). "
                f"{_hint(engine)}"
            )
        return str(explicit)
    if available:
        return available[0]
    raise MaterialSourceError(
        f"{name}: no optical-constant source usable by the {engine!r} engine. {_hint(engine)}"
    )


def _hint(engine: str) -> str:
    eda = {
        "tidy3d": "a 'tidy3d' database model",
        "lumerical": "a 'lumerical' database name",
    }.get(engine)
    opts = ([eda] if eda else []) + ["an 'rii' reference", "an 'nk' constant"]
    return "Give the material " + ", ".join(opts[:-1]) + f", or {opts[-1]}."


def iter_materials(tech_dict: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(label, material_dict)`` for every material in a solver-dict
    technology (substrate, superstrate, each device layer)."""
    for slab in tech_dict.get("substrate", []) or []:
        yield "substrate", slab.get("material", {})
    for slab in tech_dict.get("superstrate", []) or []:
        yield "superstrate", slab.get("material", {})
    for i, d in enumerate(tech_dict.get("device", []) or []):
        yield f"device layer {i} (GDS {d.get('layer')})", d.get("material", {})


def check_materials(tech_dict: dict[str, Any], engine: str) -> list[str]:
    """Validate that every material has a usable source for ``engine``.

    Returns a list of human-readable problems ([] means every material is fine)
    — used by each adapter's ``validate()`` so a mismatched technology is
    reported clearly before anything runs.
    """
    problems: list[str] = []
    for label, mat in iter_materials(tech_dict):
        try:
            select_source(mat, engine, name=label)
        except MaterialSourceError as e:
            problems.append(str(e))
    return problems


def source_index(
    material: dict[str, Any], source: str, wavelength_um: float, *, rii_db_dir: Any = None
) -> complex:
    """Constant complex index ``n + ik`` for a chosen ``source`` at one wavelength.

    Handles the two engine-independent, offline-resolvable sources — ``nk`` and
    ``rii`` (sampled at ``wavelength_um``) — plus an ``eda`` hint that carries a
    plain ``nk``. Engines that need a *dispersive* medium build it themselves
    (see ``RiiMaterial.to_tidy3d_medium`` and the Lumerical sampled-material
    path); this is the single-value fallback used by beamz and rasterization.
    """
    if source == "nk" or (source == "eda" and isinstance(material.get("tidy3d_db"), dict)):
        raw = material["nk"] if source == "nk" else material["tidy3d_db"].get("nk")
        if raw is None:
            raise MaterialSourceError("eda source is a model reference, not a constant nk")
        if isinstance(raw, (list, tuple)):
            return complex(raw[0], raw[1] if len(raw) > 1 else 0.0)
        return complex(raw)
    if source == "rii":
        from .rii import load_rii_material

        ref = material["rii"]
        if hasattr(ref, "model_dump"):
            ref = ref.model_dump()
        mat = load_rii_material(ref["shelf"], ref["book"], ref["page"], db_dir=rii_db_dir)
        return complex(mat.nk_at(wavelength_um))
    raise MaterialSourceError(f"cannot resolve a constant index for source {source!r}")
