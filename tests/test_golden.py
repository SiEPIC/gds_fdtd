"""
Golden (characterization) tests — WP0.3 of MODERNIZATION_PLAN.md.

These pin the CURRENT geometry-building behavior of load_component_from_tech
(bugs and all) so that the Phase 1/2 refactors can prove geometric equivalence.
Materials are serialized as the raw technology-YAML mapping (never solver-object
reprs), and the lumerical tech file is used so this runs with zero optional
extras installed.

Regenerate (only when a WP card explicitly authorizes it):
    python tests/test_golden.py --regenerate
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

TESTS_DIR = pathlib.Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden"
EXAMPLES_DIR = TESTS_DIR.parent / "examples"

FIXTURES = [
    # (fixture id, gds path, top cell or None, tech yaml path)
    (
        "si_sin_escalator",
        TESTS_DIR / "si_sin_escalator.gds",
        None,
        TESTS_DIR / "tech_lumerical.yaml",
    ),
    (
        "crossing_te1550",
        EXAMPLES_DIR / "devices.gds",
        "crossing_te1550",
        TESTS_DIR / "tech_lumerical.yaml",
    ),
    (
        "directional_coupler_te1550",
        EXAMPLES_DIR / "devices.gds",
        "directional_coupler_te1550",
        TESTS_DIR / "tech_lumerical.yaml",
    ),
]


def _round(x: float) -> float:
    return round(float(x), 9)


def _material_to_dict(material) -> object:
    """Serialize a material as plain data. get_material() returns a dict of
    per-solver entries; lum entries are plain strings. Never repr() solver objects."""
    if isinstance(material, dict):
        return {k: _material_to_dict(v) for k, v in sorted(material.items())}
    if material is None or isinstance(material, (str, int, float, bool)):
        return material
    # Unexpected (e.g. a tidy3d Medium) — record its type name only, never its repr.
    return f"<{type(material).__module__}.{type(material).__name__}>"


def _structure_to_dict(s) -> dict:
    return {
        "name": s.name,
        "polygon": [[_round(x), _round(y)] for x, y in s.polygon],
        "z_base": _round(s.z_base),
        "z_span": _round(s.z_span),
        "sidewall_angle": _round(s.sidewall_angle),
        "layer": list(s.layer),
        "material": _material_to_dict(s.material),
    }


def component_to_dict(comp) -> dict:
    """Serialize a component (current mixed structure/list-of-structure shape) flatly,
    preserving order."""
    flat_structures = []
    for entry in comp.structures:
        if isinstance(entry, list):
            flat_structures.extend(_structure_to_dict(s) for s in entry)
        else:
            flat_structures.append(_structure_to_dict(entry))
    return {
        "name": comp.name,
        "structures": flat_structures,
        "ports": [
            {
                "name": p.name,
                "idx": p.idx,
                "center": [_round(c) if c is not None else None for c in p.center],
                "width": _round(p.width),
                "direction": p.direction,
                "height": _round(p.height) if p.height is not None else None,
            }
            for p in comp.ports
        ],
        "bounds": {
            "vertices": [[_round(x), _round(y)] for x, y in comp.bounds.vertices],
            "z_center": _round(comp.bounds.z_center),
            "z_span": _round(comp.bounds.z_span),
        },
    }


def build_component(gds_path: pathlib.Path, top_cell: str | None, tech_path: pathlib.Path):
    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.simprocessor import load_component_from_tech

    tech = parse_yaml_tech(str(tech_path))
    cell, _layout = load_cell(str(gds_path), top_cell=top_cell)
    return load_component_from_tech(cell=cell, tech=tech)


def _assert_close(actual, expected, path=""):
    """Deep compare with approx for floats."""
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-9), f"mismatch at {path}"
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), (
            f"length mismatch at {path}"
        )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and sorted(actual) == sorted(expected), (
            f"keys mismatch at {path}"
        )
        for k in expected:
            _assert_close(actual[k], expected[k], f"{path}.{k}")
    else:
        assert actual == expected, f"mismatch at {path}: {actual!r} != {expected!r}"


@pytest.mark.parametrize("fixture_id,gds,top_cell,tech", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_golden_geometry(fixture_id, gds, top_cell, tech):
    golden_file = GOLDEN_DIR / f"{fixture_id}.json"
    assert golden_file.exists(), (
        f"golden file {golden_file} missing — generate with: python tests/test_golden.py --regenerate"
    )
    expected = json.loads(golden_file.read_text())
    actual = component_to_dict(build_component(gds, top_cell, tech))
    _assert_close(actual, expected)


def regenerate() -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    for fixture_id, gds, top_cell, tech in FIXTURES:
        data = component_to_dict(build_component(gds, top_cell, tech))
        out = GOLDEN_DIR / f"{fixture_id}.json"
        out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out} ({len(data['structures'])} structures, {len(data['ports'])} ports)")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        print(__doc__)
