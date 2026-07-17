"""Tests for gds_fdtd.technology (WP2.2): pydantic models + rii materials."""

from __future__ import annotations

import pathlib

import pytest

from gds_fdtd.technology import Technology

TESTS_DIR = pathlib.Path(__file__).parent
TECH_FILES = [
    TESTS_DIR / "tech_lumerical.yaml",
    TESTS_DIR / "tech_tidy3d.yaml",
    TESTS_DIR / "tech_unified.yaml",
    TESTS_DIR.parent / "examples" / "tech.yaml",  # ONE tech for every engine
]


@pytest.mark.parametrize("path", TECH_FILES, ids=lambda p: str(p.name) + "/" + p.parent.name)
def test_all_shipped_tech_files_load(path):
    tech = Technology.from_yaml(path)
    assert tech.schema_version in (1, 2)  # examples/tech.yaml is the v2 reference
    assert tech.device and tech.pinrec and tech.devrec
    legacy = tech.to_solver_dict()
    assert set(legacy) == {"name", "substrate", "superstrate", "pinrec", "devrec", "device"}


def test_legacy_dict_matches_old_parser_shape(tmp_path):
    """Technology.to_solver_dict() reproduces the old parser's shape exactly."""
    d = Technology.from_yaml(str(TESTS_DIR / "tech_tidy3d.yaml")).to_solver_dict()
    assert d["device"][0]["layer"] == [1, 0]
    assert d["device"][0]["sidewall_angle"] == 85
    assert isinstance(d["substrate"], list) and len(d["substrate"]) == 1
    assert d["pinrec"][0] == {"layer": [1, 10]}


def _tech_yaml(tmp_path, body: str) -> str:
    f = tmp_path / "tech.yaml"
    f.write_text(body)
    return str(f)


BASE = """technology:
  name: t
  substrate: {{z_base: 0.0, z_span: -2, material: {{lum_db: {{model: SiO2}}}}}}
  superstrate: {{z_base: 0.0, z_span: 3, material: {{lum_db: {{model: SiO2}}}}}}
  pinrec: [{{layer: [1, 10]}}]
  devrec: [{{layer: [68, 0]}}]
  device:
    - {device}
"""


def test_missing_z_span_names_field(tmp_path):
    path = _tech_yaml(
        tmp_path,
        BASE.format(device="{layer: [1, 0], z_base: 0.0, material: {lum_db: {model: Si}}}"),
    )
    with pytest.raises(ValueError, match="z_span"):
        Technology.from_yaml(path)


def test_bad_layer_length_names_field(tmp_path):
    path = _tech_yaml(
        tmp_path,
        BASE.format(
            device="{layer: [1, 0, 7], z_base: 0.0, z_span: 0.22, material: {lum_db: {model: Si}}}"
        ),
    )
    with pytest.raises(ValueError, match="layer"):
        Technology.from_yaml(path)


def test_lum_db_without_model_names_key(tmp_path):
    path = _tech_yaml(
        tmp_path,
        BASE.format(
            device="{layer: [1, 0], z_base: 0.0, z_span: 0.22, material: {lum_db: {oops: Si}}}"
        ),
    )
    with pytest.raises(ValueError, match="lum_db"):
        Technology.from_yaml(path)


def test_future_schema_version_rejected(tmp_path):
    body = BASE.format(
        device="{layer: [1, 0], z_base: 0.0, z_span: 0.22, material: {lum_db: {model: Si}}}"
    ).replace("name: t", "name: t\n  schema_version: 3")
    with pytest.raises(ValueError, match="schema_version"):
        Technology.from_yaml(_tech_yaml(tmp_path, body))


# ---------- rii materials (owner request; offline via committed fixture) ----------


def test_rii_material_loads_offline_and_matches_silicon(tmp_path):
    body = BASE.format(
        device="{layer: [1, 0], z_base: 0.0, z_span: 0.22, "
        "material: {rii: {shelf: main, book: Si, page: Li-293}}}"
    )
    tech = Technology.from_yaml(_tech_yaml(tmp_path, body))
    rii_ref = tech.device[0].material.rii
    assert rii_ref is not None
    mat = rii_ref.load(db_dir=TESTS_DIR / "rii_db")
    assert mat.n_at(1.55) == pytest.approx(3.48, abs=0.05)  # WP2.2 acceptance
    assert mat.k_at(1.55) == 0.0
    assert mat.nk_at(1.55) == pytest.approx(3.4757 + 0j, abs=0.05)
    # legacy dict carries the rii mapping through
    legacy = tech.to_solver_dict()
    assert legacy["device"][0]["material"]["rii"] == {
        "shelf": "main",
        "book": "Si",
        "page": "Li-293",
    }


def test_rii_out_of_range_raises():
    from gds_fdtd.materials import load_rii_material

    mat = load_rii_material("main", "Si", "Li-293", db_dir=TESTS_DIR / "rii_db")
    with pytest.raises(ValueError, match="outside the tabulated range"):
        mat.n_at(10.0)


def test_rii_missing_page_gives_actionable_error():
    from gds_fdtd.errors import TechnologyError
    from gds_fdtd.materials import load_rii_material

    with pytest.raises(TechnologyError, match="GDS_FDTD_RII_DB"):
        load_rii_material("main", "Unobtainium", "X-0", db_dir=TESTS_DIR / "rii_db")


def test_rii_sellmeier_formula(tmp_path):
    """formula 1 (Sellmeier) support — fused silica coefficients, n(1.55)≈1.444."""
    page = tmp_path / "main" / "SiO2" / "Malitson.yml"
    page.parent.mkdir(parents=True)
    page.write_text(
        "DATA:\n"
        "  - type: formula 1\n"
        "    wavelength_range: 0.21 6.7\n"
        "    coefficients: 0 0.6961663 0.0684043 0.4079426 0.1162414 0.8974794 9.896161\n"
    )
    from gds_fdtd.materials import load_rii_material

    mat = load_rii_material("main", "SiO2", "Malitson", db_dir=tmp_path)
    assert mat.n_at(1.55) == pytest.approx(1.444, abs=0.005)
