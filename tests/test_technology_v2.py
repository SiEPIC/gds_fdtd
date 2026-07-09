"""WP6.4: technology schema v2 — named materials, equivalence-by-construction."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import yaml

from gds_fdtd.technology import Technology

TESTS_DIR = pathlib.Path(__file__).parent

V2_DOC = {
    "technology": {
        "name": "EBeam",
        "schema_version": 2,
        "materials": {
            "Si": {
                "nk": 3.476,
                "tidy3d": ["cSi", "Li1993_293K"],
                "lumerical": "Si (Silicon) - Palik",
            },
            "SiN": {
                "nk": 1.997,
                "tidy3d": ["Si3N4", "Luke2015PMLStable"],
                "lumerical": "Si3N4 (Silicon Nitride) - Luke",
            },
            "SiO2": {"nk": 1.444, "tidy3d": 1.444, "lumerical": "SiO2 (Glass) - Palik"},
        },
        "substrate": {"z_base": 0.0, "z_span": -2, "material": "SiO2"},
        "superstrate": {"z_base": 0.0, "z_span": 3, "material": "SiO2"},
        "pinrec": [{"layer": [1, 10]}],
        "devrec": [{"layer": [68, 0]}],
        "device": [
            {
                "layer": [1, 0],
                "z_base": 0.0,
                "z_span": 0.22,
                "material": "Si",
                "sidewall_angle": 85,
            },
            {
                "layer": [4, 0],
                "z_base": 0.3,
                "z_span": 0.4,
                "material": "SiN",
                "sidewall_angle": 83,
            },
        ],
    }
}


@pytest.fixture()
def v2_file(tmp_path):
    f = tmp_path / "tech_v2.yaml"
    f.write_text(yaml.safe_dump(V2_DOC, sort_keys=False))
    return f


def test_v2_equals_v1_unified(v2_file):
    """The v2 doc above is the named-materials form of tech_unified.yaml —
    both must produce the IDENTICAL legacy dict (equivalence by construction)."""
    v1 = Technology.from_yaml(TESTS_DIR / "tech_unified.yaml").to_legacy_dict()
    v2 = Technology.from_yaml(v2_file).to_legacy_dict()
    assert v2 == v1


def test_v2_material_forms(v2_file):
    tech = Technology.from_yaml(v2_file)
    si = tech.device[0].material
    assert si.tidy3d_db == {"model": ["cSi", "Li1993_293K"]}
    assert si.lum_db == {"model": "Si (Silicon) - Palik"}
    assert si.model_extra["nk"] == 3.476
    assert tech.substrate.material.tidy3d_db == {"nk": 1.444}  # number form


def test_v2_unknown_material_named_in_error(v2_file):
    doc = yaml.safe_load(v2_file.read_text())
    doc["technology"]["device"][0]["material"] = "Unobtainium"
    v2_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="Unobtainium"):
        Technology.from_yaml(v2_file)


def test_v2_inline_material_escape_hatch(v2_file):
    doc = yaml.safe_load(v2_file.read_text())
    doc["technology"]["device"][0]["material"] = {"nk": 2.0}
    v2_file.write_text(yaml.safe_dump(doc))
    tech = Technology.from_yaml(v2_file)
    assert tech.device[0].material.model_extra["nk"] == 2.0


def test_v3_rejected(v2_file):
    doc = yaml.safe_load(v2_file.read_text())
    doc["technology"]["schema_version"] = 3
    v2_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="schema_version 3"):
        Technology.from_yaml(v2_file)


def test_convert_tech_cli_roundtrip(tmp_path):
    """gds-fdtd convert-tech migrates v1 -> v2, dedupes materials, round-trips."""
    out = tmp_path / "unified_v2.yaml"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "gds_fdtd.cli",
            "convert-tech",
            str(TESTS_DIR / "tech_unified.yaml"),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    doc = yaml.safe_load(out.read_text())["technology"]
    assert doc["schema_version"] == 2
    # SiO2 appears twice in v1 (substrate + superstrate) -> ONE named material
    assert len(doc["materials"]) == 3
    v1 = Technology.from_yaml(TESTS_DIR / "tech_unified.yaml").to_legacy_dict()
    assert Technology.from_yaml(out).to_legacy_dict() == v1
