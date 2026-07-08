"""Validated technology files + refractiveindex.info materials.

Technology YAMLs are validated pydantic models: bad files fail loudly with the
offending key named. Materials can reference the refractiveindex.info database
(https://github.com/polyanskiy/refractiveindex.info-database) with a neutral
`rii:` entry that any solver adapter can resolve — no per-solver material
config needed.
"""

import os

from gds_fdtd.technology import RiiRef, Technology

if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(__file__))
    tech = Technology.from_yaml(os.path.join(here, "tech_tidy3d.yaml"))
    print(f"technology '{tech.name}': {len(tech.device)} device layers, schema v{tech.schema_version}")
    for d in tech.device:
        print(f"  layer {d.layer}: z={d.z_base}..{d.z_base + d.z_span} um, sidewall {d.sidewall_angle} deg")

    # a material referencing refractiveindex.info (shelf/book/page):
    #   material:
    #     rii: {shelf: main, book: Si, page: Li-293}
    si = RiiRef(shelf="main", book="Si", page="Li-293")

    # resolution is OFFLINE: point GDS_FDTD_RII_DB (or db_dir=) at a local copy
    # of the database's data/ folder. The repo ships a tiny Si test fixture:
    db = os.path.join(here, "..", "tests", "rii_db")
    if os.path.isdir(db):
        mat = si.load(db_dir=db)
        print(f"n_Si(1.55 um) = {mat.n_at(1.55):.4f}  (tabulated {mat.wavelength_range_um} um)")
    else:
        print("set GDS_FDTD_RII_DB to a refractiveindex.info database copy to resolve rii materials")

    # invalid files fail with the offending key named:
    try:
        Technology.model_validate({"name": "broken", "device": []})
    except Exception as e:
        print(f"validation catches bad files: {type(e).__name__}")
