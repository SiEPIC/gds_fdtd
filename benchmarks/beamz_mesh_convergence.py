"""Evaluate the BeamZ 0.5.3 S-bend sweep from retained JSON magnitudes.

Run beamz_devices.py at meshes 14/25/30 and beamz_port_probe.py at mesh 30
into results/beamz-0.5.3-convergence first. Reuse validated 0.5.3 meshes 10/20.
This follows convergence.max_delta_db's magnitude metric without fabricating
complex phases from the retained JSON records. No simulation runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compare_beamz_devices import db_at
from magnitude_results import MagnitudeResults

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
OUTPUT = RESULTS / "beamz-0.5.3-convergence"
MESHES = (10, 14, 20, 25, 30)
TOL_DB = 0.05
FLOOR_DB = -10.0


def magnitude_delta(a: MagnitudeResults, b: MagnitudeResults, floor_db: float) -> float:
    """Same amplitude interpolation, overlap, and floor as max_delta_db."""
    assert set(a.port_names) == set(b.port_names)
    lo, hi = max(a.f.min(), b.f.min()), min(a.f.max(), b.f.max())
    assert lo <= hi
    mask = (a.f >= lo) & (a.f <= hi)
    f = a.f[mask]
    floor = 10 ** (floor_db / 20)
    worst = 0.0
    for out in a.port_names:
        for source in a.port_names:
            sa = 10 ** (
                a.magnitude_db(out=a.port_names.index(out) + 1, in_=a.port_names.index(source) + 1)[
                    mask
                ]
                / 20
            )
            sb = np.interp(
                f,
                b.f,
                10
                ** (
                    b.magnitude_db(
                        out=b.port_names.index(out) + 1, in_=b.port_names.index(source) + 1
                    )
                    / 20
                ),
            )
            valid = np.isfinite(sa) & np.isfinite(sb) & (np.maximum(sa, sb) > floor)
            if np.any(valid):
                delta = np.abs(
                    20 * np.log10(np.maximum(sa[valid], floor))
                    - 20 * np.log10(np.maximum(sb[valid], floor))
                )
                worst = max(worst, float(delta.max()))
    return worst


def main():
    matrices = []
    runs = []
    for mesh in MESHES:
        folder = RESULTS / "beamz-0.5.3-validation" if mesh in (10, 20) else OUTPUT
        path = folder / f"sbend-mesh{mesh}/results.json"
        sm = MagnitudeResults(path)
        r = sm.record
        assert r["beamz"] == "0.5.3"
        assert r["mesh"] == mesh
        assert r["finite"]
        assert all(s["all_incident_samples_valid"] for s in r["modal_sources"].values())
        if matrices:
            baseline = matrices[0].record
            assert r["gds_sha256"] == baseline["gds_sha256"]
            assert r["ports"] == baseline["ports"]
            for key in ("diagnostic_monitor_offset_um", "diagnostic_source_offset_um"):
                assert r[key] == baseline[key]
            assert {k: v for k, v in r["spec"].items() if k != "mesh"} == {
                k: v for k, v in baseline["spec"].items() if k != "mesh"
            }
            np.testing.assert_array_equal(sm.wavelength_um, matrices[0].wavelength_um)
        matrices.append(sm)
        runs.append(
            {
                "mesh": mesh,
                "dx_nm": r["setup"]["dx_nm"],
                "grid_cells": int(np.prod(r["setup"]["grid_shape"])),
                "s21_db_at_1550": db_at(sm, 2, 1),
                "s12_db_at_1550": db_at(sm, 1, 2),
                "s11_db_at_1550": db_at(sm, 1, 1),
                "s22_db_at_1550": db_at(sm, 2, 2),
                "reciprocity_max_abs": r["reciprocity_max_abs"],
                "max_power_balance": r["max_power_balance"],
                "all_temporally_converged": all(x["termination"]["converged"] for x in r["runs"]),
            }
        )
    deltas = [
        {
            "from_mesh": MESHES[i],
            "to_mesh": MESHES[i + 1],
            "through_max_delta_db": magnitude_delta(a, b, FLOOR_DB),
            "matrix_max_delta_db_floor_minus40": magnitude_delta(a, b, -40.0),
        }
        for i, (a, b) in enumerate(zip(matrices[:-1], matrices[1:], strict=True))
    ]
    probes = []
    for mesh in (10, 20, 30):
        if mesh == 10:
            path = RESULTS / "beamz-0.5.3-validation/native-probe-mesh10.json"
        else:
            folder = RESULTS / "beamz-0.5.3-validation" if mesh == 20 else OUTPUT
            path = folder / f"sbend-plane-probe-mesh{mesh}/probe.json"
        p = json.loads(path.read_text())
        assert p["beamz"] == "0.5.3" and all(p["valid_mask"])
        values = np.asarray([p["s21_db"][f"out_{i}"] for i in range(5)])
        probes.append(
            {
                "mesh": mesh,
                "wavelength_um": p["wavelength_um"],
                "s21_db_by_plane": values.tolist(),
                "outward_distances_um": (-np.asarray(p["output_inward_offsets_um"])).tolist(),
                "lead_spread_db": np.ptp(values[1:], axis=0).tolist(),
                "temporally_converged": p["termination"]["converged"],
            }
        )
    passed = all(d["through_max_delta_db"] < TOL_DB for d in deltas[-2:])
    probe_passed = max(probes[-1]["lead_spread_db"]) < TOL_DB
    references = {}
    for engine in ("tidy3d", "lumerical"):
        path = (
            ROOT / f"examples/06_convergence_and_caching/recorded/sbend_{engine}_convergence.json"
        )
        references[engine] = json.loads(path.read_text())["mesh"]
    summary = {
        "beamz": "0.5.3",
        "tolerance_db": TOL_DB,
        "floor_db": FLOOR_DB,
        "criterion": (
            "Both final successive through-magnitude deltas below tolerance "
            "across both directions and all three sampled wavelengths"
        ),
        "mesh_criterion_passed": passed,
        "finest_probe_criterion_passed": probe_passed,
        "all_temporally_converged": (
            all(r["all_temporally_converged"] for r in runs)
            and all(p["temporally_converged"] for p in probes)
        ),
        "execution_note": (
            "Mesh-30 device and probe use XLA_PYTHON_CLIENT_ALLOCATOR=platform "
            "to reduce allocator pooling; physical settings are unchanged."
        ),
        "wavelength_um": matrices[0].wavelength_um.tolist(),
        "runs": runs,
        "successive_deltas": deltas,
        "probes": probes,
        "recorded_references": references,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axs[0, 0]
    ax.plot(
        MESHES, [r["s21_db_at_1550"] for r in runs], "o-", label="BeamZ 0.5.3 S21", color="#0072B2"
    )
    ax.plot(
        MESHES, [r["s12_db_at_1550"] for r in runs], "o--", label="BeamZ 0.5.3 S12", color="#56B4E9"
    )
    for engine, color in (("tidy3d", "#D55E00"), ("lumerical", "#009E73")):
        vals = references[engine]
        meshes = sorted(map(int, vals))
        ax.plot(
            meshes,
            [vals[str(m)]["s21_db"] for m in meshes],
            "s--",
            color=color,
            label=f"{engine} S21 (recorded)",
        )
    ax.set(
        title="Transmission at 1.55 µm", xlabel="Mesh setting (engine-specific)", ylabel="|S| (dB)"
    )
    ax = axs[0, 1]
    ax.plot(MESHES, [r["s11_db_at_1550"] for r in runs], "o-", label="BeamZ S11", color="#0072B2")
    ax.plot(MESHES, [r["s22_db_at_1550"] for r in runs], "o--", label="BeamZ S22", color="#56B4E9")
    for engine, color in (("tidy3d", "#D55E00"), ("lumerical", "#009E73")):
        vals = references[engine]
        meshes = sorted(map(int, vals))
        ax.plot(
            meshes,
            [vals[str(m)]["s11_db"] for m in meshes],
            "s--",
            color=color,
            label=f"{engine} S11 (recorded)",
        )
    ax.set(
        title="Reflection at 1.55 µm (separate from through criterion)",
        xlabel="Mesh setting (engine-specific)",
        ylabel="|S| (dB)",
    )
    ax = axs[1, 0]
    ax.semilogy(
        MESHES[1:],
        [d["through_max_delta_db"] for d in deltas],
        "o-",
        color="#0072B2",
        label="Worst through change, three wavelengths",
    )
    ax.axhline(TOL_DB, color="#D55E00", linestyle="--", label="0.05 dB tolerance")
    ax.set(
        title=f"Final two refinements: {'PASS' if passed else 'NOT CONVERGED'}",
        xlabel="Finer mesh of successive pair",
        ylabel="Maximum change (dB)",
    )
    ax = axs[1, 1]
    for p, color in zip(probes, ("0.5", "#56B4E9", "#0072B2"), strict=True):
        ax.plot(
            p["outward_distances_um"],
            np.asarray(p["s21_db_by_plane"])[:, 1],
            "o-",
            color=color,
            label=f"Mesh {p['mesh']}",
        )
    ax.axvline(0, color="0.5", linestyle=":")
    ax.set(
        title="Same-run monitor probe at 1.55 µm",
        xlabel="Distance outward from bend port (µm)",
        ylabel="S21 (dB)",
    )
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Sharp S-bend · BeamZ 0.5.3 · RTX 3090 · fixed setup, refined mesh")
    fig.savefig(OUTPUT / "convergence.png", dpi=180)
    fig.savefig(OUTPUT / "convergence.svg")
    svg = OUTPUT / "convergence.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
