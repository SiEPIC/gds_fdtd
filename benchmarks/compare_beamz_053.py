"""Compare the preserved BeamZ 0.5.2 runs with the 0.5.3 validation runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from compare_beamz_devices import db_at, references

from gds_fdtd import SMatrix

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/results"
NEW = BASE / "beamz-0.5.3-validation"


def main():
    summary = {"probes": [], "devices": []}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for mesh, ax in zip((10, 20), axes, strict=True):
        old_path = BASE / f"devices/sbend-plane-probe-mesh{mesh}/probe.json"
        new_path = (
            NEW / "native-probe-mesh10.json"
            if mesh == 10
            else NEW / "sbend-plane-probe-mesh20/probe.json"
        )
        for version, path, color in (
            ("0.5.2", old_path, "#777777"),
            ("0.5.3", new_path, "#0072B2"),
        ):
            data = json.loads(path.read_text())
            values = np.array([data["s21_db"][f"out_{i}"] for i in range(5)])
            distances = -np.array(data["output_inward_offsets_um"])
            summary["probes"].append(
                {
                    "beamz": version,
                    "mesh": mesh,
                    "s21_db_at_1550": values[:, 1].tolist(),
                    "lead_spread_db_at_1550": float(np.ptp(values[1:, 1])),
                    "lead_spread_db_by_wavelength": np.ptp(values[1:], axis=0).tolist(),
                    "termination": data["termination"],
                }
            )
            ax.plot(distances, values[:, 1], "o-", label=f"BeamZ {version}", color=color)
        ax.axvline(0, color="0.6", linestyle=":")
        ax.set(
            title=f"Sharp S-bend, mesh {mesh}",
            xlabel="Distance outward from output port (µm)",
            ylabel="S21 at 1.55 µm (dB)",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Same-run monitor-position probe: before and after the material-snapshot fix")
    fig.savefig(NEW / "monitor_comparison.png", dpi=170)
    fig.savefig(NEW / "monitor_comparison.svg")
    plt.close(fig)
    for path in sorted(NEW.glob("*/results.json")):
        data = json.loads(path.read_text())
        sm = SMatrix.from_npz(str(path.parent / "smatrix.npz"))
        old = SMatrix.from_npz(str(BASE / "devices" / path.parent.name / "smatrix.npz"))
        row = {
            "device": data["device"],
            "mesh": data["mesh"],
            "finite": data["finite"],
            "all_converged": all(r["termination"]["converged"] for r in data["runs"]),
            "reciprocity_max_abs": data["reciprocity_max_abs"],
            "max_power_balance": data["max_power_balance"],
            "matrix_db_at_1550": {
                version: [
                    [db_at(matrix, i + 1, j + 1) for j in range(matrix.n_ports)]
                    for i in range(matrix.n_ports)
                ]
                for version, matrix in {"0.5.2": old, "0.5.3": sm}.items()
            },
        }
        if data["device"] != "sbend":
            row["reference_matrix_db_at_1550"] = {
                engine: [
                    [db_at(matrix, i + 1, j + 1) for j in range(matrix.n_ports)]
                    for i in range(matrix.n_ports)
                ]
                for engine, matrix in references(data["device"]).items()
            }
        row["incident_valid"] = {}
        for modal in path.parent.glob("modal_*.npz"):
            with np.load(modal, allow_pickle=False) as waves:
                row["incident_valid"][modal.stem] = bool(np.all(waves["diagnostics/valid_mask"]))
        summary["devices"].append(row)
    (NEW / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
