"""Regenerate the version comparison from saved integration results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from magnitude_results import MagnitudeResults

from gds_fdtd import SMatrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    rows = []
    for version in ("0.5.0", "0.5.1", "0.5.2"):
        folder = RESULTS / f"beamz-{version}-mesh10"
        result = json.loads((folder / "results.json").read_text())
        sm = MagnitudeResults(folder / "results.json")
        axes[0].plot(sm.wavelength_um, sm.magnitude_db(out=2, in_=1), label=version)
        axes[1].plot(sm.wavelength_um, sm.magnitude_db(out=1, in_=1), label=version)
        rows.append(result)
    for engine, label in (
        ("beamz", "BeamZ 0.4.3 (recorded)"),
        ("tidy3d", "Tidy3D (recorded)"),
        ("lumerical", "Lumerical (recorded)"),
    ):
        old = SMatrix.from_npz(str(ROOT / f"tests/recorded/straight_mesh10_{engine}.npz"))
        axes[0].plot(
            old.wavelength_um, old.magnitude_db(out=2, in_=1), "--", alpha=0.65, label=label
        )
    axes[0].set(title="Forward transmission", xlabel="Wavelength (um)", ylabel="S21 (dB)")
    axes[1].set(title="Input reflection", xlabel="Wavelength (um)", ylabel="S11 (dB)")
    for ax in axes[:2]:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    times = [r["run_seconds"] for r in rows]
    bars = axes[2].bar([r["beamz"] for r in rows], times, color=["C0", "C1", "C2"])
    axes[2].bar_label(bars, fmt="%.1f s")
    axes[2].set(
        title="Both input ports, including compilation",
        ylabel="Solver.run wall time (s)",
        ylim=(0, max(times) * 1.2),
    )
    fig.suptitle("GDS_FDTD / BeamZ · RTX 3090 · 5 um straight · mesh 10 · 11 wavelengths")
    fig.savefig(RESULTS / "comparison.png", dpi=180)
    fig.savefig(RESULTS / "comparison.svg")
    latest = MagnitudeResults(RESULTS / "beamz-0.5.2-mesh10/results.json")
    comparison = {}
    for engine in ("beamz", "tidy3d", "lumerical"):
        old = SMatrix.from_npz(str(ROOT / f"tests/recorded/straight_mesh10_{engine}.npz"))
        # Historical engines sampled different grids (uniform wavelength vs
        # uniform frequency). Compare magnitudes at matched frequencies.
        reference = np.interp(latest.f, old.f, old.magnitude_db(out=2, in_=1))
        comparison[engine] = float(np.max(np.abs(latest.magnitude_db(out=2, in_=1) - reference)))
    (RESULTS / "historical_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print("version | build s | run s | S21 min/max dB | max S11 dB | max power sum")
    for r in rows:
        print(
            f"{r['beamz']} | {r['build_seconds']:.2f} | {r['run_seconds']:.2f} | "
            f"{min(r['s21_db']):.5f}/{max(r['s21_db']):.5f} | "
            f"{max(r['s11_db']):.2f} | {r['max_power_balance']:.6f}"
        )
    print("Maximum S21 difference from historical results (dB):", comparison)


if __name__ == "__main__":
    main()
