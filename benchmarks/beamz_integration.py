"""Run the local BeamZ integration benchmark in a fresh interpreter.

Example (CUDA-enabled JAX must be installed):
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
    python benchmarks/beamz_integration.py --mesh 10 --output benchmarks/results/latest

Runs both directions of the same 5 um straight used by the recorded three-engine
comparison. Timings include setup, compilation, FDTD and modal extraction; they
are application timings, not kernel throughput measurements.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import gdsfactory as gf
import jax
import numpy as np

from gds_fdtd import SimulationSpec, Technology, get_solver
from gds_fdtd.layout.gdsfactory import from_gdsfactory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=10)
    parser.add_argument("--length", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    gf.gpdk.PDK.activate()
    tech = Technology.from_yaml(str(root / "examples/tech.yaml"))
    comp = from_gdsfactory(gf.components.straight(length=args.length), tech)
    spec = SimulationSpec(wavelength_points=11, mesh=args.mesh, z_min=-1, z_max=1.22)
    solver = get_solver("beamz")(comp, tech, spec)
    devices = [str(device) for device in jax.devices()]
    if os.environ.get("JAX_PLATFORMS") == "cuda" and not all(
        d.platform == "gpu" for d in jax.devices()
    ):
        raise RuntimeError("CUDA requested but no GPU selected")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    start = time.perf_counter()
    artifacts = solver.build()
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sm = solver.run()
    run_seconds = time.perf_counter() - start
    sm.to_npz(str(args.output / "smatrix.npz"))
    fig, _ = solver.plot_fields()
    fig.savefig(args.output / "fields.png", dpi=150)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for output, source in [(2, 1), (1, 2), (1, 1), (2, 2)]:
        ax.plot(
            sm.wavelength_um, sm.magnitude_db(out=output, in_=source), label=f"S{output}{source}"
        )
    ax.set(
        xlabel="Wavelength (um)",
        ylabel="Magnitude (dB)",
        title=f"BeamZ {importlib.metadata.version('beamz')} / mesh {args.mesh}",
    )
    ax.legend()
    fig.savefig(args.output / "sparameters.png", dpi=150)
    result = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "beamz": importlib.metadata.version("beamz"),
        "jax": importlib.metadata.version("jax"),
        "devices": devices,
        "gpu_before": gpu,
        "mesh": args.mesh,
        "length_um": args.length,
        "spec": spec.model_dump(mode="json"),
        "setup": artifacts.summary,
        "build_seconds": build_seconds,
        "run_seconds": run_seconds,
        "wavelength_um": sm.wavelength_um.tolist(),
        "s21_db": sm.magnitude_db(out=2, in_=1).tolist(),
        "s12_db": sm.magnitude_db(out=1, in_=2).tolist(),
        "s11_db": sm.magnitude_db(out=1, in_=1).tolist(),
        "s22_db": sm.magnitude_db(out=2, in_=2).tolist(),
        "max_power_balance": float(np.max(sm.power_balance())),
        "reciprocity_max_abs": float(np.max(np.abs(sm.s - sm.s.swapaxes(1, 2)))),
        "runs": [
            {"source": d["source"], "termination": dataclasses.asdict(d["termination"])}
            for d in solver.run_diagnostics
        ],
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not np.isfinite(sm.s).all():
        raise RuntimeError("Benchmark returned non-finite S-parameters")
    if not all(d["termination"].converged for d in solver.run_diagnostics):
        raise RuntimeError("Benchmark reached its time limit without convergence")
    if max(result["s11_db"] + result["s22_db"]) >= -15:
        raise RuntimeError("Straight-waveguide reflection exceeds -15 dB")
    if min(result["s21_db"] + result["s12_db"]) <= -0.5 or result["max_power_balance"] >= 1.15:
        raise RuntimeError("Straight-waveguide transmission/power regression")
    if result["reciprocity_max_abs"] >= 0.05:
        raise RuntimeError("Straight-waveguide reciprocity regression")


if __name__ == "__main__":
    main()
