"""Fresh-process BeamZ runs of the README's nontrivial photonic devices.

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
  .venv/bin/python benchmarks/beamz_devices.py sbend --mesh 6

References remain recorded data; no cloud or licensed engine is executed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gds_fdtd import SimulationSpec, Technology, get_solver
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.plotting import plot_component
from gds_fdtd.simprocessor import load_component_from_tech

ROOT = Path(__file__).resolve().parents[1]


def make_job(device: str, mesh: int):
    tech = Technology.from_yaml(ROOT / "examples/tech.yaml")
    if device == "ybranch":
        import siepic_ebeam_pdk

        path = Path(siepic_ebeam_pdk.__file__).parent / "gds/EBeam/ebeam_y_1550.gds"
        cell, layout = load_cell(str(path), top_cell="ebeam_y_1550")
        points = 5
    elif device == "sbend":
        path = ROOT / "examples/devices.gds"
        cell, layout = load_cell(str(path), top_cell="sbend_dontfabme")
        points = 3
    else:
        path = ROOT / "examples/10_cookbook/si_sin_escalator.gds"
        cell, layout = load_cell(str(path))
        points = 11
    component = load_component_from_tech(cell=cell, tech=tech)
    component.name = device
    spec = SimulationSpec(
        wavelength_start=1.5,
        wavelength_end=1.6,
        wavelength_points=points,
        mesh=mesh,
        z_min=-1,
        z_max=1.11,
        buffer=0.8 if device == "ybranch" else 1.0,
    )
    # Keep the owning KLayout layout alive while using its cell.
    return get_solver("beamz")(component, tech, spec), layout, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("sbend", "ybranch", "escalator"))
    parser.add_argument("--mesh", type=int, default=6)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--monitor-offset", type=float, help="Diagnostic inward offset in um")
    parser.add_argument("--source-offset", type=float, help="Diagnostic inward offset in um")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / f"benchmarks/results/devices/{args.device}-mesh{args.mesh}"
    out.mkdir(parents=True, exist_ok=True)
    if args.monitor_offset is not None or args.source_offset is not None:
        import gds_fdtd.solvers.beamz as adapter

        if args.monitor_offset is not None:
            adapter._OUTPUT_MONITOR_OFFSET = args.monitor_offset * 1e-6
        if args.source_offset is not None:
            adapter._SOURCE_OFFSET = args.source_offset * 1e-6
    solver, layout, gds = make_job(args.device, args.mesh)
    start = time.perf_counter()
    artifacts = solver.build()
    record = {
        "device": args.device,
        "beamz": importlib.metadata.version("beamz"),
        "jax": importlib.metadata.version("jax"),
        "mesh": args.mesh,
        "diagnostic_monitor_offset_um": args.monitor_offset,
        "diagnostic_source_offset_um": args.source_offset,
        "gds_sha256": hashlib.sha256(gds.read_bytes()).hexdigest(),
        "gds_file": gds.name,
        "spec": solver.spec.model_dump(mode="json"),
        "setup": artifacts.summary,
        "build_seconds": time.perf_counter() - start,
        "ports": [
            {"name": p.name, "center_um": list(p.center), "direction": p.direction}
            for p in solver.component.ports
        ],
    }
    (out / "setup.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if args.build_only:
        return
    import jax
    import matplotlib.pyplot as plt

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("This benchmark requires a local GPU")
    record["devices"] = [str(d) for d in jax.devices()]
    record["gpu_before"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    fig, _ = plot_component(solver.component, spec=solver.spec)
    fig.savefig(out / "geometry.png", dpi=150)
    plt.close(fig)
    # Preserve upstream modal diagnostics for reproducible issue reports.
    from collections.abc import Mapping

    import beamz.analysis

    extract = beamz.analysis.s_parameters

    def capture_modal(*positional, **keywords):
        result = extract(*positional, **keywords)
        arrays = {}

        def flatten(value, key):
            if isinstance(value, Mapping):
                for child, item in value.items():
                    flatten(item, f"{key}/{child}")
            else:
                array = np.asarray(value)
                if array.dtype.kind != "O":
                    arrays[key] = array

        record.setdefault("modal_sources", {})[keywords["source_port"]] = {
            "all_incident_samples_valid": bool(np.all(result.diagnostics["valid_mask"])),
            "min_incident_power": float(np.min(result.diagnostics["P_in"])),
        }
        flatten(result.diagnostics, "diagnostics")
        np.savez_compressed(out / f"modal_{keywords['source_port']}.npz", **arrays)
        return result

    beamz.analysis.s_parameters = capture_modal
    start = time.perf_counter()
    sm = solver.run()
    record["run_seconds"] = time.perf_counter() - start
    record["recorded_at_utc"] = datetime.now(UTC).isoformat()
    sm.to_npz(str(out / "smatrix.npz"))
    record.update(
        {
            "port_names": sm.port_names,
            "finite": bool(np.isfinite(sm.s).all()),
            "max_power_balance": float(np.nanmax(sm.power_balance())),
            "reciprocity_max_abs": float(np.nanmax(np.abs(sm.s - sm.s.swapaxes(1, 2)))),
            "wavelength_um": sm.wavelength_um.tolist(),
            "entries_db": {
                f"{a}<-{b}": sm.magnitude_db(out=a, in_=b).tolist()
                for a in sm.port_names
                for b in sm.port_names
            },
            "runs": [
                {"source": d["source"], "termination": dataclasses.asdict(d["termination"])}
                for d in solver.run_diagnostics
            ],
        }
    )
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(out / "field_z.npz", **solver._field_z, **solver._field_z_meta)
    fig, _ = solver.plot_fields(scale="db")
    fig.savefig(out / "fields.png", dpi=150)
    plt.close(fig)
    print(json.dumps(record, indent=2), flush=True)
    # Failed physical checks stay visible in the report instead of deleting
    # the evidence or treating a low transmission as a software exception.
    if not record["finite"]:
        raise RuntimeError("Non-finite S-matrix; saved for diagnosis")
    del layout


if __name__ == "__main__":
    main()
