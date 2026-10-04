"""Isolate output-plane dependence in a single native BeamZ simulation.

The source and input monitor stay fixed; every output probe samples the same
physical waveguide. Probe planes are diagnostic samples, not independent ports.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import beamz
import numpy as np
from beamz.analysis import s_parameters
from beamz.devices.sources.time import gaussian_band_pulse
from beamz_devices import ROOT, make_job

from gds_fdtd import SMatrix
from gds_fdtd.solvers.beamz import BeamzSolver

OFFSETS_UM = (0.05, -0.25, -0.5, -0.75, -1.0)


class ProbeSolver(BeamzSolver):
    def run(self) -> SMatrix:
        nat = self.build().native
        dx, dt, wl = nat["dx"], nat["dt"], nat["wl0"]
        # Include 1.55 um exactly, rather than interpolating three samples.
        frequencies = 299792458 / (np.array([1.6, 1.55, 1.5]) * 1e-6)
        pulse = gaussian_band_pulse(
            frequencies,
            carrier_frequency=299792458 / wl,
            dt=dt,
            run_after_sources_uoc=90,
            max_output_distance_um=3,
        )
        mode = beamz.ModeSpec(polarization="te")
        signal = beamz.SampledSignal(
            values=pulse.signal, quadrature=pulse.signal_quadrature, dt=dt, freq0=299792458 / wl
        )

        def port(original, name, offset):
            base, plane = nat["ports"][original], nat["planes"][original]
            sign = 1 if base["direction"][0] == "+" else -1
            return beamz.Port(
                center=(
                    base["center"][0] + sign * offset * 1e-6,
                    base["center"][1],
                    plane["z_center"],
                ),
                size=(0.0, plane["span"], plane["z_span"]),
                name=name,
                direction=base["direction"][0],
                mode_spec=mode,
            )

        src = port("opt1", "source", -1.4)
        input_port = port("opt1", "input", -0.5)
        probes = [port("opt2", f"out_{i}", offset) for i, offset in enumerate(OFFSETS_UM)]
        source = beamz.ModeSource(
            center=src.center,
            size=src.size,
            direction=src.direction,
            mode_spec=mode,
            source_time=signal,
        )
        sim = beamz.Simulation(
            design=nat["design"],
            sources=[source],
            monitors=[p.to_monitor(frequencies) for p in [input_port, *probes]],
            boundaries=[
                beamz.PML(edges=("left", "right", "top", "bottom", "front", "back"), thickness=1e-6)
            ],
            time=pulse.time,
            resolution=dx,
            setup_device="cpu",
        )
        result = sim.run(
            termination=beamz.AutoTermination(
                min_steps=int(np.ceil((pulse.source_end_time + pulse.tail_time) / dt)),
                field_decay=1e-4,
            )
        )
        extracted = s_parameters(
            result, source_port="input", ports=[input_port, *probes], frequencies=frequencies
        )
        self.probe_record = {
            "beamz": beamz.__version__,
            "mesh": self.spec.mesh,
            "source_offset_um": -1.4,
            "input_monitor_offset_um": -0.5,
            "output_inward_offsets_um": OFFSETS_UM,
            "wavelength_um": (299792458 / frequencies / 1e-6).tolist(),
            "output_centers_m": [p.center for p in probes],
            "s21_db": {
                p.name: (20 * np.log10(np.abs(extracted.s_matrix[(p.name, "input")]))).tolist()
                for p in probes
            },
            "P_in": np.asarray(extracted.diagnostics["P_in"]).tolist(),
            "valid_mask": np.asarray(extracted.diagnostics["valid_mask"]).tolist(),
            "termination": dataclasses.asdict(result.termination),
        }
        self.probe_waves = {}
        for name, values in extracted.diagnostics["waves"].items():
            for key, value in values.items():
                array = np.asarray(value)
                if array.dtype.kind != "O":
                    self.probe_waves[f"{name}/{key}"] = array
        return SMatrix.from_entries(
            [
                ("input", p.name, 1, 1, frequencies, extracted.s_matrix[(p.name, "input")])
                for p in probes
            ],
            name="same_run_output_probes",
            port_names=["input", *[p.name for p in probes]],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    solver, layout, _ = make_job("sbend", args.mesh)
    probe = ProbeSolver(solver.component, solver.technology, solver.spec)
    start = time.perf_counter()
    sm = probe.run()
    probe.probe_record["wall_seconds"] = time.perf_counter() - start
    out = args.output or ROOT / f"benchmarks/results/devices/sbend-plane-probe-mesh{args.mesh}"
    out.mkdir(parents=True, exist_ok=True)
    sm.to_npz(str(out / "smatrix.npz"))
    np.savez_compressed(out / "modal_waves.npz", **probe.probe_waves)
    (out / "probe.json").write_text(json.dumps(probe.probe_record, indent=2) + "\n")
    print(json.dumps(probe.probe_record, indent=2))
    del layout


if __name__ == "__main__":
    main()
