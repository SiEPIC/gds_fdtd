"""Standalone BeamZ reproduction: no GDS_FDTD, PDK, or KLayout dependency.

pip install 'beamz==0.5.2' 'jax[cuda12]'
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python \
  beamz_plane_dependence.py --mesh 10 --output probe10.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path

import beamz as bz
import numpy as np
from beamz.analysis import s_parameters
from beamz.devices.sources.time import gaussian_band_pulse


class Probe:
    def __init__(self, scene, mesh):
        self.scene, self.mesh = scene, mesh

    def run(self):
        scene = self.scene
        design = bz.Design(
            width=scene["size_m"][0],
            height=scene["size_m"][1],
            depth=scene["size_m"][2],
            material=bz.Material(permittivity=scene["background_epsilon"]),
            structures=[
                bz.Polygon(
                    vertices=s["vertices_m"],
                    z=s["z_m"],
                    depth=s["depth_m"],
                    material=bz.Material(permittivity=s["epsilon"]),
                )
                for s in scene["structures"]
            ],
        )
        dx, dt = bz.dxdt(
            1.55e-6,
            n_max=scene["n_max"],
            dims=3,
            safety_factor=0.999,
            points_per_wavelength=self.mesh,
        )
        frequencies = 299792458 / (np.array([1.6, 1.55, 1.5]) * 1e-6)
        pulse = gaussian_band_pulse(
            frequencies,
            carrier_frequency=299792458 / 1.55e-6,
            dt=dt,
            run_after_sources_uoc=90,
            max_output_distance_um=3,
        )
        mode = bz.ModeSpec(polarization="te")

        def port(original, name, inward_offset_um):
            p = scene["ports"][original]
            center = list(p["center_m"])
            center[0] += (1 if p["direction"] == "+" else -1) * inward_offset_um * 1e-6
            return bz.Port(
                center=tuple(center),
                size=tuple(p["size_m"]),
                direction=p["direction"],
                name=name,
                mode_spec=mode,
            )

        src = port("opt1", "source", -1.4)
        input_port = port("opt1", "input", -0.5)
        offsets = [0.05, -0.25, -0.5, -0.75, -1.0]
        outputs = [port("opt2", f"out_{i}", offset) for i, offset in enumerate(offsets)]
        source = bz.ModeSource(
            center=src.center,
            size=src.size,
            direction=src.direction,
            mode_spec=mode,
            source_time=bz.SampledSignal(
                values=pulse.signal,
                quadrature=pulse.signal_quadrature,
                dt=dt,
                freq0=299792458 / 1.55e-6,
            ),
        )
        sim = bz.Simulation(
            design=design,
            sources=[source],
            monitors=[p.to_monitor(frequencies) for p in [input_port, *outputs]],
            boundaries=[
                bz.PML(edges=("left", "right", "top", "bottom", "front", "back"), thickness=1e-6)
            ],
            time=pulse.time,
            resolution=dx,
            setup_device="cpu",
        )
        results = sim.run(
            termination=bz.AutoTermination(
                min_steps=int(np.ceil((pulse.source_end_time + pulse.tail_time) / dt)),
                field_decay=1e-4,
            )
        )
        extracted = s_parameters(
            results, source_port="input", ports=[input_port, *outputs], frequencies=frequencies
        )
        waves = extracted.diagnostics["waves"]
        return {
            "beamz": bz.__version__,
            "jax": importlib.metadata.version("jax"),
            "mesh": self.mesh,
            "dx_nm": dx / 1e-9,
            "wavelength_um": (299792458 / frequencies / 1e-6).tolist(),
            "output_inward_offsets_um": offsets,
            "s21_db": {
                p.name: (20 * np.log10(np.abs(extracted.s_matrix[(p.name, "input")]))).tolist()
                for p in outputs
            },
            "P_in": np.asarray(extracted.diagnostics["P_in"]).tolist(),
            "valid_mask": np.asarray(extracted.diagnostics["valid_mask"]).tolist(),
            "termination": dataclasses.asdict(results.termination),
            "waves": {
                name: {
                    key: {
                        "real": np.real(values[key]).tolist(),
                        "imag": np.imag(values[key]).tolist(),
                    }
                    for key in [
                        "a_plus",
                        "a_minus",
                        "mode_neff",
                        "projection_residual",
                        "condition_number",
                    ]
                }
                for name, values in waves.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(Path(__file__).with_name("sbend_scene.json").read_text())
    record = Probe(scene, args.mesh).run()
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
