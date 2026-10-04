## Observation

While revalidating the GDS_FDTD README devices with BeamZ 0.5.2, the sharp
S-bend's extracted fundamental-TE transmission changes with output-monitor
position **within one FDTD run**. The source and incident monitor are fixed;
several output monitors simultaneously sample the same straight output lead.
This remains measurable after refining from mesh 10 to mesh 20.

This is a focused follow-up to #161, not a claim that the old normalization
failure persists unchanged. The new y-branch full matrix works (including its
formerly dead reverse column), and the escalator agrees well with the recorded
references. I have not isolated whether this remaining sensitivity is
finite-aperture/radiation contamination, Yee-grid/modal-projection error, or a
setup requirement that should be documented. No monitor position is selected
as the "correct" answer merely because it matches a reference.

## Controlled results

All values below are **directly sampled at 1.55 um**, not interpolated.
Output distance is measured outward from the end of the bend into its straight
extension; -0.05 um is inside the bend and is shown only for context. The four
positive-distance planes lie in uniform straight waveguide. They have the same
2.7 um × 2.196 um transverse aperture. Their solved effective indices agree
(to approximately 1e-8 at mesh 10).

| Output distance into straight lead (um) | S21, mesh 10 (dB) | S21, mesh 20 (dB) |
|---:|---:|---:|
| -0.05 | -6.070334 | -5.765388 |
| +0.25 | -5.670282 | -5.605458 |
| +0.50 | -5.641913 | -5.624334 |
| +0.75 | -5.821254 | -5.694380 |
| +1.00 | -5.956498 | -5.735579 |

The spread among the **straight-lead planes only** is 0.314585 dB at mesh 10
and 0.130121 dB at mesh 20. The input monitor is 0.5 um before the device;
the source is 1.4 um before it, so neither moves between these measurements.
This is not a comparison between separately excited simulations.

- All sampled incident-wave masks are valid. Mesh-10 incident modal power at
  1.55 um is 1.001793985 after BeamZ source normalization.
- Automatic termination reports `converged` for both runs. At mesh 10:
  energy/peak = 1.33e-10, monitor change =
  1.79e-07; at mesh 20:
  9.32e-11 and 6.88e-08.
- Mode-projection condition numbers are 1. Projection residuals are high
  (see raw samples below); the bend strongly radiates. That may be relevant,
  and is a reason to investigate rather than assume an extraction defect.
- BeamZ emits the absorber-normal material-variation warning on left/right
  boundaries. Port rectangles extend through the full raster domain, including
  one extra grid cell. This warning is not suppressed.

Independent historical references for this geometry converge near S21
**-5.63544 dB (Tidy3D mesh 25)** and **-5.63251 dB (Lumerical accuracy 5)**:
[Tidy3D data](https://github.com/SiEPIC/gds_fdtd/blob/66b28f1c26649d4362aec2375d891ad0cd60c67e/examples/06_convergence_and_caching/recorded/sbend_tidy3d_convergence.json),
[Lumerical data](https://github.com/SiEPIC/gds_fdtd/blob/66b28f1c26649d4362aec2375d891ad0cd60c67e/examples/06_convergence_and_caching/recorded/sbend_lumerical_convergence.json).
Those are recorded references, not fresh commercial-engine reruns; differences
in grid, material dispersion, and sidewalls limit absolute cross-engine
comparisons. They are contextual evidence, not necessary to observe the
same-run output-plane sensitivity.

## Environment and geometry

- BeamZ **0.5.2**, PyPI release; release-tag commit
  `160b06f74ea9f67ec313ef390fc655ccf232be57`.
- JAX **0.10.2**, CUDA 12 packages; local RTX 3090 (24 GiB), NVIDIA driver 610.43.03.
- Three frequencies: wavelengths **1.6, 1.55, 1.5 um**.
- Uniform 3D Yee grid: 44.591485 nm (mesh 10), 22.295742 nm (mesh 20).
- Silicon n=3.476, oxide n=1.444, core thickness 0.22 um; vertical extrusion.
- Bend footprint: 1 um long, 0.5 um offset, 0.5 um guide width. Domain:
  7 × 7 × 5.22 um. PML/absorber thickness 1 um, default sponge formulation.
- Exact polygon source: `sbend_dontfabme` from GDS_FDTD
  [`examples/devices.gds`](https://github.com/SiEPIC/gds_fdtd/blob/66b28f1c26649d4362aec2375d891ad0cd60c67e/examples/devices.gds),
  SHA256 `efcb1f267ccc1b0296c6531814127b29c080e6c7b9b03c126f75f1e9849218ee`.
  This is a repository GDS cell, not a PDK-generated cell.

## Standalone reproduction

Save the two files below beside each other. **No GDS_FDTD, KLayout, or PDK is
needed.** The mesh-10 standalone script was run in a fresh interpreter and
reproduced all 15 complex-magnitude samples within 1e-6 dB of the integration
probe. The mesh-20 measurement used the same native BeamZ source/monitor/run/
analysis sequence, with the GDS_FDTD offline builder preparing its geometry.

```bash
pip install 'beamz==0.5.2' 'jax[cuda12]'
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python beamz_plane_dependence.py --mesh 10 --output probe10.json
# Finer check (several minutes on an RTX 3090):
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python beamz_plane_dependence.py --mesh 20 --output probe20.json
```

The script saves the S values, raw directional modal amplitudes, effective
indices, residuals, validity masks, and termination diagnostics.

<details><summary>beamz_plane_dependence.py</summary>

```python
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
```

</details>

<details><summary>sbend_scene.json (coordinates in meters)</summary>

```json
{
  "size_m": [
    7e-06,
    7e-06,
    5.219999999999999e-06
  ],
  "background_epsilon": 2.085136,
  "n_max": 3.476,
  "structures": [
    {
      "vertices_m": [
        [
          3.038e-06,
          3e-06,
          2.4999999999999998e-06
        ],
        [
          3.074e-06,
          3.0020000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.1090000000000002e-06,
          3.0040000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.1430000000000002e-06,
          3.007e-06,
          2.4999999999999998e-06
        ],
        [
          3.176e-06,
          3.011e-06,
          2.4999999999999998e-06
        ],
        [
          3.209e-06,
          3.0160000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.271e-06,
          3.028e-06,
          2.4999999999999998e-06
        ],
        [
          3.3010000000000002e-06,
          3.0360000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.3300000000000003e-06,
          3.045e-06,
          2.4999999999999998e-06
        ],
        [
          3.358e-06,
          3.0550000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.412e-06,
          3.0770000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.438e-06,
          3.09e-06,
          2.4999999999999998e-06
        ],
        [
          3.486e-06,
          3.118e-06,
          2.4999999999999998e-06
        ],
        [
          3.53e-06,
          3.1500000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.55e-06,
          3.1660000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.569e-06,
          3.1830000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.603e-06,
          3.2170000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.618e-06,
          3.2340000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.632e-06,
          3.2510000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.645e-06,
          3.2680000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.658e-06,
          3.2860000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.678e-06,
          3.3140000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.696e-06,
          3.3420000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.712e-06,
          3.367e-06,
          2.4999999999999998e-06
        ],
        [
          3.725e-06,
          3.389e-06,
          2.4999999999999998e-06
        ],
        [
          3.737e-06,
          3.407e-06,
          2.4999999999999998e-06
        ],
        [
          3.75e-06,
          3.4250000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.753e-06,
          3.429e-06,
          2.4999999999999998e-06
        ],
        [
          3.757e-06,
          3.435e-06,
          2.4999999999999998e-06
        ],
        [
          3.761e-06,
          3.44e-06,
          2.4999999999999998e-06
        ],
        [
          3.7750000000000003e-06,
          3.4540000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.779e-06,
          3.457e-06,
          2.4999999999999998e-06
        ],
        [
          3.782e-06,
          3.4590000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.79e-06,
          3.4650000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.795e-06,
          3.467e-06,
          2.4999999999999998e-06
        ],
        [
          3.801e-06,
          3.4700000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.815e-06,
          3.4760000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.824e-06,
          3.479e-06,
          2.4999999999999998e-06
        ],
        [
          3.834e-06,
          3.4820000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.846e-06,
          3.485e-06,
          2.4999999999999998e-06
        ],
        [
          3.859e-06,
          3.4880000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.874e-06,
          3.491e-06,
          2.4999999999999998e-06
        ],
        [
          3.89e-06,
          3.4940000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.909e-06,
          3.496e-06,
          2.4999999999999998e-06
        ],
        [
          3.929e-06,
          3.4980000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.951e-06,
          3.4990000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.975e-06,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          4e-06,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          4e-06,
          4.000000000000001e-06,
          2.4999999999999998e-06
        ],
        [
          3.962e-06,
          4.000000000000001e-06,
          2.4999999999999998e-06
        ],
        [
          3.926e-06,
          3.9980000000000005e-06,
          2.4999999999999998e-06
        ],
        [
          3.8910000000000005e-06,
          3.996e-06,
          2.4999999999999998e-06
        ],
        [
          3.857e-06,
          3.9930000000000006e-06,
          2.4999999999999998e-06
        ],
        [
          3.824e-06,
          3.989e-06,
          2.4999999999999998e-06
        ],
        [
          3.791e-06,
          3.984e-06,
          2.4999999999999998e-06
        ],
        [
          3.729e-06,
          3.972e-06,
          2.4999999999999998e-06
        ],
        [
          3.699e-06,
          3.9640000000000005e-06,
          2.4999999999999998e-06
        ],
        [
          3.67e-06,
          3.955e-06,
          2.4999999999999998e-06
        ],
        [
          3.642e-06,
          3.945e-06,
          2.4999999999999998e-06
        ],
        [
          3.588e-06,
          3.923e-06,
          2.4999999999999998e-06
        ],
        [
          3.562e-06,
          3.91e-06,
          2.4999999999999998e-06
        ],
        [
          3.514e-06,
          3.882e-06,
          2.4999999999999998e-06
        ],
        [
          3.4700000000000002e-06,
          3.85e-06,
          2.4999999999999998e-06
        ],
        [
          3.45e-06,
          3.834e-06,
          2.4999999999999998e-06
        ],
        [
          3.431e-06,
          3.817e-06,
          2.4999999999999998e-06
        ],
        [
          3.3970000000000003e-06,
          3.783e-06,
          2.4999999999999998e-06
        ],
        [
          3.382e-06,
          3.766e-06,
          2.4999999999999998e-06
        ],
        [
          3.368e-06,
          3.749e-06,
          2.4999999999999998e-06
        ],
        [
          3.355e-06,
          3.732e-06,
          2.4999999999999998e-06
        ],
        [
          3.342e-06,
          3.7140000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.322e-06,
          3.6860000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.304e-06,
          3.6580000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.288e-06,
          3.633e-06,
          2.4999999999999998e-06
        ],
        [
          3.275e-06,
          3.6110000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.263e-06,
          3.593e-06,
          2.4999999999999998e-06
        ],
        [
          3.2500000000000002e-06,
          3.575e-06,
          2.4999999999999998e-06
        ],
        [
          3.247e-06,
          3.571e-06,
          2.4999999999999998e-06
        ],
        [
          3.243e-06,
          3.565e-06,
          2.4999999999999998e-06
        ],
        [
          3.2390000000000002e-06,
          3.5600000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.225e-06,
          3.546e-06,
          2.4999999999999998e-06
        ],
        [
          3.221e-06,
          3.5430000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.218e-06,
          3.541e-06,
          2.4999999999999998e-06
        ],
        [
          3.21e-06,
          3.535e-06,
          2.4999999999999998e-06
        ],
        [
          3.2050000000000002e-06,
          3.5330000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.199e-06,
          3.53e-06,
          2.4999999999999998e-06
        ],
        [
          3.185e-06,
          3.524e-06,
          2.4999999999999998e-06
        ],
        [
          3.176e-06,
          3.5210000000000003e-06,
          2.4999999999999998e-06
        ],
        [
          3.1660000000000003e-06,
          3.518e-06,
          2.4999999999999998e-06
        ],
        [
          3.154e-06,
          3.5150000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.141e-06,
          3.5120000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.1260000000000002e-06,
          3.509e-06,
          2.4999999999999998e-06
        ],
        [
          3.11e-06,
          3.5060000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3.091e-06,
          3.5040000000000002e-06,
          2.4999999999999998e-06
        ],
        [
          3.071e-06,
          3.502e-06,
          2.4999999999999998e-06
        ],
        [
          3.049e-06,
          3.501e-06,
          2.4999999999999998e-06
        ],
        [
          3.0250000000000003e-06,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3e-06,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          3e-06,
          3e-06,
          2.4999999999999998e-06
        ]
      ],
      "z_m": 2.4999999999999998e-06,
      "depth_m": 2.1999999999999998e-07,
      "epsilon": 12.082576
    },
    {
      "vertices_m": [
        [
          4e-06,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          7.044591484464902e-06,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          7.044591484464902e-06,
          4.000000000000001e-06,
          2.4999999999999998e-06
        ],
        [
          4e-06,
          4.000000000000001e-06,
          2.4999999999999998e-06
        ]
      ],
      "z_m": 2.4999999999999998e-06,
      "depth_m": 2.1999999999999998e-07,
      "epsilon": 12.082576
    },
    {
      "vertices_m": [
        [
          3e-06,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          -4.459148446490207e-08,
          3.5000000000000004e-06,
          2.4999999999999998e-06
        ],
        [
          -4.459148446490207e-08,
          3e-06,
          2.4999999999999998e-06
        ],
        [
          3e-06,
          3e-06,
          2.4999999999999998e-06
        ]
      ],
      "z_m": 2.4999999999999998e-06,
      "depth_m": 2.1999999999999998e-07,
      "epsilon": 12.082576
    }
  ],
  "ports": {
    "opt2": {
      "center_m": [
        4e-06,
        3.75e-06,
        2.6099999999999996e-06
      ],
      "size_m": [
        0,
        2.7e-06,
        2.196e-06
      ],
      "direction": "-"
    },
    "opt1": {
      "center_m": [
        3e-06,
        3.2500000000000002e-06,
        2.6099999999999996e-06
      ],
      "size_m": [
        0,
        2.7e-06,
        2.196e-06
      ],
      "direction": "+"
    }
  }
}
```

</details>

## Raw mesh-10 modal samples at 1.55 um

`a_plus`/`a_minus` are the native analysis branch names; for +x propagation in
this 3D setup the transmitted wave is `a_minus`. Output monitors point -x into
the device. Each row is a different probe in the **same** run.

| Monitor | a_plus | a_minus | n_eff | Projection residual |
|---|---|---|---:|---:|
| input | `+0.008527649+0.024700382j` | `-0.992656594-0.128167355j` | 2.441002934 | 0.088029 |
| out_0 | `+0.000130218-0.012665761j` | `+0.355092363+0.348577446j` | 2.440480953 | 0.954424 |
| out_1 | `-0.002327933+0.009302636j` | `-0.424732254-0.301811588j` | 2.440050892 | 0.855849 |
| out_2 | `+0.014796335-0.004557642j` | `+0.521545592-0.035456539j` | 2.440050904 | 0.852323 |
| out_3 | `-0.007323564-0.003395310j` | `-0.380418781+0.342773708j` | 2.440050904 | 0.909371 |
| out_4 | `+0.006950571+0.002108014j` | `+0.098051818-0.494528442j` | 2.440050892 | 0.932763 |

## Requested investigation

Please establish output-plane/aperture convergence for this radiating device:
is the observed variation an expected limitation of the finite-aperture modal
estimator, or a projection/colocation problem? A regression test and guidance
on required straight-lead distance/aperture would help downstream compact-model
users avoid reporting a location-dependent transmission as a converged result.
