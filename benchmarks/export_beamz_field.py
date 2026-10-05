"""Export the saved mesh-30 BeamZ field as a portable, cropped intensity map.

Run after beamz_devices.py sbend --mesh 30 has produced the local field_z.npz.
The complex archive stays local; the notebook replays the derived JSON map.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "benchmarks/results/beamz-0.5.3-convergence/sbend-mesh30"
OUTPUT = ROOT / "examples/06_convergence_and_caching/recorded/sbend_beamz_053_intensity.json"


def main() -> None:
    record = json.loads((RUN / "results.json").read_text())
    setup = json.loads((RUN / "setup.json").read_text())
    assert record["beamz"] == "0.5.3" and record["finite"]
    assert all(r["termination"]["converged"] for r in record["runs"])
    with np.load(RUN / "field_z.npz", allow_pickle=False) as field:
        intensity = sum(np.abs(field[key]) ** 2 for key in ("Ex", "Ey", "Ez"))
        assert np.isfinite(intensity).all() and intensity.max() > 0
        dx = setup["setup"]["dx_nm"] / 1000
        # Device center is (0.5, 0.25) um; native design lower corner is
        # center minus half its extent. Display samples at grid-cell centers.
        x = (np.arange(intensity.shape[1]) + 0.5) * dx + 0.5 - float(field["width_um"]) / 2
        y = (np.arange(intensity.shape[0]) + 0.5) * dx + 0.25 - float(field["height_um"]) / 2
        keep_x = (x >= -1.4 - dx) & (x <= 2.4 + dx)
        keep_y = (y >= -1.75 - dx) & (y <= 2.25 + dx)
        normalized = (intensity / intensity.max())[np.ix_(keep_y, keep_x)]
        payload = {
            "beamz": record["beamz"],
            "mesh": record["mesh"],
            "source": str(field["source"]),
            "wavelength_um": float(np.median(record["wavelength_um"])),
            "z_um": 0.11,
            "dx_um": dx,
            "source_run": str(RUN.relative_to(ROOT)),
            "source_field_sha256": hashlib.sha256((RUN / "field_z.npz").read_bytes()).hexdigest(),
            "gds_sha256": record["gds_sha256"],
            "quantity": "sum(abs(Ex,Ey,Ez)**2), normalized to full-plane maximum",
            "coordinates": "cell-center display coordinates; native Yee components not collocated",
            "notes": "Cropped to notebook viewport; six significant digits; no spatial decimation.",
            "x_um": x[keep_x].tolist(),
            "y_um": y[keep_y].tolist(),
            "intensity_normalized": [
                [float(f"{value:.6g}") for value in row] for row in normalized
            ],
        }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    replay = np.asarray(json.loads(OUTPUT.read_text())["intensity_normalized"])
    np.testing.assert_allclose(replay, normalized, rtol=5e-6, atol=1e-12)
    print(f"Exported {replay.shape} normalized intensity map to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
