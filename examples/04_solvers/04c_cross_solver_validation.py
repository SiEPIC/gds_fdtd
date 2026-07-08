"""Cross-solver validation (WP5.5) — the payoff of solver agnosticism.

THREE engines ran the IDENTICAL job (gdsfactory straight, unified tech,
mesh 10, zero engine-specific kwargs): tidy3d and Lumerical agree within
0.004 dB; beamz — free and open-source — lands within 0.06 dB of both.
This script reproduces the comparison OFFLINE from the recorded results
shipped in tests/recorded/ (real solver output, 2026-07-08).

To rerun live (tidy3d ~0.05 FC, Lumerical license seat, beamz free):

    from gds_fdtd.validation import validate_across
    report = validate_across(
        [get_solver("tidy3d"), get_solver("lumerical"), get_solver("beamz")],
        component, tech, spec, cache_dir=".gds_fdtd_cache",
    )
"""

import os

from gds_fdtd.smatrix import SMatrix
from gds_fdtd.validation import compare_smatrices

if __name__ == "__main__":
    recorded = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests", "recorded"
    )
    report = compare_smatrices(
        {
            name: SMatrix.from_npz(os.path.join(recorded, f"straight_mesh10_{name}.npz"))
            for name in ("tidy3d", "lumerical", "beamz")
        }
    )
    print(report.summary())
    print("agreement within 0.1 dB:", report.passed(tol_db=0.1))

    # overlay of the thru path from all three engines
    report.plot(out=2, in_=1, savefig="three_engine_agreement.png")
