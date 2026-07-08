"""Cross-solver validation (WP5.5) — the payoff of solver agnosticism.

``validate_across`` runs the SAME job through several engines and reports the
worst pairwise |ΔS| in dB. This script reproduces that comparison OFFLINE and
free from the recorded artifacts shipped in tests/recorded/: the same Si→SiN
escalator run for real on tidy3d (cloud, 2026-07-07) and Lumerical FDTD v252
(local license) — they agree within a fraction of a dB.

To run the comparison live instead (spends credits + a license seat):

    from gds_fdtd.validation import validate_across
    from gds_fdtd.solvers import get_solver
    report = validate_across(
        [get_solver("tidy3d"), get_solver("lumerical")],
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
            "tidy3d (cloud)": SMatrix.from_hdf5(
                os.path.join(recorded, "si_sin_escalator_smatrix.h5")
            ),
            "lumerical v252": SMatrix.from_hdf5(
                os.path.join(recorded, "si_sin_escalator_lum_smatrix.h5")
            ),
        }
    )
    print(report.summary())
    print("agreement within 1 dB:", report.passed(tol_db=1.0))

    # overlay of the thru path from both engines (ports align by digit id)
    report.plot(out=2, in_=1, savefig="escalator_cross_solver.png")
