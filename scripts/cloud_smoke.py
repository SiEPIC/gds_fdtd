#!/usr/bin/env python3
"""Tidy3D cloud smoke test (WP7.5.1) — budget-gated, human-triggered.

Builds the tiny Si->SiN escalator fixture, estimates the FlexCredit cost of
every task OFFLINE-SAFELY (upload -> estimate_cost -> delete; estimates are
free), ABORTS if the total exceeds $BUDGET_FC (default 0.5), then runs and
asserts the physics. Credentials come from TIDY3D_API_KEY in the environment.

    BUDGET_FC=0.5 python scripts/cloud_smoke.py [--out results/] [--refresh-fixtures]

Exit codes: 0 ok · 4 budget exceeded · 1 physics/infra failure.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cloud_smoke_out")
    parser.add_argument(
        "--refresh-fixtures",
        action="store_true",
        help="also write tests/recorded/-style artifacts for a human to commit",
    )
    args = parser.parse_args()
    budget_fc = float(os.environ.get("BUDGET_FC", "0.5"))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.simprocessor import load_component_from_tech
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec

    tech = parse_yaml_tech(str(REPO / "tests" / "tech_tidy3d.yaml"))
    cell, layout = load_cell(str(REPO / "tests" / "si_sin_escalator.gds"))
    component = load_component_from_tech(cell=cell, tech=tech)
    spec = SimulationSpec(wavelength_points=11, mesh=6, z_min=-1.0, z_max=1.11)
    solver = get_solver("tidy3d")(component, tech, spec, workdir=out)

    problems = solver.validate()
    if problems:
        print(f"job invalid: {problems}", file=sys.stderr)
        return 1
    modeler = solver.build().native

    # --- budget gate: estimates are free (upload -> estimate -> delete) ---
    import tidy3d.web as web

    total_fc = 0.0
    for task_name, sim in modeler.sim_dict.items():
        tid = web.upload(sim, task_name=f"smoke_{task_name}", verbose=False)
        fc = float(web.estimate_cost(tid, verbose=False))
        web.delete(tid)
        total_fc += fc
        print(f"estimate {task_name}: {fc:.3f} FC")
    print(f"total estimated: {total_fc:.3f} FC (budget {budget_fc})")
    if total_fc > budget_fc:
        print("BUDGET EXCEEDED - aborting before any run", file=sys.stderr)
        return 4

    # --- run + physics assertions ---
    sm = solver.run()
    s21_max = float(sm.magnitude_db(out=2, in_=1).max())
    s11_max = float(sm.magnitude_db(out=1, in_=1).max())
    checks = {
        "|S21| max > -1 dB": s21_max > -1.0,
        "|S11| max < -20 dB": s11_max < -20.0,
        "reciprocal": sm.is_reciprocal(atol=0.05),
        "passive (tol 0.02)": sm.is_passive(atol=0.02),
    }
    sm.to_npz(str(out / "smoke_smatrix.npz"))
    if args.refresh_fixtures:
        sm.to_hdf5(str(out / "si_sin_escalator_smatrix.h5"))
        sm.to_dat(str(out / "si_sin_escalator.dat"))

    lines = [f"S21 max: {s21_max:.2f} dB · S11 max: {s11_max:.2f} dB"]
    lines += [f"{'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()]
    lines += [f"estimated cost: {total_fc:.3f} FC (budget {budget_fc})"]
    report = "\n".join(lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        pathlib.Path(summary).write_text(f"## tidy3d cloud smoke\n\n```\n{report}\n```\n")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
