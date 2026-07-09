"""
gds_fdtd simulation toolbox.

Command-line interface (WP7.3). Every simulation is a JSON JobSpec; every
subcommand consumes one (except ``convert``/``solvers``).

    gds-fdtd solvers                        # what engines are available here
    gds-fdtd validate job.json
    gds-fdtd build job.json --out setup/    # offline artifacts, free
    gds-fdtd estimate job.json
    gds-fdtd run job.json --out results/    # the ONLY money-spending command
    gds-fdtd convert results.dat --to snp

Exit codes: 0 ok · 2 validation failed · 3 solver unavailable ·
4 budget exceeded · 1 anything else.

Credentials are read from the environment ONLY (TIDY3D_API_KEY, Lumerical
license config); a job file never contains secrets and is safe to ship.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INVALID = 2
EXIT_UNAVAILABLE = 3
EXIT_BUDGET = 4


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=repr))
    else:
        for k, v in payload.items():
            print(f"{k}: {v}")


def _load_job(path: str):
    from .execution import JobSpec

    return JobSpec.from_file(path)


def _check_available(name: str) -> str | None:
    """None if the solver is usable here, else the reason it isn't."""
    from .solvers import available_solvers

    status = available_solvers().get(name)
    if status is None:
        return f"unknown solver {name!r}; registered: {sorted(available_solvers())}"
    return None if status == "ok" else status


def cmd_solvers(args) -> int:
    from .solvers import available_solvers, get_solver

    rows = {}
    for name, status in sorted(available_solvers().items()):
        try:
            caps = get_solver(name).capabilities
            rows[name] = {
                "available": status,
                "execution": caps.execution,
                "cost": caps.cost_model,
                "multimode": caps.supports_multimode,
            }
        except Exception as e:  # entry point failed to load
            rows[name] = {"available": f"broken: {e}"}
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for name, info in rows.items():
            print(
                f"{name:12s} {info.get('available'):20s} "
                + " ".join(f"{k}={v}" for k, v in info.items() if k != "available")
            )
    return EXIT_OK


def cmd_validate(args) -> int:
    job = _load_job(args.job)
    reason = _check_available(job.solver)
    if reason:
        _emit({"solver": job.solver, "unavailable": reason}, args.json)
        return EXIT_UNAVAILABLE
    problems = job.make_solver().validate()
    _emit({"valid": not problems, "problems": problems}, args.json)
    return EXIT_OK if not problems else EXIT_INVALID


def cmd_build(args) -> int:
    job = _load_job(args.job)
    reason = _check_available(job.solver)
    if reason:
        _emit({"solver": job.solver, "unavailable": reason}, args.json)
        return EXIT_UNAVAILABLE
    solver = job.make_solver(workdir=args.out)
    problems = solver.validate()
    if problems:
        _emit({"valid": False, "problems": problems}, args.json)
        return EXIT_INVALID
    artifacts = solver.build()
    _emit(
        {
            "files": {k: str(v) for k, v in artifacts.files.items()},
            "summary": artifacts.summary,
        },
        args.json,
    )
    return EXIT_OK


def cmd_estimate(args) -> int:
    job = _load_job(args.job)
    reason = _check_available(job.solver)
    if reason:
        _emit({"solver": job.solver, "unavailable": reason}, args.json)
        return EXIT_UNAVAILABLE
    est = job.make_solver().estimate()
    _emit(
        {
            "grid_cells": est.grid_cells,
            "memory_gb": est.memory_gb,
            "n_simulations": est.n_simulations,
            "cost_hint": est.cost_hint,
        },
        args.json,
    )
    return EXIT_OK


def cmd_run(args) -> int:
    from .execution import SubprocessBackend, run_job

    job = _load_job(args.job)
    reason = _check_available(job.solver)
    if reason:
        _emit({"solver": job.solver, "unavailable": reason}, args.json)
        return EXIT_UNAVAILABLE
    try:
        if args.backend == "local":
            result = run_job(job, args.out)
        else:
            backend = SubprocessBackend(extra_imports=tuple(args.imports or ()))
            result = backend.result(backend.submit(job, args.out))
    except ValueError as e:
        _emit({"valid": False, "problems": str(e)}, args.json)
        return EXIT_INVALID
    except (PermissionError, TimeoutError) as e:
        _emit({"budget_exceeded": str(e)}, args.json)
        return EXIT_BUDGET
    _emit(result.model_dump(), args.json)
    return EXIT_OK


def cmd_convert(args) -> int:
    from .smatrix import SMatrix

    src = args.results
    if src.endswith(".dat"):
        sm = SMatrix.from_dat(src)
    elif src.endswith(".npz"):
        sm = SMatrix.from_npz(src)
    elif src.endswith((".h5", ".hdf5")):
        sm = SMatrix.from_hdf5(src)
    else:
        _emit({"error": f"unsupported input {src!r} (use .dat/.npz/.h5)"}, args.json)
        return EXIT_INVALID
    base = src.rsplit(".", 1)[0]
    if args.to == "snp":
        out = sm.to_touchstone(f"{base}.s{sm.n_ports * sm.n_modes}p")
    elif args.to == "dat":
        out = sm.to_dat(f"{base}.dat")
    elif args.to == "npz":
        out = sm.to_npz(f"{base}.npz")
    else:  # h5
        out = sm.to_hdf5(f"{base}.h5")
    _emit({"written": out}, args.json)
    return EXIT_OK


def cmd_convert_tech(args) -> int:
    """Migrate a v1 technology YAML to schema v2 (named materials)."""
    import yaml

    from .technology import Technology

    tech = Technology.from_yaml(args.tech)

    # dedupe identical material mappings into named entries
    def v2_material(legacy: dict) -> dict:
        out = {}
        for key, value in legacy.items():
            if key == "tidy3d_db":
                out["tidy3d"] = value.get("model", value.get("nk"))
            elif key == "lum_db":
                out["lumerical"] = value["model"]
            else:
                out[key] = value
        return out

    def suggest_name(mat: dict, taken: set[str]) -> str:
        base = None
        lum = mat.get("lumerical")
        if isinstance(lum, str):
            base = lum.split()[0].strip("(),")
        elif isinstance(mat.get("tidy3d"), list):
            base = str(mat["tidy3d"][0])
        base = base or f"mat{len(taken) + 1}"
        name, i = base, 1
        while name in taken:
            i += 1
            name = f"{base}{i}"
        return name

    named: dict[str, dict] = {}

    def register(legacy: dict) -> str:
        mat = v2_material(legacy)
        for name, existing in named.items():
            if existing == mat:
                return name
        name = suggest_name(mat, set(named))
        named[name] = mat
        return name

    legacy = tech.to_legacy_dict()
    doc: dict = {
        "name": legacy["name"],
        "schema_version": 2,
        "materials": named,  # filled by register() below
        "substrate": {
            "z_base": legacy["substrate"][0]["z_base"],
            "z_span": legacy["substrate"][0]["z_span"],
            "material": register(legacy["substrate"][0]["material"]),
        },
        "superstrate": {
            "z_base": legacy["superstrate"][0]["z_base"],
            "z_span": legacy["superstrate"][0]["z_span"],
            "material": register(legacy["superstrate"][0]["material"]),
        },
        "pinrec": legacy["pinrec"],
        "devrec": legacy["devrec"],
        "device": [
            {
                "layer": d["layer"],
                "z_base": d["z_base"],
                "z_span": d["z_span"],
                "material": register(d["material"]),
                "sidewall_angle": d["sidewall_angle"],
            }
            for d in legacy["device"]
        ],
    }
    out_path = args.out or args.tech.rsplit(".", 1)[0] + "_v2.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump({"technology": doc}, f, sort_keys=False, default_flow_style=None)
    # round-trip guard: the emitted v2 must reproduce the v1 legacy dict
    converted = Technology.from_yaml(out_path).to_legacy_dict()
    if converted != legacy:
        _emit({"error": f"round-trip mismatch writing {out_path}"}, args.json)
        return EXIT_INVALID
    _emit({"written": out_path, "materials": sorted(named)}, args.json)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gds-fdtd",
        description="GDS layout -> FDTD simulation -> S-parameters, solver-agnostic.",
        epilog="Secrets come from the environment only (e.g. TIDY3D_API_KEY); "
        "job files are safe to ship to other machines.",
    )
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        metavar="MODULE",
        help="import a python module first (lets external solver plugins register)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("solvers", help="list registered engines and availability")
    for name in ("validate", "estimate"):
        p = sub.add_parser(name)
        p.add_argument("job", help="JobSpec JSON file")
    p = sub.add_parser("build", help="produce offline setup artifacts (free)")
    p.add_argument("job")
    p.add_argument("--out", default="build_artifacts")
    p = sub.add_parser("run", help="execute the simulation (may spend money/licenses)")
    p.add_argument("job")
    p.add_argument("--out", default="results")
    p.add_argument("--backend", choices=("local", "subprocess"), default="local")
    p = sub.add_parser("convert-tech", help="migrate a v1 technology YAML to schema v2")
    p.add_argument("tech", help="path to a v1 technology YAML")
    p.add_argument("-o", "--out", help="output path (default: <name>_v2.yaml)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("convert", help="convert S-parameter files between formats")
    p.add_argument("results")
    p.add_argument("--to", choices=("snp", "dat", "npz", "h5"), required=True)

    args = parser.parse_args(argv)
    for mod in args.imports or ():
        importlib.import_module(mod)

    handler = {
        "solvers": cmd_solvers,
        "validate": cmd_validate,
        "build": cmd_build,
        "estimate": cmd_estimate,
        "run": cmd_run,
        "convert": cmd_convert,
        "convert-tech": cmd_convert_tech,
    }[args.command]
    try:
        return handler(args)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
