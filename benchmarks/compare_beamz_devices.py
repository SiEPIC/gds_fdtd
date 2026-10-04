"""Compare fresh BeamZ device runs with the repository's recorded references."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gds_fdtd import SMatrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/devices"
F0 = 299792458 / 1.55e-6
COLORS = {"beamz": "#0072B2", "tidy3d": "#D55E00", "lumerical": "#009E73"}


def db_at(sm: SMatrix, output: int, source: int) -> float:
    return float(np.interp(F0, sm.f, sm.magnitude_db(out=output, in_=source)))


def references(device: str) -> dict[str, SMatrix]:
    if device == "ybranch":
        folder, stem = "07_choosing_an_engine", "ybranch"
    else:
        folder, stem = "10_cookbook", "si_sin_escalator"
    return {
        engine: SMatrix.from_npz(str(ROOT / f"examples/{folder}/recorded/{stem}_{engine}.npz"))
        for engine in ("tidy3d", "lumerical")
    }


def main() -> None:
    records = []
    for path in sorted(RESULTS.glob("*/results.json")):
        r = json.loads(path.read_text())
        sm = SMatrix.from_npz(str(path.parent / "smatrix.npz"))
        row = {
            "run": path.parent.name,
            "device": r["device"],
            "mesh": r["mesh"],
            "run_seconds": r["run_seconds"],
            "finite": r["finite"],
            "reciprocity_max_abs": r["reciprocity_max_abs"],
            "max_power_balance": r["max_power_balance"],
            "all_converged": all(x["termination"]["converged"] for x in r["runs"]),
            "s21_db": db_at(sm, 2, 1),
            "s12_db": db_at(sm, 1, 2),
            "s11_db": db_at(sm, 1, 1),
        }
        modal_sources = {}
        for modal_path in path.parent.glob("modal_*.npz"):
            with np.load(modal_path, allow_pickle=False) as modal:
                modal_sources[modal_path.stem.removeprefix("modal_")] = {
                    "all_incident_samples_valid": bool(np.all(modal["diagnostics/valid_mask"])),
                    "min_incident_power": float(np.min(modal["diagnostics/P_in"])),
                }
        if modal_sources:
            row["modal_sources"] = modal_sources
        if sm.n_ports == 3:
            row.update(s31_db=db_at(sm, 3, 1), s13_db=db_at(sm, 1, 3))
        if r["device"] != "sbend":
            row["references"] = {
                eng: {
                    "s21_db": db_at(ref, 2, 1),
                    "s12_db": db_at(ref, 1, 2),
                    **(
                        {"s31_db": db_at(ref, 3, 1), "s13_db": db_at(ref, 1, 3)}
                        if ref.n_ports == 3
                        else {}
                    ),
                }
                for eng, ref in references(r["device"]).items()
            }
        records.append(row)
    (RESULTS / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    standard = [r for r in records if r["run"] == f"{r['device']}-mesh{r['mesh']}"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for device, ax in zip(("sbend", "ybranch", "escalator"), axes, strict=True):
        runs = sorted([r for r in standard if r["device"] == device], key=lambda r: r["mesh"])
        if not runs:
            continue
        if device == "sbend":
            ax.plot(
                [r["mesh"] for r in runs],
                [r["s21_db"] for r in runs],
                "o-",
                color=COLORS["beamz"],
                label="BeamZ 0.5.2 (fresh)",
            )
            for engine in ("tidy3d", "lumerical", "beamz"):
                path = (
                    ROOT
                    / "examples/06_convergence_and_caching/recorded"
                    / f"sbend_{engine}_convergence.json"
                )
                old = json.loads(path.read_text())["mesh"]
                meshes = sorted(int(m) for m in old)
                ax.plot(
                    meshes,
                    [old[str(m)]["s21_db"] for m in meshes],
                    "s--",
                    color=COLORS[engine] if engine != "beamz" else "0.5",
                    alpha=0.8,
                    label=f"{engine if engine != 'beamz' else 'BeamZ 0.4.3'} (recorded)",
                )
            ax.set(
                xlabel="Mesh setting (engine-specific)",
                ylabel="S21 at 1.55 um (dB)",
                title="Sharp S-bend: convergence",
            )
        else:
            latest = runs[-1]
            sm = SMatrix.from_npz(str(RESULTS / latest["run"] / "smatrix.npz"))
            sources = {"beamz": sm, **references(device)}
            for engine, matrix in sources.items():
                for output in [2, 3] if device == "ybranch" else [2]:
                    ax.plot(
                        matrix.wavelength_um,
                        matrix.magnitude_db(out=output, in_=1),
                        "-" if output == 2 else "--",
                        marker={"beamz": "o", "tidy3d": "s", "lumerical": "^"}[engine],
                        markersize=4,
                        markerfacecolor=COLORS[engine] if output == 2 else "none",
                        color=COLORS[engine],
                        label=f"{engine} S{output}1",
                    )
            ax.set(
                xlabel="Wavelength (um)",
                ylabel="Transmission (dB)",
                title=f"{device.title()}: BeamZ mesh {latest['mesh']}",
            )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Fresh BeamZ 0.5.2 / RTX 3090 vs recorded Tidy3D and Lumerical")
    fig.savefig(RESULTS / "comparison.png", dpi=180)
    fig.savefig(RESULTS / "comparison.svg")
    plt.close(fig)
    for device in ("ybranch", "escalator"):
        runs = sorted([r for r in standard if r["device"] == device], key=lambda r: r["mesh"])
        if not runs:
            continue
        matrices = {
            "beamz": SMatrix.from_npz(str(RESULTS / runs[-1]["run"] / "smatrix.npz")),
            **references(device),
        }
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
        for ax, (engine, sm) in zip(axs, matrices.items(), strict=True):
            values = np.array(
                [[db_at(sm, i + 1, j + 1) for j in range(sm.n_ports)] for i in range(sm.n_ports)]
            )
            im = ax.imshow(values, vmin=-45, vmax=0, cmap="viridis")
            for (i, j), value in np.ndenumerate(values):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if value > -20 else "white",
                    fontsize=11,
                )
            names = [f"opt{i + 1}" for i in range(sm.n_ports)]
            ax.set(
                xticks=range(sm.n_ports),
                yticks=range(sm.n_ports),
                xticklabels=names,
                yticklabels=names,
                xlabel="Input port",
                ylabel="Output port",
                title=engine,
            )
        fig.colorbar(im, ax=axs, label="|S| (dB); colors clipped below -45 dB")
        fig.suptitle(
            f"{device.title()}: full matrix at 1.55 um (magnitudes interpolated in frequency)"
        )
        fig.savefig(RESULTS / f"{device}_matrix.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for mesh, color, marker in ((10, "0.45", "s"), (20, COLORS["beamz"], "o")):
        probe_path = RESULTS / f"sbend-plane-probe-mesh{mesh}/probe.json"
        if not probe_path.exists():
            continue
        probe = json.loads(probe_path.read_text())
        distance = -np.asarray(probe["output_inward_offsets_um"])
        values = [probe["s21_db"][f"out_{i}"][1] for i in range(len(distance))]
        spread = float(np.ptp(np.asarray(values)[distance > 0]))
        ax.plot(
            distance,
            values,
            marker=marker,
            color=color,
            label=f"Mesh {mesh}: {spread:.3f} dB spread in straight lead",
        )
    ax.axvline(0, color="0.7", linestyle=":")
    ax.set(
        xlabel="Distance from bend into output lead (um; negative = inside bend)",
        ylabel="Extracted S21 at 1.55 um (dB)",
        title="Same-run probe: source and input monitor fixed",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(RESULTS / "monitor_sensitivity.png", dpi=180)
    fig.savefig(RESULTS / "monitor_sensitivity.svg")
    plt.close(fig)
    for row in records:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
