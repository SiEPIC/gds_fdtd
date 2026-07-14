# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 09 · The CLI and serializable jobs
#
# Everything so far has been Python. For **remote and batch** compute you want a
# job you can hand to another machine — a cluster node, a cloud runner — without
# shipping your session. `gds_fdtd` gives you two things for that:
#
# - a **`JobSpec`**: one JSON document that fully describes a simulation
#   (layout file, technology file, `SimulationSpec`, engine, budget) — no live
#   objects, no secrets;
# - the **`gds-fdtd` CLI**: `solvers | validate | build | estimate | run |
#   convert | convert-tech`, with an explicit exit-code contract.
#
# `validate` / `build` / `estimate` are **offline and free** — the CLI previews a
# job (and its cost) before `run` ever touches an engine. This whole notebook
# runs offline.

# %%
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from gds_fdtd.execution import JobSpec

# Run from the repo root so every path in the job is relative and portable.
_root = next(b for b in (Path.cwd(), *Path.cwd().parents) if (b / "pyproject.toml").exists())
os.chdir(_root)


def _cli(*args: str) -> None:
    """Run `python -m gds_fdtd.cli ...` and show the exit code + output."""
    r = subprocess.run(
        [sys.executable, "-m", "gds_fdtd.cli", *args], capture_output=True, text=True
    )
    print(f"$ gds-fdtd {' '.join(args)}   →  exit {r.returncode}")
    print((r.stdout or r.stderr).rstrip() or "(no output)")


# %% [markdown]
# ## 1 · A job as a file
#
# A `JobSpec` references its layout and technology **by path** and carries the
# validated `SimulationSpec`. We point it at the Si→SiN escalator layout from
# `10_cookbook` and serialize the whole job to JSON.

# %%
workdir = Path(tempfile.mkdtemp(prefix="gdsfdtd_job_"))
job = JobSpec(
    gds_path="examples/10_cookbook/si_sin_escalator.gds",
    technology_path="examples/tech.yaml",
    solver="beamz",
    spec={"wavelength_start": 1.5, "wavelength_end": 1.6, "wavelength_points": 5, "mesh": 6,
          "z_min": -1.0, "z_max": 1.11},
)
job.to_file(workdir / "job.json")
print((workdir / "job.json").read_text())

# %% [markdown]
# The job is pure JSON — safe to commit, ship, or queue. Secrets (API keys,
# license servers) stay in the environment of whatever machine runs it, never in
# the file. It round-trips exactly:

# %%
reloaded = JobSpec.from_file(workdir / "job.json")
print("round-trips identically:", reloaded == job)

# %% [markdown]
# ## 2 · Driving it from the shell
#
# `solvers` reports the engines registered on this machine and their cost model —
# handy before you pick one for a job:

# %%
_cli("solvers")

# %% [markdown]
# `validate` / `estimate` / `build` preview the job **offline** (exit 0 = ready;
# 2 = the job has problems; 3 = the engine isn't usable here; 4 = over budget):

# %%
_cli("validate", str(workdir / "job.json"))
_cli("estimate", str(workdir / "job.json"))
_cli("build", str(workdir / "job.json"))

# %% [markdown]
# ## 3 · Running — locally or crash-isolated
#
# `run` executes the job; `--backend subprocess` runs it in a **separate process**
# (so one bad job can't take down a sweep) and enforces the job's wall-clock
# budget. The subprocess backend re-invokes this same CLI, which is why
# the JSON boundary matters — the child rebuilds the job from the file alone:
#
# ```bash
# gds-fdtd run job.json --out results/                 # in-process
# gds-fdtd run job.json --out results/ --backend subprocess   # crash-isolated
# ```
#
# ```python
# from gds_fdtd.execution import SubprocessBackend
# handle = SubprocessBackend().submit(job, out_dir="results")
# result = handle.result()          # JobResult: smatrix_path, job_hash, wall_seconds, ...
# ```
#
# `run` is the only verb that spends — and a `Budget` on the `JobSpec`
# (`max_flexcredits`, `max_wall_seconds`) lets the CLI **refuse before spending**
# (exit 4). See `docs/remote_compute.md` for Modal / AWS / SLURM recipes.
#
# ## Recap & next
#
# A simulation is a portable JSON `JobSpec`; the `gds-fdtd` CLI previews it for
# free and runs it locally or on a cluster, with budgets enforced up front. Next:
# **`10_cookbook`** — reference devices with known-good S-parameters.
