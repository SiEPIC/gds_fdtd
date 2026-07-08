# Running gds_fdtd jobs on remote compute

Every simulation is a serializable `JobSpec` (JSON) and every platform is
just an execution backend. The package ships two backends —
`LocalBackend` (in-process) and `SubprocessBackend` (crash-isolated child
process) — and the `gds-fdtd` CLI is the portable entry point the snippets
below build on. **These are reference snippets, not shipped integrations**;
they become extras (`gds_fdtd[modal]`, …) only when demand justifies it.

## The contract

```bash
gds-fdtd run job.json --out results/
# results/result.json  -> job_hash, solver, wall_seconds, smatrix_path
# results/smatrix.npz  -> SMatrix (load with SMatrix.from_npz)
```

- A job file contains **no secrets**. Credentials come from the environment
  on the machine that runs it: `TIDY3D_API_KEY` for tidy3d, the Lumerical
  license/`lumapi` configuration for Lumerical.
- Ship the referenced GDS + technology YAML with the job (they're paths in
  the JobSpec) or point them at a shared filesystem.
- Exit codes: 0 ok · 2 invalid · 3 solver unavailable · 4 budget exceeded.

## SLURM (university cluster with a Lumerical license)

```bash
#!/bin/bash
#SBATCH --job-name=gds-fdtd
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load lumerical/2025R2
export PYTHONPATH=/path/to/lumerical/api/python
source ~/venvs/gds_fdtd/bin/activate

gds-fdtd run "$1" --out "results/${SLURM_JOB_ID}"
```

Submit a sweep: `for j in jobs/*.json; do sbatch run_job.sh "$j"; done`.

## Modal (serverless GPU/CPU)

```python
import modal

app = modal.App("gds-fdtd")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("gds_fdtd[tidy3d]")
    .add_local_dir("jobs", remote_path="/jobs")
)

@app.function(image=image, secrets=[modal.Secret.from_name("tidy3d-api-key")])
def run_job(job_filename: str) -> bytes:
    import pathlib
    import subprocess

    subprocess.run(
        ["gds-fdtd", "run", f"/jobs/{job_filename}", "--out", "/tmp/out"],
        check=True,
    )
    return pathlib.Path("/tmp/out/smatrix.npz").read_bytes()
```

## AWS Batch (container + job queue)

Container: any image with `pip install gds_fdtd[tidy3d]`;
entrypoint `gds-fdtd`. Job definition sketch:

```json
{
  "jobDefinitionName": "gds-fdtd-run",
  "type": "container",
  "containerProperties": {
    "image": "<account>.dkr.ecr.<region>.amazonaws.com/gds-fdtd:latest",
    "command": ["run", "Ref::job_s3_path", "--out", "/results"],
    "secrets": [
      {"name": "TIDY3D_API_KEY", "valueFrom": "arn:aws:secretsmanager:...:tidy3d-key"}
    ],
    "resourceRequirements": [
      {"type": "VCPU", "value": "4"},
      {"type": "MEMORY", "value": "8192"}
    ]
  }
}
```

Stage job JSON + GDS + tech YAML to S3 and pull them in a thin wrapper, or
bake fixed jobs into the image.

## Parallel sweeps from Python (no infrastructure)

```python
from gds_fdtd.execution import JobSpec, SubprocessBackend

backend = SubprocessBackend()
handles = [
    backend.submit(JobSpec.from_file(f"jobs/mesh_{m}.json"), f"out/mesh_{m}")
    for m in (6, 8, 10, 12)
]
results = [backend.result(h) for h in handles]  # runs concurrently
```
