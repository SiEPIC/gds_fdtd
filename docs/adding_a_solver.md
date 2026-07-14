# Adding your own solver

This is the point of gds_fdtd: **any FDTD engine becomes a photonics
S-parameter solver by implementing four methods.** Users then drive your
engine with the exact same three lines they use for tidy3d, Lumerical, or
beamz:

```python
solver = get_solver("yourengine")(component, tech, spec)
smatrix = solver.run()
```

## The contract

Subclass `gds_fdtd.solvers.Solver` and implement:

| method | must | must NOT |
|---|---|---|
| `__init__` | store the job (handled by the base class) | touch disk, network, licenses, GPUs — constructors are **cheap and pure** |
| `validate() -> list[str]` | return every problem with the job as a human-readable string; `[]` means runnable | raise for foreseeable problems (return them instead) |
| `build() -> SetupArtifacts` | produce the engine-native scene **offline and deterministically** (two calls → identical artifacts) | contact anything: no cloud, no license checks |
| `estimate() -> ResourceEstimate` | give offline cost hints (cells, memory, #sims) | spend anything |
| `run() -> SMatrix` | execute and return the canonical S-matrix | — this is the **only** method allowed to spend money, license seats, or GPU time |

Declare what your engine can do — these are promises, not probes:

```python
from gds_fdtd.solvers import (
    ResourceEstimate, SetupArtifacts, Solver, SolverCapabilities, register_solver,
)


@register_solver
class YourSolver(Solver):
    name = "yourengine"
    capabilities = SolverCapabilities(
        tier="full",              # "full" = has its own sources/monitors;
                                  # "kernel" = raw eps in, fields out
        execution="local",        # or "cloud"
        supports_dispersion=False,
        supports_sidewall_angle=True,
        supports_multimode=False,
        supports_gpu=False,
        cost_model="free",        # or "licensed" / "credits"
    )
```

## What the base class gives you

- `self.component` — geometry: flat, role-tagged `Structure`s (device /
  substrate / superstrate), `Port`s with positions, widths, directions, and
  `bounds`. Add port-extension stubs with
  `port.polygon_extension(buffer=2 * self.spec.buffer)` so waveguides
  terminate through your absorbing boundary, never on a facet.
- `self.technology` / `Structure.material` — materials with neutral hints:
  `gds_fdtd.grid.resolve_index(material, wavelength_um)` gives you a complex
  refractive index offline (constant `nk`, refractiveindex.info `rii`, or an
  already-built tidy3d medium).
- `self.spec` — every numeric setting, validated (`SimulationSpec`).
- Helpers: `frequencies_hz()`, `injection_plan()` (per-port source
  descriptors), `domain()` (center/span incl. buffer), `describe()`,
  `run_cached(cache_dir)` (free reruns keyed on the full job hash).
- **Kernel-tier engines** (your engine only takes a permittivity grid):
  the full pipeline exists — `grid.rasterize(component, ...)` →
  `modes.Tidy3DModeSolver().solve(...)` (free, local) →
  `extraction.mode_amplitude(fields, mode, ...)` turns your recorded DFT
  fields into S-matrix entries.

## Returning results

Build the canonical `SMatrix` from per-path entries — unmeasured paths stay
NaN and every exporter (.dat / Touchstone / HDF5), checker (reciprocity,
passivity), and plot works immediately:

```python
from gds_fdtd.smatrix import SMatrix

entries = []  # (in_port, out_port, mode_in, mode_out, f_hz, complex_s)
for excitation in self.injection_plan():
    ...run, decompose...
    entries.append((excitation["name"], out_port, 1, 1, f, s_complex))
return SMatrix.from_entries(entries, name=self.component.name)
```

## Registration

- **Inside a plugin package** — declare an entry point; gds_fdtd discovers
  it automatically, and `gds-fdtd solvers` lists it with availability:

  ```toml
  [project.entry-points."gds_fdtd.solvers"]
  yourengine = "your_pkg.solver:YourSolver"
  ```

- **Availability probing** — if your engine may be missing (import, license,
  GPU), add a `@staticmethod probe_available() -> str | None` returning the
  human-readable reason it can't run here (or `None` when fine).

## Testing — you inherit a suite

The conformance suite (`tests/conformance/`) parametrizes over every
registered solver: constructor purity (no files, no sockets), deterministic
builds, artifact serialization, estimate-without-network. Register your
class and you inherit ~30 contract tests for free. Add on top:

1. offline scene assertions (your `build()` output for a known fixture),
2. one real physics smoke on a tiny straight waveguide
   (|S21|² > 0.8, |S11|² < 0.1) — marked `physics`/`licensed`/`cloud`
   per the marker taxonomy so CI never spends your money,
3. if you record real results, sanitize them (`tests/recorded/README.md`).

## Checklist

- [ ] constructor pure; `validate/build/estimate` offline; only `run()` spends
- [ ] `capabilities` accurate; `probe_available()` if the engine can be absent
- [ ] port extensions through your absorbing boundary (S11 below −25 dB on a
      straight waveguide is the sign you got this right; −7 dB means you didn't)
- [ ] `SMatrix.from_entries` with 1-based mode ids; reciprocity/passivity pass
- [ ] entry point registered; conformance suite green
- [ ] secrets from environment variables only — never in code or job files
