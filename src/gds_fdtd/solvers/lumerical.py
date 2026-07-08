"""
gds_fdtd simulation toolbox.

LumericalSolver: the Ansys Lumerical FDTD adapter on the Phase-3 Solver
contract (WP3.1d). ``build()`` generates the COMPLETE setup script (.lsf) as
pure text plus the exported GDS — offline, deterministic, no license, no
lumapi import — mirroring the validated legacy ``fdtd_solver_lumerical``
semantics (incl. the F6/F7 Lumerical-2025 fixes). ``run()`` is the only
method that opens a lumapi session (license checkout).

Unlike the legacy layer-builder flow (which asked the live session which GDS
layers exist), the script generator derives the present layers from the
component's structures — one reason build() needs no session.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..smatrix import SMatrix
from .base import (
    ResourceEstimate,
    SetupArtifacts,
    Solver,
    SolverCapabilities,
    register_solver,
)


def probe_lumapi() -> str | None:
    """None if lumapi is importable, else the reason (checks LUMERICAL_API_PATH
    and common install locations)."""
    try:
        import lumapi  # noqa: F401

        return None
    except ImportError:
        pass
    import glob
    import sys

    candidates = sorted(glob.glob("/opt/lumerical/*/api/python")) + (
        [os.environ["LUMERICAL_API_PATH"]] if "LUMERICAL_API_PATH" in os.environ else []
    )
    for c in reversed(candidates):  # newest install first
        if os.path.exists(os.path.join(c, "lumapi.py")):
            sys.path.append(c)
            try:
                import lumapi  # noqa: F401

                return None
            except ImportError as e:  # pragma: no cover - env dependent
                return f"lumapi found at {c} but failed to import: {e}"
    return "lumapi not importable (set LUMERICAL_API_PATH or install Lumerical)"


def _q(s: str) -> str:
    """Quote a string for .lsf (escape embedded double quotes)."""
    return '"' + str(s).replace('"', '\\"') + '"'


@register_solver
class LumericalSolver(Solver):
    """Ansys Lumerical FDTD adapter (tier: full-service, execution: local)."""

    name = "lumerical"
    capabilities = SolverCapabilities(
        tier="full",
        execution="local",
        supports_dispersion=True,
        supports_sidewall_angle=True,
        supports_multimode=True,
        supports_gpu=True,
        cost_model="licensed",
    )

    def __init__(self, *args, gpu: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu = gpu

    @staticmethod
    def probe_available() -> str | None:
        # build() is pure text generation and needs nothing; run() re-probes.
        return None

    # ---------------- lifecycle ----------------

    def validate(self) -> list[str]:
        problems = []
        if not self.component.ports:
            problems.append("component has no ports")
        if self.technology is None:
            problems.append("LumericalSolver requires a technology (materials)")
            return problems
        tech = self._tech_dict()
        for i, d in enumerate(tech.get("device", [])):
            mat = d.get("material", {})
            if "lum_db" not in mat:
                problems.append(f"device layer {i} has no 'lum_db' material entry")
        for key in ("substrate", "superstrate"):
            mat = tech.get(key, [{}])[0].get("material", {})
            if "lum_db" not in mat:
                problems.append(f"{key} has no 'lum_db' material entry")
        return problems

    def _tech_dict(self) -> dict:
        t = self.technology
        return t.to_legacy_dict() if hasattr(t, "to_legacy_dict") else t

    def build(self) -> SetupArtifacts:
        """Export the GDS and generate the full .lsf setup script (offline)."""
        problems = self.validate()
        if problems:
            raise ValueError("cannot build: " + "; ".join(problems))

        if self.workdir is not None:
            workdir = Path(self.workdir)
        else:
            # cache the tempdir: build() must be deterministic across calls
            if not hasattr(self, "_tmp_workdir"):
                self._tmp_workdir = Path(tempfile.mkdtemp(prefix="gds_fdtd_lum_"))
            workdir = self._tmp_workdir
        workdir.mkdir(parents=True, exist_ok=True)

        gds_name = f"{self.component.name}.gds"
        self.component.export_gds(export_dir=str(workdir), buffer=2 * self.spec.buffer)

        script = self._generate_script(gds_name)
        script_path = workdir / f"{self.component.name}_setup.lsf"
        script_path.write_text(script)

        self._artifacts = SetupArtifacts(
            native=script,
            files={"gds": workdir / gds_name, "lsf": script_path},
            summary={
                "n_ports": len(self.component.ports),
                "n_modes": len(self.spec.modes),
                "workdir": str(workdir),
            },
        )
        return self._artifacts

    def _generate_script(self, gds_name: str) -> str:
        """The complete FDTD setup as .lsf text (legacy-solver semantics)."""
        s = self.spec
        tech = self._tech_dict()
        center, span = self.domain()
        um = 1e-6
        L: list[str] = ["newproject;", "clear;"]

        # ---- layer builder (geometry) ----
        L += [
            "addlayerbuilder;",
            f'setnamed("layer group", "x", {center[0] * um});',
            f'setnamed("layer group", "y", {center[1] * um});',
            'setnamed("layer group", "z", 0);',
            f'setnamed("layer group", "x span", {(span[0] + 2 * s.buffer) * um});',
            f'setnamed("layer group", "y span", {(span[1] + 2 * s.buffer) * um});',
            'setnamed("layer group", "gds position reference", "Centered at custom coordinates");',
            f'setnamed("layer group", "gds center x", {-center[0] * um});',
            f'setnamed("layer group", "gds center y", {-center[1] * um});',
            f"loadgdsfile({_q(gds_name)});",
        ]

        substrate = tech["substrate"][0]
        L += [
            'addlayer("substrate");',
            f'setlayer("substrate", "start position", {substrate["z_base"] * um});',
            f'setlayer("substrate", "thickness", {-abs(substrate["z_span"]) * um});',
            f'setlayer("substrate", "background material", {_q(substrate["material"]["lum_db"]["model"])});',
            'setlayer("substrate", "layer number", "");',
        ]
        superstrate = tech["superstrate"][0]
        L += [
            'addlayer("superstrate");',
            f'setlayer("superstrate", "start position", {superstrate["z_base"] * um});',
            f'setlayer("superstrate", "thickness", {abs(superstrate["z_span"]) * um});',
            f'setlayer("superstrate", "background material", {_q(superstrate["material"]["lum_db"]["model"])});',
            'setlayer("superstrate", "layer number", "");',
        ]

        # device layers: only those actually present in the component
        present = {tuple(st.layer) for st in self.component.structures if st.role == "device"}
        for idx, d in enumerate(tech["device"]):
            if tuple(d["layer"]) not in present:
                continue
            lname = f"device_{idx}"
            spec_str = f"{d['layer'][0]}:{d['layer'][1]}"
            L += [
                f"addlayer({_q(lname)});",
                f'setlayer({_q(lname)}, "start position", {d["z_base"] * um});',
                f'setlayer({_q(lname)}, "thickness", {abs(d["z_span"]) * um});',
                f'setlayer({_q(lname)}, "layer number", {_q(spec_str)});',
                f'setlayer({_q(lname)}, "sidewall angle", {d["sidewall_angle"]});',
                f'setlayer({_q(lname)}, "pattern material", {_q(d["material"]["lum_db"]["model"])});',
            ]

        # ---- FDTD region ----
        L += [
            "addfdtd;",
            f'setnamed("FDTD", "x", {center[0] * um});',
            f'setnamed("FDTD", "y", {center[1] * um});',
            f'setnamed("FDTD", "z", {center[2] * um});',
            f'setnamed("FDTD", "x span", {span[0] * um});',
            f'setnamed("FDTD", "y span", {span[1] * um});',
            f'setnamed("FDTD", "z span", {span[2] * um});',
        ]

        # boundaries + symmetry (legacy mapping incl. Symmetric/Anti-Symmetric)
        for axis, bc, sym in zip("xyz", s.boundary, s.symmetry, strict=True):
            L.append(f'setnamed("FDTD", "{axis} max bc", {_q(bc)});')
            if sym == 0:
                L.append(f'setnamed("FDTD", "{axis} min bc", {_q(bc)});')
            elif sym == 1:
                L.append(f'setnamed("FDTD", "{axis} min bc", "Symmetric");')
            else:
                L.append(f'setnamed("FDTD", "{axis} min bc", "Anti-Symmetric");')

        # mesh accuracy: nearest option to requested cells/wavelength (legacy map)
        mesh_options = {1: 6, 2: 10, 3: 14, 4: 18, 5: 22, 6: 26, 7: 30, 8: 34}
        option = min(mesh_options, key=lambda o: abs(s.mesh - mesh_options[o]))
        L.append(f'setnamed("FDTD", "mesh accuracy", {option});')

        # simulation time (shared group-index-aware calculation)
        c_m_s = 299792458.0
        time_span = s.run_time_factor * (max(span) * um) / (c_m_s / 4.5)
        L.append(f'setnamed("FDTD", "simulation time", {time_span});')

        # wavelength range
        L += [
            f'setglobalsource("wavelength start", {s.wavelength_start * um});',
            f'setglobalsource("wavelength stop", {s.wavelength_end * um});',
            f'setglobalmonitor("frequency points", {s.wavelength_points});',
        ]

        # ---- ports ----
        modes_lsf = "[" + ";".join(str(m) for m in s.modes) + "]"
        for e in self.injection_plan():
            L += [
                "addport;",
                f'set("name", {_q(e["name"])});',
                f'set("x", {e["position"][0] * um});',
                f'set("y", {e["position"][1] * um});',
                f'set("z", {e["position"][2] * um});',
                f'set("direction", {_q(e["direction"].capitalize())});',
                f'set("injection axis", "{e["axis"]}-axis");',
            ]
            if e["axis"] == "x":
                L.append(f'set("y span", {s.width_ports * um});')
            else:
                L.append(f'set("x span", {s.width_ports * um});')
            L += [
                f'set("z span", {s.depth_ports * um});',
                'set("mode selection", "user select");',
                f"updateportmodes({modes_lsf});",
                f'set("number of field profile samples", {s.mode_freq_pts});',
            ]

        # ---- field monitors ----
        for axis in s.field_monitors:
            normal = {"x": "2D X-normal", "y": "2D Y-normal", "z": "2D Z-normal"}[axis]
            L += [
                f'addprofile; set("name", "profile_{axis}"); set("monitor type", {_q(normal)});',
                f'set("x", {center[0] * um}); set("y", {center[1] * um}); set("z", {center[2] * um});',
            ]
            if axis != "x":
                L.append(f'set("x span", {span[0] * um});')
            if axis != "y":
                L.append(f'set("y span", {span[1] * um});')
            if axis != "z":
                L.append(f'set("z span", {span[2] * um});')

        # ---- s-parameter sweep (fresh project: no stale entries to remove) ----
        L += [
            "addsweep(3);",
            'setsweep("s-parameter sweep", "name", "sparams");',
            'setsweep("sparams", "Excite all ports", 0);',
        ]
        active = {p.name for p in self.component.ports}  # all ports = full matrix
        for e in self.injection_plan():
            for m in s.modes:
                L += [
                    "entry = struct;",  # fresh struct per entry (clear() on an
                    # undefined variable errors in LSF — found by live bisect)
                    f"entry.Port = {_q(e['name'])};",
                    f"entry.Mode = {_q(f'mode {m}')};",
                    f"entry.Active = {1 if e['name'] in active else 0};",
                    'addsweepparameter("sparams", entry);',
                ]

        L.append(f"save({_q(self.component.name)});")
        return "\n".join(L) + "\n"

    def estimate(self) -> ResourceEstimate:
        if self._artifacts is None:
            self.build()
        _, span = self.domain()
        cells_per_um = self.spec.mesh / self.spec.wavelength_center_um
        cells = int(span[0] * span[1] * span[2] * cells_per_um**3)
        return ResourceEstimate(
            grid_cells=cells,
            n_simulations=len(self.component.ports) * len(self.spec.modes),
            cost_hint="local compute; requires a Lumerical FDTD license",
        )

    def run(self) -> SMatrix:
        """Open a lumapi session (license checkout), replay the script, sweep."""
        reason = probe_lumapi()
        if reason:
            raise RuntimeError(f"cannot run: {reason}")
        import lumapi

        if self._artifacts is None:
            self.build()
        workdir = self._artifacts.summary["workdir"]

        fdtd = lumapi.FDTD(hide=True)
        try:
            fdtd.cd(workdir)
            fdtd.eval(self._artifacts.native)

            # device type: 2025 syntax first, 2024 fallback (finding F7)
            device = "GPU" if self.gpu else "CPU"
            try:
                fdtd.setresource("FDTD", 1, "device type", device)
            except Exception:
                fdtd.setresource("FDTD", device, 1)

            fdtd.runsweep("sparams")
            dat_path = os.path.join(workdir, f"{self.component.name}.dat")
            fdtd.exportsweep("sparams", dat_path)
        finally:
            fdtd.close()

        from ..sparams import process_dat

        spar = process_dat(dat_path, verbose=False)
        sm = spar.to_smatrix(name=self.component.name)
        return sm
