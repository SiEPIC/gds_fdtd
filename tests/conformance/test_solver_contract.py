"""Solver conformance suite (WP3.1b / WP7.1).

Every adapter registered in gds_fdtd.solvers must pass these tests. They run
against FakeSolver today; as WP3.1c/d land, the ported adapters join the
`all_solver_classes` list automatically via the registry.

Contract highlights (Part 4 of MODERNIZATION_PLAN.md):
- constructor is cheap and PURE: no files created, no sockets;
- validate() returns list[str];
- build() is offline, deterministic, and returns SetupArtifacts;
- estimate() works offline;
- run() is the only money/license-spending method (not exercised here).
"""

from __future__ import annotations

import pathlib
import socket

import numpy as np
import pytest

from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.smatrix import SMatrix
from gds_fdtd.solvers import (
    ResourceEstimate,
    SetupArtifacts,
    Solver,
    SolverCapabilities,
    available_solvers,
    get_solver,
    register_solver,
)
from gds_fdtd.spec import SimulationSpec
from gds_fdtd.technology import Technology

TESTS_DIR = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# FakeSolver: the reference implementation of the contract
# ---------------------------------------------------------------------------


@register_solver
class FakeSolver(Solver):
    """In-memory adapter used by the conformance suite, the convergence
    framework tests (WP5.5), and the execution-backend tests (WP7.3)."""

    name = "fake"
    capabilities = SolverCapabilities(
        tier="full",
        execution="local",
        supports_dispersion=True,
        supports_sidewall_angle=True,
        supports_multimode=True,
        supports_gpu=False,
        cost_model="free",
    )

    def validate(self) -> list[str]:
        problems = []
        if not self.component.ports:
            problems.append("component has no ports")
        return problems

    def build(self) -> SetupArtifacts:
        # deterministic "scene": the injection plan + domain, no side effects
        center, span = self.domain()
        self._artifacts = SetupArtifacts(
            native={"ports": self.injection_plan(), "center": center, "span": span},
            summary={"n_ports": len(self.component.ports), "mesh": self.spec.mesh},
        )
        return self._artifacts

    def estimate(self) -> ResourceEstimate:
        _, span = self.domain()
        cells = int(np.prod([s * self.spec.mesh / self.spec.wavelength_center_um for s in span]))
        return ResourceEstimate(grid_cells=cells, n_simulations=len(self.component.ports))

    def run(self) -> SMatrix:
        # perfect reciprocal splitter-ish placeholder: identity-lossless thru
        f = self.frequencies_hz()
        names = [p.name for p in self.component.ports]
        entries = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                thru = np.full(f.size, 1 / np.sqrt(len(names) - 1) + 0j)
                entries.append((a, b, 1, 1, f, thru))
                entries.append((b, a, 1, 1, f, thru))
        return SMatrix.from_entries(entries, name=self.component.name, port_names=names)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_job(cls):
    """Build a valid (component, technology, keepalive, extra_kwargs) job for
    the given solver class. Solvers that resolve materials need the tech whose
    entries they understand; beamz v1 additionally needs the source gdsfactory
    component; FakeSolver accepts anything.
    """
    if cls.name == "beamz":
        gf = pytest.importorskip("gdsfactory")
        gf.gpdk.PDK.activate()
        from gds_fdtd.layout.gdsfactory import from_gdsfactory

        # AGNOSTIC setup (owner directive): the unified tech carries neutral
        # nk entries and from_gdsfactory attaches the source component, so
        # beamz needs NO solver-specific kwargs.
        tech_dict = Technology.from_yaml(str(TESTS_DIR / "tech_unified.yaml")).to_solver_dict()
        gf_c = gf.components.straight(length=5)
        comp = from_gdsfactory(gf_c, tech_dict)
        return comp, tech_dict, None, {}

    tech_file = "tech_tidy3d.yaml" if cls.name == "tidy3d" else "tech_lumerical.yaml"
    tech_dict = Technology.from_yaml(str(TESTS_DIR / tech_file)).to_solver_dict()
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech_dict)
    technology = tech_dict if cls.name != "fake" else None
    return comp, technology, layout, {}


@pytest.fixture(scope="module")
def component():
    tech = Technology.from_yaml(str(TESTS_DIR / "tech_lumerical.yaml"))
    cell, layout = load_cell(str(TESTS_DIR / "si_sin_escalator.gds"))
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp
    del layout


def all_solver_classes():
    return [get_solver(name) for name in sorted(available_solvers())]


@pytest.fixture(params=all_solver_classes(), ids=lambda c: c.name)
def solver(request):
    cls = request.param
    probe = getattr(cls, "probe_available", None)
    if probe is not None and probe():
        pytest.skip(f"{cls.name}: {probe()}")
    comp, technology, layout, extra = _make_job(cls)
    s = cls(comp, technology=technology, spec=SimulationSpec(), **extra)
    s._keepalive = layout  # klayout layout must outlive the cell
    return s


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


def test_declares_name_and_capabilities(solver):
    assert isinstance(solver.name, str) and solver.name
    assert isinstance(solver.capabilities, SolverCapabilities)


def test_constructor_is_pure(tmp_path, monkeypatch):
    """No files, no sockets during construction."""
    monkeypatch.chdir(tmp_path)

    def _no_net(*a, **k):
        raise AssertionError("constructor opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    for cls in all_solver_classes():
        probe = getattr(cls, "probe_available", None)
        if probe is not None and probe():
            continue
        comp, technology, layout, extra = _make_job(cls)
        cls(comp, technology=technology, **extra)
        assert list(tmp_path.iterdir()) == [], f"{cls.name} constructor wrote files"
        del layout


def test_validate_returns_list_of_str(solver):
    problems = solver.validate()
    assert isinstance(problems, list)
    assert all(isinstance(p, str) for p in problems)
    assert problems == []  # the escalator fixture is a valid job


def test_build_is_offline_and_deterministic(solver, monkeypatch):
    def _no_net(*a, **k):
        raise AssertionError("build() opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    a1 = solver.build()
    a2 = solver.build()
    assert isinstance(a1, SetupArtifacts)
    # determinism is judged on the summary (always) and on native when it is
    # text — rich native objects (e.g. beamz grids) embed memory addresses
    assert a1.summary == a2.summary, "build() summary is not deterministic"
    if isinstance(a1.native, str | bytes):
        assert a1.native == a2.native, "build() native scene is not deterministic"
    assert isinstance(a1.summary, dict)


def test_estimate_is_offline(solver, monkeypatch):
    def _no_net(*a, **k):
        raise AssertionError("estimate() opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    est = solver.estimate()
    assert isinstance(est, ResourceEstimate)


def test_describe_mentions_component_and_ports(solver):
    text = solver.describe()
    assert solver.component.name in text
    for p in solver.component.ports:
        assert p.name in text


def test_injection_plan_sorted_and_shaped(solver):
    plan = solver.injection_plan()
    assert len(plan) == len(solver.component.ports)
    names = [e["name"] for e in plan]
    assert names == sorted(names, key=lambda n: int("".join(c for c in n if c.isdigit())))
    for e in plan:
        assert e["axis"] in ("x", "y") and e["direction"] in ("forward", "backward")
        assert len(e["position"]) == 3 and len(e["size"]) == 3
        assert e["size"][0 if e["axis"] == "x" else 1] == 0.0


def test_frequencies_ascending(solver):
    f = solver.frequencies_hz()
    assert f.size == solver.spec.wavelength_points
    assert np.all(np.diff(f) > 0)


def test_domain_covers_bounds(solver):
    center, span = solver.domain()
    b = solver.component.bounds
    assert center[0] == pytest.approx(b.x_center) and center[1] == pytest.approx(b.y_center)
    assert span[0] > b.x_span and span[1] > b.y_span


# ---------------------------------------------------------------------------
# registry + FakeSolver end-to-end
# ---------------------------------------------------------------------------


def test_registry_roundtrip():
    assert "fake" in available_solvers()
    assert get_solver("fake") is FakeSolver
    with pytest.raises(KeyError, match="No solver named"):
        get_solver("bogus")


def test_fake_solver_runs_to_valid_smatrix(component):
    sm = FakeSolver(component, technology=None).run()
    assert isinstance(sm, SMatrix)
    assert sm.port_names == [p.name for p in component.ports]
    assert sm.is_reciprocal()
    assert sm.is_passive()


def test_entry_point_discovery():
    """The installed package advertises its adapters via entry points (WP3.1e)."""
    from importlib.metadata import entry_points

    eps = {ep.name for ep in entry_points(group="gds_fdtd.solvers")}
    assert {"tidy3d", "lumerical"} <= eps
    avail = available_solvers()
    assert "tidy3d" in avail and "lumerical" in avail
