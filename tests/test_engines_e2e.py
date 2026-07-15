"""Real end-to-end engine tests on the free engine (WS2).

A tiny straight waveguide runs through beamz's FULL run() path — mode
injection, FDTD, modal DFT extraction — and must produce the physics a
straight guarantees: near-total transmission, deep reflection, passivity.
Marked slow (~1-2 min on a laptop CPU); the CI all-extras leg runs it.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def straight_job():
    pytest.importorskip("beamz")
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    from gds_fdtd.layout.gdsfactory import from_gdsfactory
    from gds_fdtd.spec import SimulationSpec
    from gds_fdtd.technology import Technology

    tech = Technology.from_yaml(str(TESTS_DIR / "tech_unified.yaml"))
    component = from_gdsfactory(gf.components.straight(length=2.0), tech)
    spec = SimulationSpec(
        wavelength_start=1.5,
        wavelength_end=1.6,
        wavelength_points=3,
        mesh=5,
        z_min=-0.6,
        z_max=0.8,
    )
    return component, tech, spec


@pytest.mark.slow
def test_beamz_straight_end_to_end(straight_job, tmp_path):
    """The full free-engine path: run(), physics checks, field, and cache."""
    from gds_fdtd.solvers import get_solver

    component, tech, spec = straight_job
    solver = get_solver("beamz")(component, tech, spec)
    assert solver.validate() == []

    est = solver.estimate()
    assert est.grid_cells > 0 and est.n_simulations == 2

    sm = solver.run_cached(tmp_path)  # cold: executes the real FDTD

    # physics of a short straight: near-0 dB through, deep reflection
    thru_db = sm.magnitude_db(out=2, in_=1)
    refl_db = sm.magnitude_db(out=1, in_=1)
    assert float(thru_db.max()) > -0.5, f"through path lossy: {thru_db}"
    assert float(refl_db.max()) < -15.0, f"reflection too high: {refl_db}"
    # mesh 5 is deliberately coarse: beamz's single-mode normalization
    # overshoots unity by up to ~10% there (see 06_convergence_and_caching).
    # Assert the overshoot stays BOUNDED - a normalization regression like the
    # F14 +40 dB bug would blow far past this.
    assert float(np.nanmax(sm.power_balance())) < 1.15
    assert sm.is_reciprocal(atol=5e-2)  # coarse mesh: engineering tolerance

    # field profile recorded and renderable in both scales
    fig, ax = solver.plot_fields(axis="z", scale="linear")
    assert ax.images or ax.collections
    fig2, ax2 = solver.plot_fields(axis="z", scale="db")
    assert ax2.images or ax2.collections

    # warm cache: identical result, no re-run
    solver2 = get_solver("beamz")(component, tech, spec)
    sm2 = solver2.run_cached(tmp_path)
    np.testing.assert_allclose(sm2.s, sm.s)
