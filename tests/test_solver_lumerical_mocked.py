"""Legacy Lumerical adapter, exercised end-to-setup with the lumapi mock.

The real engine is license-bound; everything BEFORE `runsweep` is
deterministic script generation against the lumapi API, which the recording
mock captures. This is the offline transcript-assertion strategy of WP7.1.3.
"""

from __future__ import annotations

import pathlib

import pytest

from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.lyprocessor import load_cell
from gds_fdtd.simprocessor import load_component_from_tech

from .mocks.lumapi import install

TESTS_DIR = pathlib.Path(__file__).parent

# these tests exercise the DEPRECATED fdtd_solver_lumerical on purpose
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture()
def mock_lumapi(monkeypatch):
    return install(monkeypatch)


@pytest.fixture()
def escalator():
    # the unified tech pairs with the EXAMPLES gds (SiN on [4,0]); the tests
    # escalator draws SiN on (1,5) - mismatching them reproduces finding F4
    tech = parse_yaml_tech(str(TESTS_DIR / "tech_unified.yaml"))
    cell, layout = load_cell(
        str(TESTS_DIR.parent / "examples" / "devices.gds"),
        top_cell="si_sin_escalator_te1550",
    )
    comp = load_component_from_tech(cell=cell, tech=tech)
    yield comp, tech
    del layout


def _make(mock_lumapi, escalator, tmp_path, **kwargs):
    from gds_fdtd.solver_lumerical import fdtd_solver_lumerical

    comp, tech = escalator
    defaults = {
        "component": comp,
        "tech": tech,
        "wavelength_points": 5,
        "mesh": 4,
        "z_min": -1.0,
        "z_max": 1.11,
        "working_dir": str(tmp_path),
    }
    defaults.update(kwargs)
    return fdtd_solver_lumerical(**defaults)


def test_setup_runs_offline_and_records(mock_lumapi, escalator, tmp_path):
    solver = _make(mock_lumapi, escalator, tmp_path)
    fdtd = solver.fdtd
    assert isinstance(fdtd, mock_lumapi)
    # working dir is set first, project saved last
    assert any("cd(" in line for line in fdtd.script)
    assert fdtd.calls_named("save"), "setup must save the .fsp project"
    # the GDS with port extensions was exported for the layer builder
    assert solver._gds_filepath and pathlib.Path(solver._gds_filepath).exists()


def test_setup_transcript_builds_all_layers(mock_lumapi, escalator, tmp_path):
    solver = _make(mock_lumapi, escalator, tmp_path)
    t = solver.fdtd.transcript
    # the layer builder script constructs substrate/superstrate + device layers
    assert 'addlayer("substrate")' in t and 'addlayer("superstrate")' in t
    assert t.count("addlayer(") >= 4  # background x2 + both device layers
    # both material models from the unified tech reach the script
    assert "Si (Silicon) - Palik" in t
    assert "Si3N4 (Silicon Nitride) - Luke" in t


def test_setup_configures_ports_and_monitors(mock_lumapi, escalator, tmp_path):
    solver = _make(mock_lumapi, escalator, tmp_path, field_monitors=["z"])
    names = [c[0] for c in solver.fdtd.calls]
    assert "addprofile" in names  # field monitor
    assert "setsweep" in names or "addsweep" in names  # s-parameter sweep
    # per-port configuration happened
    assert len(solver.fdtd.calls_named("setnamed")) > 5


def test_f6_empty_runsystemcheck_survives(mock_lumapi, escalator, tmp_path):
    """Lumerical 2025 returns {} from runsystemcheck (F6) — setup must not crash."""
    solver = _make(mock_lumapi, escalator, tmp_path)
    assert solver is not None  # constructor calls setup() -> get_resources()


def test_device_type_uses_v252_setresource_syntax(mock_lumapi, escalator, tmp_path):
    """F7: Lumerical 2025 wants setresource('FDTD', 1, 'device type', dev)."""
    solver = _make(mock_lumapi, escalator, tmp_path, gpu=True)
    solver._set_device_type("GPU")  # run() applies this before runsweep
    res = solver.fdtd.calls_named("setresource")
    assert res and res[0][1] == ("FDTD", 1, "device type", "GPU")


def test_setup_is_deterministic(mock_lumapi, escalator, tmp_path):
    a = _make(mock_lumapi, escalator, tmp_path / "a")
    b = _make(mock_lumapi, escalator, tmp_path / "b")

    def strip(s: str) -> str:
        return s.replace(str(tmp_path / "a"), "WD").replace(str(tmp_path / "b"), "WD")

    assert strip(a.fdtd.transcript) == strip(b.fdtd.transcript)


def test_run_completes_on_recorded_dat(mock_lumapi, escalator, tmp_path):
    """run() drives runsweep -> exportsweep -> process_dat; the mock's
    exportsweep writes a REAL recorded engine .dat, so the whole results
    pipeline executes offline (F7 fallback path included)."""
    solver = _make(mock_lumapi, escalator, tmp_path)
    solver.run()
    names = [c[0] for c in solver.fdtd.calls]
    assert "runsweep" in names and "exportsweep" in names
    spar = solver.sparameters
    assert len(spar.data) == 4
    thru = spar.S(in_port=1, out_port=2)
    import numpy as np

    assert 10 * np.log10(max(np.abs(thru.s_mag)) ** 2) > -1.0


def test_deprecation_warning(mock_lumapi, escalator, tmp_path):
    """fdtd_solver_lumerical warns on use; users should move to get_solver."""
    from gds_fdtd.solver_lumerical import fdtd_solver_lumerical

    comp, tech = escalator
    with pytest.warns(DeprecationWarning, match="deprecated since gds_fdtd 0.5"):
        fdtd_solver_lumerical(
            component=comp,
            tech=tech,
            wavelength_points=5,
            mesh=4,
            z_min=-1.0,
            z_max=1.11,
            working_dir=str(tmp_path),
        )
