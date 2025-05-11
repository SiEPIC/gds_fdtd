"""
gds_fdtd.core unit tests.

Unit tests for gds_fdtd.core.
@author: Mustafa Hammood, 2025
"""

import sys
import pytest
import logging
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from gds_fdtd.core import (
    is_point_inside_polygon,
    layout,
    calculate_polygon_extension,
    port,
    structure,
    component,
    initialize_ports_z,
    sparam,
    s_parameters,
    parse_yaml_tech,
)
from gds_fdtd.core import region as Region  # <-- alias

# Configure logging
logger = logging.getLogger(__name__)

# ---------- Fixtures for is_point_inside_polygon ----------


@pytest.fixture
def response_is_point_inside_polygon() -> (
    Dict[str, Union[List[float], List[List[float]]]]
):
    """
    Sample pytest fixture that returns a point and polygon.

    Returns:
        Dict containing a point inside a square polygon.
    """
    logger.debug("Creating fixture with point inside polygon")
    point = [0, 0]
    polygon = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    return {"point": point, "polygon": polygon}


@pytest.fixture
def response_is_point_outside_polygon() -> (
    Dict[str, Union[List[float], List[List[float]]]]
):
    """
    Sample pytest fixture that returns a point outside a polygon.

    Returns:
        Dict containing a point outside a square polygon.
    """
    logger.debug("Creating fixture with point outside polygon")
    point = [2, 2]
    polygon = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    return {"point": point, "polygon": polygon}


def test_is_point_inside_polygon(
    response_is_point_inside_polygon: Dict[str, Any],
) -> None:
    """
    Verify that a point inside a polygon is correctly identified.

    Args:
        response_is_point_inside_polygon: Fixture with point and polygon data.
    """
    point = response_is_point_inside_polygon["point"]
    polygon = response_is_point_inside_polygon["polygon"]
    logger.info(f"Testing point {point} inside polygon with {len(polygon)} vertices")
    assert is_point_inside_polygon(point, polygon) is True


def test_is_point_outside_polygon(
    response_is_point_outside_polygon: Dict[str, Any],
) -> None:
    """
    Verify that a point outside a polygon is correctly identified.

    Args:
        response_is_point_outside_polygon: Fixture with point and polygon data.
    """
    point = response_is_point_outside_polygon["point"]
    polygon = response_is_point_outside_polygon["polygon"]
    logger.info(f"Testing point {point} outside polygon with {len(polygon)} vertices")
    assert is_point_inside_polygon(point, polygon) is False


# ---------- Fixture & tests for layout.dbu property ----------


@pytest.fixture
def dummy_layout() -> layout:
    """
    Create a lightweight dummy layout object without KLayout dependency.

    Returns:
        A layout instance with mocked underlying objects.
    """
    logger.debug("Creating dummy layout with database unit 1e-3")
    DummyLy = SimpleNamespace(dbu=1e-3)
    DummyCell = SimpleNamespace()
    return layout("demo", DummyLy, DummyCell)


def test_layout_dbu_property(dummy_layout: layout) -> None:
    """
    Verify that layout.dbu returns the underlying ly.dbu value.

    Args:
        dummy_layout: A fixture providing a layout instance.
    """
    logger.info(f"Testing layout.dbu property, expected value: 1e-3")
    assert dummy_layout.dbu == pytest.approx(1e-3)


# ---------- Tests for calculate_polygon_extension ----------


@pytest.mark.parametrize(
    "direction,expected",
    [
        (
            0,
            [[0, 1], [4, 1], [4, -1], [0, -1]],
        ),
        (
            180,
            [[0, 1], [-4, 1], [-4, -1], [0, -1]],
        ),
        (
            90,
            [[-1, 0], [-1, 4], [1, 4], [1, 0]],
        ),
        (
            270,
            [[-1, 0], [-1, -4], [1, -4], [1, 0]],
        ),
    ],
)
def test_calculate_polygon_extension_default_buffer(
    direction: int, expected: List[List[float]]
) -> None:
    """
    Test polygon extension calculation with default buffer value.

    Args:
        direction: Port direction in degrees (0, 90, 180, 270).
        expected: Expected polygon coordinates.
    """
    center: List[float] = [0, 0]
    width: float = 2
    logger.info(f"Testing polygon extension with direction {direction}°")
    result = calculate_polygon_extension(center, width, direction)
    assert result == expected


def test_calculate_polygon_extension_custom_buffer() -> None:
    """Test polygon extension calculation with a custom buffer value."""
    center: List[float] = [5, 5]
    width: float = 3
    direction: int = 0
    buffer: float = 2
    logger.info(f"Testing polygon extension with custom buffer {buffer}")

    poly = calculate_polygon_extension(center, width, direction, buffer)
    expected = [
        [5, 5 + 1.5],
        [5 + 2, 5 + 1.5],
        [5 + 2, 5 - 1.5],
        [5, 5 - 1.5],
    ]
    assert poly == expected


# ---------- Fixtures & tests for port class ----------


@pytest.fixture
def dummy_port() -> port:
    """
    Create a port instance for testing.

    Returns:
        A port instance with predefined parameters.
    """
    logger.debug("Creating dummy port 'port42' at [10, 20, 30]")
    return port("port42", [10, 20, 30], 4, 90)  # name, center(x,y,z), width, dir


def test_port_idx_extraction(dummy_port: port) -> None:
    """
    Test port index extraction from name.

    Args:
        dummy_port: A fixture providing a port instance.
    """
    logger.info("Testing port index extraction from name 'port42'")
    # name "port42" → reversed digits "24" → idx = 24
    assert dummy_port.idx == 24


def test_port_polygon_extension_delegation(dummy_port: port) -> None:
    """
    Test that port.polygon_extension delegates to calculate_polygon_extension.

    Args:
        dummy_port: A fixture providing a port instance.
    """
    logger.info("Testing port polygon extension delegation")
    expected = calculate_polygon_extension(
        dummy_port.center, dummy_port.width, dummy_port.direction
    )
    assert dummy_port.polygon_extension() == expected


# ---------- structure ----------


@pytest.fixture
def dummy_structure() -> structure:
    """
    Create a structure instance for testing.

    Returns:
        A structure instance with predefined parameters.
    """
    logger.debug("Creating dummy structure 'wg'")
    poly: List[List[float]] = [[0, 0], [10, 0], [10, 5], [0, 5]]
    return structure("wg", poly, z_base=0, z_span=2, material="SiO2")


def test_structure_attrs(dummy_structure: structure) -> None:
    """
    Test structure attribute access.

    Args:
        dummy_structure: A fixture providing a structure instance.
    """
    logger.info("Testing structure attributes")
    s = dummy_structure
    assert s.name == "wg"
    assert s.z_base == 0
    assert s.z_span == 2
    assert s.material == "SiO2"
    assert s.sidewall_angle == 90.0
    # polygon equivalence without relying on ordering mutations
    assert set(map(tuple, s.polygon)) == {(0, 0), (10, 0), (10, 5), (0, 5)}

# ---------- port ---------------------------------------------------------


@pytest.fixture
def sample_port() -> port:
    """
    Create a sample port for testing.

    Returns:
        A port instance with predefined parameters.
    """
    logger.debug("Creating sample port 'port42'")
    # name,          center(x,y,z), width, direction
    return port("port42", [10, 20, 30], 4.0, 90)


# ---------- Coordinate properties -------------------------------------------


def test_port_coordinates(sample_port: port) -> None:
    """
    Test port coordinate access properties.

    Args:
        sample_port: A fixture providing a port instance.
    """
    logger.info("Testing port coordinate properties")
    assert sample_port.x == 10
    assert sample_port.y == 20
    assert sample_port.z == 30


# ---------- Index extraction -------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("port42", 24),  # digits reversed → "24"
        ("p7", 7),
        ("abc123", 321),  # digits reversed → "321"
        ("noDigits0", 0),
    ],
)
def test_port_idx(name: str, expected: int) -> None:
    """
    Test port index extraction from various name patterns.

    Args:
        name: Port name to test.
        expected: Expected extracted index.
    """
    logger.info(f"Testing port index extraction from name '{name}'")
    p = port(name, [0, 0, 0], 1.0, 0)
    assert p.idx == expected


# ---------- Polygon extension delegation ------------------------------------


@pytest.mark.parametrize(
    "center,width,direction,buffer",
    [
        ([0, 0, 0], 2, 0, 4.0),
        ([0, 0, 0], 2, 180, 4.0),
        ([0, 0, 0], 2, 90, 4.0),
        ([0, 0, 0], 2, 270, 4.0),
        ([5, 5, 0], 3, 0, 2.0),
    ],
)
def test_polygon_extension_consistency(
    center: List[float], width: float, direction: int, buffer: float
) -> None:
    """
    Test consistency between port.polygon_extension and calculate_polygon_extension.

    Args:
        center: Port center coordinates.
        width: Port width.
        direction: Port direction in degrees.
        buffer: Extension buffer distance.
    """
    logger.info(f"Testing polygon extension consistency with direction {direction}°")
    p = port("test0", center, width, direction)
    assert p.polygon_extension(buffer) == calculate_polygon_extension(
        center, width, direction, buffer
    )


# ---------- Validation: unsupported direction -------------------------------


def test_invalid_direction_raises_value_error() -> None:
    """Test that creating a port with invalid direction raises ValueError."""
    logger.info("Testing invalid port direction validation")
    with pytest.raises(ValueError, match="Invalid direction"):
        port("bad", [0, 0, 0], 1.0, 45)  # 45° is not allowed


# ---------- Fixtures ---------------------------------------------------------


@pytest.fixture
def default_structure() -> structure:
    """
    Create a default structure for testing.

    Returns:
        A structure instance with predefined parameters.
    """
    logger.debug("Creating default structure")
    # simple 10 µm × 5 µm rectangle, vertical sidewalls
    poly: List[List[float]] = [[0, 0], [10, 0], [10, 5], [0, 5]]
    return structure("wg", poly, z_base=0.0, z_span=2.0, material="SiO2")


# ---------- Attribute checks -------------------------------------------------


def test_structure_attributes(default_structure: structure) -> None:
    """
    Test structure attribute access.

    Args:
        default_structure: A fixture providing a structure instance.
    """
    logger.info("Testing structure attributes")
    s = default_structure
    assert s.name == "wg"
    assert s.polygon == [[0, 0], [10, 0], [10, 5], [0, 5]]
    assert (s.z_base, s.z_span) == (0.0, 2.0)
    assert s.material == "SiO2"
    assert s.sidewall_angle == 90.0


def test_structure_custom_sidewall() -> None:
    """Test structure creation with custom sidewall angle."""
    logger.info("Testing structure with custom sidewall angle")
    poly: List[List[float]] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    s = structure(
        "taper", poly, z_base=1.0, z_span=0.5, material="Si", sidewall_angle=80.0
    )
    assert s.sidewall_angle == 80.0


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def make_square(
    side: float = 4,
    centre: Tuple[float, float] = (0, 0),
    z_base: float = 0.0,
    z_span: float = 2.0,
    mat: str = "Si",
) -> structure:
    """
    Return **one** structure describing a square cross‑section.

    Args:
        side: Side length of the square.
        centre: (x, y) coordinates of the square center.
        z_base: Base z-coordinate.
        z_span: Vertical height/thickness.
        mat: Material identifier.

    Returns:
        A structure instance representing a square.
    """
    logger.debug(f"Creating square structure with side={side}, centre={centre}")
    cx, cy = centre
    h = side / 2
    poly: List[List[float]] = [
        [cx - h, cy - h],
        [cx + h, cy - h],
        [cx + h, cy + h],
        [cx - h, cy + h],
    ]
    return structure("core", poly, z_base, z_span, mat)


# We need a double‑nested list because the production code
# treats `[struct]` as a *region* (cladding hack).
@pytest.fixture
def region() -> List[List[structure]]:
    """
    Create a region (nested list of structures) for testing.

    Returns:
        A region containing a single square structure.
    """
    logger.debug("Creating region with single square structure")
    return [[make_square()]]  # ⟨region⟩ -> list[list[structure]]


@pytest.fixture
def p_inside() -> port:
    """
    Create a port located inside the test square.

    Returns:
        A port instance at the center of the square.
    """
    logger.debug("Creating port inside square")
    return port("in0", [0, 0, 0], width=0.5, direction=0)  # centre of square


@pytest.fixture
def p_outside() -> port:
    """
    Create a port located outside the test square.

    Returns:
        A port instance outside the square.
    """
    logger.debug("Creating port outside square")
    return port("out1", [10, 10, 0], width=0.5, direction=180)


# ------------------------------------------------------------------------------
# initialize_ports_z
# ------------------------------------------------------------------------------


def test_initialize_ports_sets_attributes(
    region: List[List[structure]], p_inside: port
) -> None:
    """
    Test that initialize_ports_z correctly sets port attributes.

    Args:
        region: A region containing structures.
        p_inside: A port inside the region.
    """
    logger.info("Testing initialize_ports_z sets port attributes")
    initialize_ports_z([p_inside], region)

    # square is 2 µm thick starting at z=0 ⇒ centre z = 1 µm
    assert p_inside.center[2] == pytest.approx(1.0)
    assert p_inside.height == 2.0
    assert p_inside.material == "Si"


def test_initialize_ports_warns_for_unmatched(
    region: List[List[structure]], p_outside: port, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test that initialize_ports_z warns for ports outside any structure.

    Args:
        region: A region containing structures.
        p_outside: A port outside the region.
        caplog: Pytest fixture to capture log output.
    """
    logger.info("Testing initialize_ports_z warning for unmatched port")
    with caplog.at_level(logging.WARNING):
        initialize_ports_z([p_outside], region)

    assert p_outside.height is None
    assert "Cannot find height for port out1" in caplog.text


# ------------------------------------------------------------------------------
# component.__init__  –  ensure it triggers initialize_ports_z
# ------------------------------------------------------------------------------


def test_component_init_calls_initialize(
    monkeypatch: pytest.MonkeyPatch, region: List[List[structure]], p_inside: port
) -> None:
    """
    Test that component.__init__ calls initialize_ports_z.

    Args:
        monkeypatch: Pytest fixture for patching.
        region: A region containing structures.
        p_inside: A port inside the region.
    """
    logger.info("Testing component.__init__ calls initialize_ports_z")
    # Spy on initialise‑call count
    calls: Dict[str, int] = {"n": 0}

    def spy(ports: List[port], structures: List[List[structure]]) -> None:
        calls["n"] += 1
        initialize_ports_z(ports, structures)  # keep behaviour

    monkeypatch.setattr("gds_fdtd.core.initialize_ports_z", spy)

    comp = component("demo", structures=region, ports=[p_inside], bounds=[])

    assert calls["n"] == 1
    assert p_inside.material == "Si"
    assert comp.name == "demo"


# ------------------------------------------------------------------------------
# component.export_gds  –  pure I/O path, mocked klayout
# ------------------------------------------------------------------------------


# Dummy klayout stand‑ins
class _DummyPoint:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


class _DummyPolygon(list):
    pass


class _DummyCell:
    def __init__(self) -> None:
        # shapes(layer) should return an object that has .insert(...)
        self.shapes = lambda *_: self

    def insert(self, _poly: Any) -> None:
        pass


class _DummyLayout:
    def __init__(self) -> None:
        self.dbu: Optional[float] = None
        self._cells: List[_DummyCell] = []

    def create_cell(self, _name: str) -> _DummyCell:
        cell = _DummyCell()
        self._cells.append(cell)
        return cell

    def layer(self, *_: Any) -> int:
        return 0  # arbitrary layer handle

    def write(self, path: str) -> None:
        # emulate writing by touching the file
        logger.debug(f"Writing dummy GDS file to {path}")
        open(path, "wb").close()


class _DummyLayerInfo:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def test_export_gds_creates_file(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    region: List[List[structure]],
) -> None:
    """
    Test that component.export_gds creates a GDS file.

    Args:
        tmp_path: Pytest fixture for temporary directory.
        monkeypatch: Pytest fixture for patching.
        region: A region containing structures.
    """
    logger.info("Testing component.export_gds creates file")
    # Inject dummy klayout module tree
    kl_db = SimpleNamespace(
        Layout=_DummyLayout,
        LayerInfo=_DummyLayerInfo,
        Polygon=_DummyPolygon,
        Point=_DummyPoint,
    )
    sys.modules["klayout"] = SimpleNamespace(db=kl_db)
    sys.modules["klayout.db"] = kl_db

    comp = component(
        "export_test",
        structures=region,
        ports=[port("p0", [0, 0, 0], 1.0, 0)],
        bounds=[],
    )

    outfile = tmp_path / "export_test.gds"
    comp.export_gds(export_dir=tmp_path)

    assert outfile.exists() and outfile.stat().st_size == 0  # dummy file produced


# ----------------------------------------------------------------------
# region - fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def square():
    verts = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    return verts, 0.0, 2.0


@pytest.fixture
def offset_rect():
    verts = [[0, 8], [10, 8], [10, 13], [0, 13]]
    return verts, 0.5, 1.0


# ----------------------------------------------------------------------
# region - helpers
# ----------------------------------------------------------------------


def _check_geom(reg, xmin, xmax, ymin, ymax):
    assert reg.x_min == xmin
    assert reg.x_max == xmax
    assert reg.y_min == ymin
    assert reg.y_max == ymax
    assert reg.x_span == pytest.approx(xmax - xmin)
    assert reg.y_span == pytest.approx(ymax - ymin)
    assert reg.x_center == pytest.approx((xmin + xmax) / 2)
    assert reg.y_center == pytest.approx((ymin + ymax) / 2)


# ----------------------------------------------------------------------
# region - tests
# ----------------------------------------------------------------------


def test_square_region(square):
    verts, zc, zs = square
    reg = Region(verts, zc, zs)
    _check_geom(reg, -2, 2, -2, 2)
    assert reg.z_center == 0.0
    assert reg.z_span == 2.0


def test_offset_rectangle(offset_rect):
    verts, zc, zs = offset_rect
    reg = Region(verts, zc, zs)
    _check_geom(reg, 0, 10, 8, 13)
    assert reg.z_center == 0.5
    assert reg.z_span == 1.0


@pytest.mark.parametrize(
    "verts,x_span,y_span",
    [
        ([[-1, 0], [1, 0], [1, 1], [-1, 1]], 2, 1),
        ([[0, 0], [4, 0], [4, 4], [0, 4]], 4, 4),
        ([[2, 2], [3, 2], [3, 3], [2, 3]], 1, 1),
    ],
)
def test_span_properties(verts, x_span, y_span):
    reg = Region(verts, z_center=0.0, z_span=1.0)
    assert reg.x_span == pytest.approx(x_span)
    assert reg.y_span == pytest.approx(y_span)


# ----------------------------------------------------------------------
# fixtures ‑‑ sparam / s_parameters
# ----------------------------------------------------------------------


@pytest.fixture
def sample_freq() -> np.ndarray:
    """1 THz span centred at 200 THz (≈1.5 µm)."""
    return np.linspace(195e12, 205e12, 5)


@pytest.fixture
def sp11(sample_freq: np.ndarray) -> sparam:
    """S11 reflection example (port 1 → port 1)."""
    s_vals = np.full_like(sample_freq, 0.01)  # flat −40 dB
    return sparam(1, 1, 0, 0, sample_freq, s_vals)


@pytest.fixture
def sp21(sample_freq: np.ndarray) -> sparam:
    """S21 transmission example (port 1 → port 2)."""
    s_vals = np.full_like(sample_freq, 0.9)  # −0.9 dB
    return sparam(1, 2, 0, 0, sample_freq, s_vals)


@pytest.fixture
def sblock(sp11: sparam, sp21: sparam) -> s_parameters:
    """Aggregate S‑parameters."""
    sp = s_parameters()
    sp.add_param(sp11)
    sp.add_param(sp21)
    return sp


# ----------------------------------------------------------------------
# sparam ----------------------------------------------------------------


def test_sparam_label(sp11: sparam, sp21: sparam) -> None:
    """Label string follows S<out><in>_idx<modeout><modein> pattern."""
    assert sp11.label == "S11_idx00"
    assert sp21.label == "S21_idx00"


def test_sparam_plot_returns_fig_ax(sp11: sparam) -> None:
    """plot() returns a (figure, axes) tuple."""
    fig, ax = sp11.plot()
    assert fig is not None and ax is not None


# ----------------------------------------------------------------------
# s_parameters ----------------------------------------------------------


def test_S_mapping_keys(sblock: s_parameters) -> None:
    """s_parameters.S maps labels → objects."""
    labels = set(sblock.S.keys())
    assert labels == {"S11_idx00", "S21_idx00"}


@pytest.mark.parametrize(
    "mode_in,mode_out,expected_labels",
    [
        (0, 0, {"S11_idx00", "S21_idx00"}),  # both modes 0→0
        (1, 0, set()),  # no such mode pair
    ],
)
def test_entries_in_mode(
    sblock: s_parameters, mode_in: int, mode_out: int, expected_labels: set
) -> None:
    entries = sblock.entries_in_mode(mode_in, mode_out)
    assert {e.label for e in entries} == expected_labels


@pytest.mark.parametrize(
    "idx_in,idx_out,expected_labels",
    [
        (1, 1, {"S11_idx00"}),
        (1, 2, {"S21_idx00"}),
        (2, 1, set()),
    ],
)
def test_entries_in_ports(
    sblock: s_parameters,
    idx_in: int,
    idx_out: int,
    expected_labels: set,
) -> None:
    entries = sblock.entries_in_ports(idx_in=idx_in, idx_out=idx_out)
    assert {e.label for e in entries} == expected_labels


def test_sblock_plot_returns_fig_ax(sblock: s_parameters) -> None:
    fig, ax = sblock.plot()
    assert fig is not None and ax is not None


# ----------------------------------------------------------------------
# parse_yaml_tech -------------------------------------------------------
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _write_yaml(tmp: Path, name: str, content: str) -> str:
    """Write *content* to tmp/<name>.yaml and return its str path."""
    path = tmp / f"{name}.yaml"
    path.write_text(content.strip())
    return str(path)


# ----------------------------------------------------------------------
# YAML fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def tidy3d_yaml(tmp_path: Path) -> str:
    """Return path to a tidy3d‑style tech YAML file."""
    return _write_yaml(
        tmp_path,
        "tech_tidy3d",
        """
technology:
  name: "EBeam"
  substrate:
    z_base: 0.0
    z_span: -2
    material:
      tidy3d_db:
        nk: 1.48
  superstrate:
    z_base: 0.0
    z_span: 3
    material:
      tidy3d_db:
        nk: 1.48
  pinrec:
    - layer: [1, 10]
  devrec:
    - layer: [68, 0]
  device:
    - layer: [1, 0]
      z_base: 0.0
      z_span: 0.22
      material:
        tidy3d_db:
          model: [cSi, Li1993_293K]
      sidewall_angle: 85
    - layer: [4, 0]
      z_base: 0.3
      z_span: 0.4
      material:
        tidy3d_db:
          model: [Si3N4, Luke2015PMLStable]
      sidewall_angle: 90
""",
    )


@pytest.fixture
def lumerical_yaml(tmp_path: Path) -> str:
    """Return path to a lumerical‑style tech YAML file."""
    return _write_yaml(
        tmp_path,
        "tech_lum",
        """
technology:
  name: "EBeam"
  substrate:
    z_base: 0.0
    z_span: -2
    material:
      lum_db:
        model: SiO2 (Glass) - Palik
  superstrate:
    z_base: 0.0
    z_span: 3
    material:
      lum_db:
        model: SiO2 (Glass) - Palik
  pinrec:
    - layer: [1, 10]
  devrec:
    - layer: [68, 0]
  device:
    - layer: [1, 0]
      z_base: 0.0
      z_span: 0.22
      material:
        lum_db:
          model: Si (Silicon) - Palik
      sidewall_angle: 85
    - layer: [4, 0]
      z_base: 0.3
      z_span: 0.4
      material:
        lum_db:
          model: Si3N4 (Silicon Nitride) - Luke
      sidewall_angle: 83
""",
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def _common_assertions(parsed: Dict[str, Any]) -> None:
    """Checks common to both formats."""
    assert parsed["name"] == "EBeam"
    assert parsed["substrate"][0]["z_span"] == -2
    assert parsed["superstrate"][0]["z_span"] == 3
    # first device entry exists
    assert parsed["device"][0]["layer"] == [1, 0]


def test_parse_yaml_tidy3d(tidy3d_yaml: str) -> None:
    parsed = parse_yaml_tech(tidy3d_yaml)
    _common_assertions(parsed)
    # material dict carries tidy3d_db key
    assert "tidy3d_db" in parsed["device"][0]["material"]


def test_parse_yaml_lumerical(lumerical_yaml: str) -> None:
    parsed = parse_yaml_tech(lumerical_yaml)
    _common_assertions(parsed)
    # material dict carries lum_db key
    assert "lum_db" in parsed["device"][0]["material"]


# ----------------------------------------------------------------------
# Entry point to run directly
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
