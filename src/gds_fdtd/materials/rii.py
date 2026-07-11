"""
refractiveindex.info database support.

Reads material pages from a LOCAL copy of the refractiveindex.info database
(https://github.com/polyanskiy/refractiveindex.info-database) — the DB is
plain YAML, so a small reader avoids a wrapper dependency. Offline-first by
design: nothing here ever downloads; CI and docs use a committed page fixture.

Database directory resolution order:
1. explicit ``db_dir`` argument,
2. ``GDS_FDTD_RII_DB`` environment variable,
3. ``~/.cache/gds_fdtd/rii`` (where a user may clone/copy the database's
   ``data`` folder).

Page path convention: ``<db_dir>/<shelf>/<book>/<page>.yml``.

Supported data blocks (the overwhelmingly common ones):
- ``tabulated nk`` / ``tabulated n`` / ``tabulated k``
- ``formula 1`` (Sellmeier) and ``formula 2`` (Sellmeier-2)

Other formula types raise a clear error naming the page.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


def _default_db_dir() -> Path:
    env = os.environ.get("GDS_FDTD_RII_DB")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "gds_fdtd" / "rii"


@dataclass
class RiiMaterial:
    """Tabulated optical constants for one refractiveindex.info page."""

    shelf: str
    book: str
    page: str
    wavelength_um: np.ndarray  # ascending
    n: np.ndarray
    k: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.k is None:
            self.k = np.zeros_like(self.n)

    @property
    def wavelength_range_um(self) -> tuple[float, float]:
        return float(self.wavelength_um[0]), float(self.wavelength_um[-1])

    def _check_range(self, wavelength_um) -> None:
        w = np.atleast_1d(np.asarray(wavelength_um, dtype=float))
        lo, hi = self.wavelength_range_um
        if w.min() < lo or w.max() > hi:
            raise ValueError(
                f"wavelength {w.min():.4g}-{w.max():.4g} um outside the tabulated "
                f"range [{lo:.4g}, {hi:.4g}] um of {self.shelf}/{self.book}/{self.page}"
            )

    def n_at(self, wavelength_um) -> np.ndarray | float:
        """Interpolated refractive index n at the given wavelength(s) in um."""
        self._check_range(wavelength_um)
        out = np.interp(np.asarray(wavelength_um, dtype=float), self.wavelength_um, self.n)
        return float(out) if np.ndim(wavelength_um) == 0 else out

    def k_at(self, wavelength_um) -> np.ndarray | float:
        """Interpolated extinction coefficient k at the given wavelength(s) in um."""
        self._check_range(wavelength_um)
        out = np.interp(np.asarray(wavelength_um, dtype=float), self.wavelength_um, self.k)
        return float(out) if np.ndim(wavelength_um) == 0 else out

    def nk_at(self, wavelength_um) -> complex | np.ndarray:
        """Complex refractive index n + ik at the given wavelength(s) in um."""
        n = self.n_at(wavelength_um)
        k = self.k_at(wavelength_um)
        return n + 1j * k

    def to_tidy3d_medium(self, wavelength_um=None, max_num_poles: int = 5):
        """Fit these optical constants to a dispersive tidy3d medium.

        Returns a ``tidy3d`` pole-residue medium that carries the full complex,
        wavelength-dependent ``n + ik`` — plug it straight into a tidy3d
        simulation to run with the refractiveindex.info model itself, not a
        constant. Needs tidy3d (``pip install gds_fdtd[tidy3d]``).

        Args:
            wavelength_um: wavelengths to fit over (defaults to the material's
                whole tabulated range).
            max_num_poles: fit complexity ceiling.
        """
        try:
            from tidy3d.plugins.dispersion import FastDispersionFitter
        except ImportError as e:  # pragma: no cover - optional engine
            raise ImportError("to_tidy3d_medium needs tidy3d: pip install gds_fdtd[tidy3d]") from e
        wl = self.wavelength_um if wavelength_um is None else np.asarray(wavelength_um, dtype=float)
        n = np.asarray(self.n_at(wl), dtype=float)
        k = np.asarray(self.k_at(wl), dtype=float)
        medium, _rms = FastDispersionFitter(
            wvl_um=np.asarray(wl, dtype=float), n_data=n, k_data=k
        ).fit(max_num_poles=max_num_poles)
        return medium


def _parse_tabulated(block: dict, columns: int) -> tuple[np.ndarray, ...]:
    rows = [
        [float(x) for x in line.split()]
        for line in str(block["data"]).strip().splitlines()
        if line.strip()
    ]
    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] < columns:
        raise ValueError(f"tabulated block has {arr.shape[1]} columns; expected >= {columns}")
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return tuple(arr[:, i] for i in range(columns))


def _sellmeier(coeffs: list[float], wavelength_um: np.ndarray, formula: int) -> np.ndarray:
    """refractiveindex.info formula 1 (Sellmeier) and 2 (Sellmeier-2)."""
    w2 = wavelength_um**2
    rhs = np.full_like(wavelength_um, 1.0 + coeffs[0])
    pairs = coeffs[1:]
    for b, c in zip(pairs[0::2], pairs[1::2], strict=False):
        if formula == 1:
            rhs = rhs + b * w2 / (w2 - c**2)
        else:  # formula 2: C is already squared in the database convention
            rhs = rhs + b * w2 / (w2 - c)
    return np.sqrt(rhs)


def load_rii_material(
    shelf: str, book: str, page: str, db_dir: str | Path | None = None
) -> RiiMaterial:
    """Load one refractiveindex.info page from the local database copy."""
    base = Path(db_dir) if db_dir is not None else _default_db_dir()
    path = base / shelf / book / f"{page}.yml"
    if not path.exists():
        raise FileNotFoundError(
            f"refractiveindex.info page not found: {path}. Set GDS_FDTD_RII_DB (or pass "
            "db_dir=) to a local copy of the database's 'data' directory "
            "(https://github.com/polyanskiy/refractiveindex.info-database)."
        )

    with open(path) as f:
        doc = yaml.safe_load(f)

    data_blocks = doc.get("DATA")
    if not data_blocks:
        raise ValueError(f"{path}: no DATA section")

    wavelength = n = k = None
    for block in data_blocks:
        btype = str(block.get("type", "")).strip()
        if btype == "tabulated nk":
            wavelength, n, k = _parse_tabulated(block, 3)
        elif btype == "tabulated n":
            w_n, n_only = _parse_tabulated(block, 2)
            wavelength = w_n if wavelength is None else wavelength
            n = np.interp(wavelength, w_n, n_only)
        elif btype == "tabulated k":
            w_k, k_only = _parse_tabulated(block, 2)
            if wavelength is None:
                wavelength = w_k
            k = np.interp(wavelength, w_k, k_only)
        elif btype in ("formula 1", "formula 2"):
            formula = int(btype.split()[1])
            lo, hi = (float(x) for x in str(block["wavelength_range"]).split())
            coeffs = [float(c) for c in str(block["coefficients"]).split()]
            wavelength = np.linspace(lo, hi, 512)
            n = _sellmeier(coeffs, wavelength, formula)
        elif btype.startswith("formula"):
            raise ValueError(
                f"{path}: dispersion '{btype}' is not supported yet "
                "(supported: tabulated nk/n/k, formula 1, formula 2)"
            )

    if wavelength is None or n is None:
        raise ValueError(f"{path}: no usable refractive-index data block found")

    return RiiMaterial(shelf=shelf, book=book, page=page, wavelength_um=wavelength, n=n, k=k)
