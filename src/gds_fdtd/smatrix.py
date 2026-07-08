"""
gds_fdtd simulation toolbox.

The canonical S-matrix container (WP2.4a). One representation for every solver:

- ``f``: frequency grid in Hz, shape (F,), ascending.
- ``s``: complex ndarray, shape (F, P, P, M, M) indexed as
  ``s[f, out_port, in_port, out_mode, in_mode]``. Mode indices are 0-based
  internally; the public accessors take the package-wide 1-based mode ids.
- ``port_names``: list of P port names; row/column order follows this list.
- Unmeasured entries (partial matrices) are NaN.

I/O: ``.npz`` always (numpy built-in); ``.h5`` when ``h5py`` is installed.
Checks: reciprocity, passivity, per-excitation power balance.
Plotting lives in gds_fdtd.plotting (WP2.4d); Lumerical ``.dat`` and
Touchstone interop arrive in WP2.4b/c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C_M_S = 299792458.0  # speed of light in m/s


@dataclass
class SMatrix:
    """Multi-port, multi-mode scattering matrix on a frequency grid."""

    f: np.ndarray
    s: np.ndarray
    port_names: list[str]
    name: str = "smatrix"
    _port_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=float)
        self.s = np.asarray(self.s, dtype=complex)
        self.port_names = [str(n) for n in self.port_names]
        P = len(self.port_names)
        if self.f.ndim != 1:
            raise ValueError(f"f must be 1-D; got shape {self.f.shape}")
        if self.s.ndim != 5 or self.s.shape[:3] != (self.f.size, P, P):
            raise ValueError(
                f"s must have shape (F, P, P, M, M) = ({self.f.size}, {P}, {P}, M, M); "
                f"got {self.s.shape}"
            )
        if self.s.shape[3] != self.s.shape[4]:
            raise ValueError(f"mode axes must be square; got {self.s.shape[3:]} ")
        if len(set(self.port_names)) != P:
            raise ValueError(f"duplicate port names: {self.port_names}")
        if self.f.size > 1 and not np.all(np.diff(self.f) > 0):
            raise ValueError("f must be strictly ascending")
        self._port_index = {n: i for i, n in enumerate(self.port_names)}

    # ---------------- basic properties ----------------

    @property
    def n_ports(self) -> int:
        return len(self.port_names)

    @property
    def n_modes(self) -> int:
        return self.s.shape[3]

    @property
    def wavelength_um(self) -> np.ndarray:
        """Free-space wavelength grid in micrometers (descending when f ascends)."""
        return C_M_S / self.f * 1e6

    # ---------------- construction ----------------

    @classmethod
    def from_entries(
        cls,
        entries,
        name: str = "smatrix",
        port_names: list[str] | None = None,
    ) -> SMatrix:
        """Build from an iterable of per-path entries.

        Each entry: ``(in_port, out_port, mode_in, mode_out, f, s_complex)``
        with 1-based mode ids, ``f`` in Hz, ``s_complex`` the complex
        transmission array. All entries must share the same frequency grid.
        Missing paths stay NaN.
        """
        entries = list(entries)
        if not entries:
            raise ValueError("no entries given")

        names = list(port_names) if port_names else []
        max_mode = 1
        f_ref = None
        for in_p, out_p, m_in, m_out, f, _s in entries:
            for p in (str(in_p), str(out_p)):
                if p not in names:
                    names.append(p)
            max_mode = max(max_mode, int(m_in), int(m_out))
            f = np.asarray(f, dtype=float)
            if f_ref is None:
                f_ref = f
            elif f.shape != f_ref.shape or not np.allclose(f, f_ref):
                raise ValueError("all entries must share one frequency grid")

        order = np.argsort(f_ref)
        P, M, F = len(names), max_mode, f_ref.size
        s = np.full((F, P, P, M, M), np.nan + 0j, dtype=complex)
        for in_p, out_p, m_in, m_out, _f, s_c in entries:
            i, o = names.index(str(in_p)), names.index(str(out_p))
            s[:, o, i, int(m_out) - 1, int(m_in) - 1] = np.asarray(s_c, dtype=complex)[order]

        return cls(f=f_ref[order], s=s, port_names=names, name=name)

    # ---------------- access ----------------

    def _pidx(self, port) -> int:
        if isinstance(port, int) and port not in self._port_index:
            # accept a trailing-digit integer id (opt1 -> 1) as a convenience
            for name, i in self._port_index.items():
                digits = "".join(ch for ch in name if ch.isdigit())
                if digits and int(digits) == port:
                    return i
            raise KeyError(f"no port with id {port}; ports: {self.port_names}")
        try:
            return self._port_index[str(port)]
        except KeyError:
            raise KeyError(f"unknown port {port!r}; ports: {self.port_names}") from None

    def sel(self, out, in_, mode_out: int = 1, mode_in: int = 1) -> np.ndarray:
        """S(out <- in) for the given 1-based mode ids; complex array (F,)."""
        return self.s[:, self._pidx(out), self._pidx(in_), mode_out - 1, mode_in - 1]

    def magnitude_db(self, out, in_, mode_out: int = 1, mode_in: int = 1) -> np.ndarray:
        """|S|^2 in dB for one path."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10 * np.log10(np.abs(self.sel(out, in_, mode_out, mode_in)) ** 2)

    # ---------------- physics checks ----------------

    def _flat(self) -> np.ndarray:
        """View as (F, P*M, P*M) with port-major flattening (p, m)."""
        F, P, _, M, _ = self.s.shape
        return self.s.transpose(0, 1, 3, 2, 4).reshape(F, P * M, P * M)

    def is_reciprocal(self, atol: float = 1e-6) -> bool:
        """S == S^T within atol (NaN-aware: only mutually measured pairs count)."""
        m = self._flat()
        d = m - m.transpose(0, 2, 1)
        return bool(np.nanmax(np.abs(np.where(np.isnan(d), 0, d))) <= atol)

    def is_passive(self, atol: float = 1e-6) -> bool:
        """No excitation column may output more power than it received."""
        balance = self.power_balance()
        finite = balance[np.isfinite(balance)]
        return bool(finite.size == 0 or np.max(finite) <= 1.0 + atol)

    def power_balance(self) -> np.ndarray:
        """Sum over outputs of |S|^2 per (F, in_port*mode) excitation; NaN-skipped."""
        m = np.abs(self._flat()) ** 2
        return np.nansum(np.where(np.isnan(m), np.nan, m), axis=1) * np.where(
            np.all(np.isnan(m), axis=1), np.nan, 1.0
        )

    # ---------------- I/O ----------------

    def to_dat(self, path: str) -> str:
        """Write to Lumerical INTERCONNECT .dat (via the WP1.6 entry-driven writer).

        Port names are mapped to numeric ids by their trailing digits (or by
        position when a name has none); NaN (unmeasured) paths are skipped.
        """
        from .sparams import sparameters, write_dat

        def _num(i: int) -> int:
            digits = "".join(ch for ch in self.port_names[i] if ch.isdigit())
            return int(digits) if digits else i + 1

        spar = sparameters(self.name)
        for i_out in range(self.n_ports):
            for i_in in range(self.n_ports):
                for m_out in range(self.n_modes):
                    for m_in in range(self.n_modes):
                        col = self.s[:, i_out, i_in, m_out, m_in]
                        if np.all(np.isnan(col)):
                            continue
                        spar.add_data(
                            in_port=f"port {_num(i_in)}",
                            out_port=f"port {_num(i_out)}",
                            mode_label=1,
                            in_modeid=m_in + 1,
                            out_modeid=m_out + 1,
                            data_type="transmission",
                            group_delay=0.0,
                            f=list(self.f),
                            s_mag=list(np.abs(col)),
                            s_phase=list(np.angle(col)),
                        )
        return write_dat(spar, path)

    @classmethod
    def from_dat(cls, path: str, name: str | None = None) -> SMatrix:
        """Read a Lumerical INTERCONNECT .dat into an SMatrix."""
        from .sparams import process_dat

        spar = process_dat(path, name=name, verbose=False)
        return spar.to_smatrix(name=name)

    def to_npz(self, path: str) -> str:
        np.savez_compressed(
            path, f=self.f, s=self.s, port_names=np.array(self.port_names), name=self.name
        )
        return path

    @classmethod
    def from_npz(cls, path: str) -> SMatrix:
        with np.load(path, allow_pickle=False) as d:
            return cls(
                f=d["f"],
                s=d["s"],
                port_names=[str(x) for x in d["port_names"]],
                name=str(d["name"]),
            )

    def to_hdf5(self, path: str) -> str:
        h5py = _require_h5py()
        with h5py.File(path, "w") as h:
            h.create_dataset("f", data=self.f)
            h.create_dataset("s", data=self.s)
            h.create_dataset("port_names", data=[n.encode() for n in self.port_names])
            h.attrs["name"] = self.name
            h.attrs["format_version"] = 1
        return path

    @classmethod
    def from_hdf5(cls, path: str) -> SMatrix:
        h5py = _require_h5py()
        with h5py.File(path, "r") as h:
            return cls(
                f=h["f"][:],
                s=h["s"][:],
                port_names=[b.decode() for b in h["port_names"][:]],
                name=str(h.attrs.get("name", "smatrix")),
            )


def _require_h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for SMatrix HDF5 I/O; install it (pip install h5py) "
            "or use to_npz/from_npz."
        ) from e
    return h5py
