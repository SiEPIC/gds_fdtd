"""The canonical SMatrix: load, check physics, export to every format.

The .dat file here is a REAL tidy3d cloud result (recorded 2026-07-07) of the
si_sin escalator test fixture — no solver or credentials needed to run this.
"""

import os

from gds_fdtd.smatrix import SMatrix

if __name__ == "__main__":
    here = os.path.dirname(__file__)

    # Lumerical INTERCONNECT .dat -> SMatrix (works with any solver's output)
    sm = SMatrix.from_dat(os.path.join(here, "si_sin_escalator.dat"), name="escalator")
    print(f"ports={sm.port_names}  modes={sm.n_modes}  freqs={sm.f.size}")
    print(f"wavelengths [um]: {sm.wavelength_um.min():.3f} - {sm.wavelength_um.max():.3f}")

    # access any path by port name or trailing-digit id, 1-based mode ids
    s21_db = sm.magnitude_db(out=2, in_=1)
    print(f"S21: {s21_db.min():.2f} .. {s21_db.max():.2f} dB")

    # built-in physics checks (NaN-aware for partial matrices)
    print("reciprocal:", sm.is_reciprocal(atol=0.05))
    print("passive:", sm.is_passive(atol=0.02))

    # exports: Touchstone (industry standard), HDF5, .dat
    sm.to_touchstone(os.path.join(here, "escalator.s2p"))  # scikit-rf readable
    sm.to_hdf5(os.path.join(here, "escalator.h5"))
    print("wrote escalator.s2p / escalator.h5")

    # plotting (matplotlib, lazy import)
    from gds_fdtd.plotting import plot_smatrix

    fig, ax = plot_smatrix(sm, kind="db")
    fig.savefig(os.path.join(here, "escalator_sparams.png"), dpi=150)
    print("wrote escalator_sparams.png")
