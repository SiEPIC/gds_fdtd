"""Read saved magnitudes for plots without requiring generated binary archives.

These records deliberately do not reconstruct complex S-parameters or phase.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MagnitudeResults:
    def __init__(self, path: Path):
        self.record = json.loads(path.read_text())
        self.wavelength_um = np.asarray(self.record["wavelength_um"])
        self.f = 299792458 / (self.wavelength_um * 1e-6)
        self.port_names = self.record.get("port_names", ["opt1", "opt2"])
        self.n_ports = len(self.port_names)

    def magnitude_db(self, *, out: int, in_: int) -> np.ndarray:
        if "entries_db" in self.record:
            key = f"{self.port_names[out - 1]}<-{self.port_names[in_ - 1]}"
            return np.asarray(self.record["entries_db"][key])
        return np.asarray(self.record[f"s{out}{in_}_db"])
