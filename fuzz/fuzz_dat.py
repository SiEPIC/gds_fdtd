"""Fuzz target: Lumerical INTERCONNECT ``.dat`` S-parameter parsing.

Feeds arbitrary bytes to ``SMatrix.from_dat`` (the hand-written text parser
in gds_fdtd.sparams). Malformed input must fail with a validation error,
never an unhandled crash.

Run locally:  python fuzz/fuzz_dat.py -atheris_runs=20000
"""

import os
import sys
import tempfile

import atheris

with atheris.instrument_imports():
    from gds_fdtd.smatrix import SMatrix

EXPECTED = (ValueError, KeyError, IndexError, TypeError, UnicodeError, OSError, StopIteration)


def test_one_input(data: bytes) -> None:
    with tempfile.NamedTemporaryFile("wb", suffix=".dat", delete=False) as fh:
        fh.write(data)
        path = fh.name
    try:
        SMatrix.from_dat(path)
    except EXPECTED:
        pass
    finally:
        os.unlink(path)


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
