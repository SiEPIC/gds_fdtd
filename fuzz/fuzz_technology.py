"""Fuzz target: technology-YAML parsing/validation.

Feeds arbitrary bytes to ``Technology.from_yaml``. Malformed input must fail
with a validation error, never with an unexpected crash (IndexError deep in
the parser, RecursionError, segfault in a C dependency, hang, ...).

Run locally:  python fuzz/fuzz_technology.py -atheris_runs=20000
"""

import os
import sys
import tempfile

import atheris

with atheris.instrument_imports():
    import yaml

    from gds_fdtd.technology import Technology

# Exceptions a well-behaved parser is allowed to raise on garbage input.
EXPECTED = (ValueError, KeyError, TypeError, yaml.YAMLError, UnicodeError, OSError)


def test_one_input(data: bytes) -> None:
    with tempfile.NamedTemporaryFile("wb", suffix=".yaml", delete=False) as fh:
        fh.write(data)
        path = fh.name
    try:
        Technology.from_yaml(path)
    except EXPECTED:
        pass
    finally:
        os.unlink(path)


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
