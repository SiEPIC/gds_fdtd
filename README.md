# gds_fdtd

![alternative text](/docs/logo.png)

[![CI](https://github.com/SiEPIC/gds_fdtd/actions/workflows/ci.yml/badge.svg)](https://github.com/SiEPIC/gds_fdtd/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/siepic/gds_fdtd/branch/main/graph/badge.svg)](https://codecov.io/gh/siepic/gds_fdtd)
[![docs](https://github.com/SiEPIC/gds_fdtd/actions/workflows/build_docs.yml/badge.svg)](https://siepic.github.io/gds_fdtd/)
[![PyPI](https://img.shields.io/pypi/v/gds_fdtd)](https://pypi.org/project/gds-fdtd/)
[![Python](https://img.shields.io/pypi/pyversions/gds_fdtd)](https://pypi.org/project/gds-fdtd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**gds_fdtd** is a minimal Python module to assist in setting up FDTD simulations for planar nanophotonic devices using FDTD solvers such as Tidy3D.

## Features

- **Automated FDTD Setup:** Easily set up Lumerical and Tidy3D simulations for devices designed in GDS.
- **Integration with SiEPIC:** Generate FDTD simulations directly from components defined in [SiEPIC](https://github.com/SiEPIC/SiEPIC-Tools) EDA and it's associated PDKs (e.g., [SiEPIC-EBeam-PDK](https://github.com/SiEPIC/SiEPIC_EBeam_PDK)).
- **Integration with gdsfactory:** Generate Tidy3D simulations directly from [gdsfactory](https://github.com/gdsfactory/gdsfactory) designs by identifying ports and simulation regions from an input technology stack.
- **S-Parameter Extraction:** Automatically generate and export S-parameters of your photonic devices in standard formats.
- **Multimode/Dual Polarization Simulations:** Set up simulations that support multimode or dual polarization configurations for device analysis.

## Installation

You can install `gds_fdtd` using the following options:

### Quick install (PyPI)

```bash
pip install gds-fdtd
```

### Option: Basic Installation from source

To install the core functionality of `gds_fdtd`, clone the repository and install using `pip`:

```bash
git clone git@github.com:mustafacc/gds_fdtd.git
cd gds_fdtd
pip install -e .
```

### Option: Development Installation

For contributing to the development or if you need testing utilities, install with the dev dependencies:

```bash
git clone git@github.com:mustafacc/gds_fdtd.git
cd gds_fdtd
pip install -e .[dev]
```

This will install additional tools like `pytest` and `coverage` for testing.

### Editable + dev tools

```bash
pip install -e .[dev]
```

### Optional extras

| extra      | purpose                        | install command                             |
|------------|--------------------------------|---------------------------------------------|
| siepic     | [SiEPIC](https://github.com/SiEPIC/SiEPIC-Tools) EDA support            | `pip install -e .[siepic]`                  |
| tidy3d     | [Tidy3D](https://github.com/flexcompute/tidy3d) simulation support      | `pip install -e .[tidy3d]`                  |
| gdsfactory | [GDSFactory](https://github.com/gdsfactory/gdsfactory) EDA support         | `pip install -e .[gdsfactory]`              |
| prefab     | [PreFab](https://github.com/PreFab-Photonics/PreFab) lithography prediction support      | `pip install -e .[prefab]`                  |
| everything | dev tools + all plugins        | `pip install -e .[dev,tidy3d,gdsfactory,prefab,siepic]`   |

### Requirements

- Python ≥ 3.11
- Runtime deps: numpy, matplotlib, shapely, PyYAML, klayout


### Running tests

If you've installed the `dev` dependencies, you can run the test suite with:

```bash
pytest --cov=gds_fdtd tests
```

## Development

### Development Setup

```bash
git clone https://github.com/SiEPIC/gds_fdtd.git
cd gds_fdtd
pip install -e .[dev]        # or: uv sync --extra dev

# install the git hooks (uses the standard .pre-commit-config.yaml;
# prek is a fast drop-in for pre-commit)
uv tool install prek && prek install
```

Canonical dev tasks live in the [justfile](justfile):

```bash
just test        # tests with coverage
just lint        # ruff check + format check (what CI runs)
just fix         # auto-fix lint + formatting
just docs        # build documentation
just gate        # quick lint+test gate
```

### Versioning & Releases

The version is derived **from git tags** via `hatch-vcs` — there is nothing to bump and no
version string in the source. To release:

```bash
git tag v0.5.0
git push --tags
```

The `release.yml` workflow then verifies the tagged commit passed CI, builds and inspects the
package, publishes to PyPI via Trusted Publishing (with PEP 740 attestations), and creates a
GitHub Release with auto-generated notes (categorized by PR labels — see
`.github/release.yml`).