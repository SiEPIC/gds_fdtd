# gds_fdtd

![alternative text](/docs/logo.png)

![codecov](https://codecov.io/gh/siepic/gds_fdtd/branch/main/graph/badge.svg)
![build](https://github.com/siepic/gds_fdtd/actions/workflows/build_and_test.yml/badge.svg)
![docs](https://github.com/siepic/gds_fdtd/actions/workflows/build_docs.yml/badge.svg)
![jekyll](https://github.com/siepic/gds_fdtd/actions/workflows/jekyll-gh-pages.yml/badge.svg)
![pypi](https://github.com/siepic/gds_fdtd/actions/workflows/python-publish.yml/badge.svg)

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

- Python ≥ 3.10 (note: gdsfactory requires Python ≥ 3.11)  
- Runtime deps: numpy, matplotlib, shapely, PyYAML, klayout


### Running tests

If you've installed the `dev` dependencies, you can run the test suite with:

```bash
pytest --cov=gds_fdtd tests

## Development

### Version Management

This project uses `bump2version` to keep version numbers in sync across all files.

**Check current version:**
```bash
make check-version
```

**Bump version:**
```bash
# For bug fixes (0.4.0 -> 0.4.1)
make bump-patch

# For new features (0.4.0 -> 0.5.0)  
make bump-minor

# For breaking changes (0.4.0 -> 1.0.0)
make bump-major
```

**Quick release:**
```bash
# Does everything in one go
./scripts/release.sh [patch|minor|major]
```

This script will:
- Run tests and build docs
- Update version numbers everywhere
- Create a git tag
- Push everything to GitHub
- Trigger GitHub release and PyPI upload

### Available Make Commands

```bash
make help          # Show all available commands
make install       # Install package in development mode
make test          # Run tests with coverage
make docs          # Build documentation
make docs-serve    # Build and serve docs locally on port 8000
make clean         # Clean build artifacts
make release       # Build package for release
```

### Development Setup

For new contributors:

```bash
# Clone the repository
git clone https://github.com/SiEPIC/gds_fdtd.git
cd gds_fdtd

# Install development dependencies
pip install -e .[dev]

# Run tests
make test

# Build documentation
make docs
```

### Making a Release

1. Make sure you're on the main branch with all changes committed
2. Run the release script: `./scripts/release.sh [patch|minor|major]`
3. The script will automatically:
   - Run tests to make sure everything works
   - Build documentation
   - Bump version numbers in all files
   - Create and push a git tag
   - Trigger GitHub Actions to create a release and upload to PyPI

The GitHub Actions workflow will:
- Run tests again
- Build the package
- Create a GitHub release with changelog
- Upload to PyPI automatically
```