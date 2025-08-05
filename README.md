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
- **Multimode/Dual Polarization Simulations:** Set up simulations that support multimode or dual polarization configurations for advanced device analysis.

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
```