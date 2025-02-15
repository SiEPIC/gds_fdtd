# gds_fdtd

![alternative text](/docs/banner.png)

![codecov](https://codecov.io/gh/mustafacc/gds_fdtd/branch/main/graph/badge.svg)
![build](https://github.com/mustafacc/gds_fdtd/actions/workflows/main.yml/badge.svg)

**gds_fdtd** is a minimal Python module to assist in setting up FDTD simulations for planar nanophotonic devices using FDTD solvers such as Tidy3D.

## Features

- **Automated FDTD Setup:** Easily set up Tidy3D simulations for devices designed in GDS.
- **Integration with gdsfactory:** Generate Tidy3D simulations directly from [gdsfactory](https://github.com/gdsfactory/gdsfactory) designs by identifying ports and simulation regions from an input technology stack.
- **S-Parameter Extraction:** Automatically generate and export S-parameters of your photonic devices in standard formats.
- **Multimode/Dual Polarization Simulations:** Set up simulations that support multimode or dual polarization configurations for advanced device analysis.

## Installation

You can install `gds_fdtd` using the following options:

### Option 1: Basic Installation

To install the core functionality of `gds_fdtd`, clone the repository and install using `pip`:

```bash
git clone git@github.com:mustafacc/gds_fdtd.git
cd gds_fdtd
pip install .
```

### Option 2: Development Installation

For contributing to the development or if you need testing utilities, install with the dev dependencies:

```bash
git clone git@github.com:mustafacc/gds_fdtd.git
cd gds_fdtd
pip install -e .[dev]
```

This will install additional tools like `pytest` and `coverage` for testing.

### Optional Dependencies

If your workflow includes specific dependencies such as gdsfactory or prefab, you can install those optional extras:

with [gdsfactory](https://github.com/gdsfactory/gdsfactory):

```bash
pip install -e .[gdsfactory]
```

Refer to [gdsfactory example](/examples/06_gdsfactory/06a_gf_bend.py) for usage.

with [prefab](https://github.com/PreFab-Photonics/PreFab):

```bash
pip install -e .[prefab]
```

Refer to [prefab example](/examples/08_prefab/08a_bragg_prefab.py) for usage.

If you want everything installed for both development and optional features, run the following command:

```bash
pip install -e .[dev,gdsfactory,prefab]
```


### Running tests

If you've installed the `dev` dependencies, you can run the test suite with:

```bash
pytest --cov=gds_fdtd tests
```
