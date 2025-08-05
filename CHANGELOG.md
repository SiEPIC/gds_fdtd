# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-08-05

### Major Release - New Architecture

This release completely rewrites how the solvers work. Everything is now more modular and easier to extend.

### Added

#### New Solver System
- Base `fdtd_solver` class that both Tidy3D and Lumerical solvers inherit from
- `fdtd_solver_tidy3d` class that uses Tidy3D's ComponentModeler for S-parameters
- `fdtd_solver_lumerical` class that works with Lumerical FDTD (with GPU support)
- `fdtd_port` class so ports work the same way in both solvers
- Better field monitor system that lets each solver handle visualization differently

#### Multi-Modal Simulations
- Both TE and TM polarizations in the same simulation
- Tidy3D ComponentModeler integration for better S-parameter accuracy
- S-parameter extraction for all mode combinations
- PDL and mode conversion analysis tools

#### Logging System (`logging_config.py`)
- Logs everything that happens during simulations
- Writes detailed logs to files in the working directory
- Shows different amounts of detail in console vs. file logs
- Creates log files with timestamps and component names automatically

#### Better S-Parameter Handling (`sparams.py`)
- Tools to validate S-parameter results
- Export to multiple formats (.dat, JSON, Touchstone)
- Check energy conservation and reciprocity
- Calculate group delay and bandwidth

#### Documentation
- Complete documentation website using Sphinx
- Guides for how the solvers work and how to use them
- Setting up multi-modal simulations
- Working with S-parameters
- Creating technology files
- Troubleshooting common problems
- Automatic API documentation
- Deployed to GitHub Pages

### Changed

#### Breaking Changes
- Removed old `lum_tools.py` and `t3d_tools.py` modules
- All solvers now inherit from the base `fdtd_solver` class
- Import solvers from `gds_fdtd.solver_tidy3d` and `gds_fdtd.solver_lumerical` now
- Changed some method names to be consistent between solvers

#### How Things Work Now
- Both solvers use the same parameter names and structure
- Simulation domain size is calculated automatically from your component
- Each component gets its own subdirectory for output files
- Better error messages when parameters don't make sense

#### Tidy3D Solver Changes
- Now uses ComponentModeler instead of manual S-parameter calculation
- Better handling of cloud simulation submission and monitoring
- Field visualization uses Tidy3D's native plotting functions
- Supports all TE/TM mode combinations in one simulation

#### Lumerical Solver Changes
- Works with Lumerical 2024 GPU acceleration syntax
- Better integration with technology files for layer building
- S-parameter sweeps are set up automatically
- Shows estimated memory and computation requirements

### Fixed

- Port detection from GDS files works better now
- Material assignment from technology files is more reliable
- Fixed buffer calculations and port extensions
- S-parameter magnitudes and phases are calculated correctly
- Field monitors are placed exactly where they should be
- File organization works the same on different operating systems

### Migration Guide from v0.3.x to v0.4.0

#### Import Changes
```python
# Old way (v0.3.x)
from gds_fdtd.lum_tools import lumerical_fdtd
from gds_fdtd.t3d_tools import sim_tidy3d

# New way (v0.4.0)
from gds_fdtd.solver_lumerical import fdtd_solver_lumerical
from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d
```

#### Solver Initialization
```python
# Old way (v0.3.x)
sim = sim_tidy3d(in_port=component.ports[0], device=component)

# New way (v0.4.0)
solver = fdtd_solver_tidy3d(
    component=component,
    tech=technology,
    port_input=[component.ports[0]],
    # ... other parameters
)
```

#### Running Simulations
```python
# Old way (v0.3.x)
sim.upload()
sim.execute()

# New way (v0.4.0)
solver.run()  # Everything happens automatically
```

#### Accessing Results
```python
# Old way (v0.3.x)
# Results were accessed differently for each solver

# New way (v0.4.0)
# Same interface for both solvers
sparams = solver.sparameters
wavl = sparams.wavelength
s41 = sparams.S(in_port=1, out_port=4, in_modeid=1, out_modeid=1)
```

---

## [0.3.10] - 2024-05-13

### Added
- Buffer parameter for Lumerical solver

---

## [0.3.9] - 2024-05-11

### Changed
- Python requirements changed from 3.11 to 3.10

---

## [0.3.8] - 2024-05-11

### Fixed
- Path for halfring function

---

## [0.3.7] - 2024-05-10

### Changed
- Updated examples
- Updated unit tests
- Updated Update_Halfring_CML for ebeam_dc_halfring_straight

---

## [0.3.6] - 2024-05-08

### Changed
- Updated compact model library for ebeam_dc_halfright_straight PCell

---

## [0.2.0] - 2024-03-07

### Added
- from_gdsfactory: create simulation recipes with gdsfactory component instance
- Multi-polarization support and examples

### Fixed
- Plotting dimensions
- Source and monitor placement
- Setup file
- S-parameters prep
- Other minor improvements

---

## [0.1.0] - 2023-12-03

### Added
- Base usable version of the package 
