Architecture Overview
=====================

This page provides a short description of the main modules in
``gds_fdtd`` and how they interact to build FDTD simulations.

Core Modules
------------

``core``
    Core data structures and helper functions used throughout the
    package. It defines geometrical primitives, ports, components,
    and S-parameter utilities used by all solver backends.

``lyprocessor``
    Utilities for parsing and processing GDS files with
    :mod:`klayout`. These helpers convert layouts into the internal
    data structures defined in ``core``.

``simprocessor``
    Higher level routines to assemble complete simulations from a
    technology description and extracted layout information.

``sparams``
    S-parameter handling, analysis, and export functionality
    supporting multi-modal calculations and various output formats.

``logging_config``
    Comprehensive logging system providing detailed simulation
    tracking and debugging capabilities.

Solver Architecture
-------------------

``solver`` (Base Class)
    Abstract base class defining the common interface for all FDTD
    solvers. Provides standardized port handling, field monitoring,
    logging, and parameter validation.

``solver_tidy3d``
    Tidy3D cloud-based FDTD solver implementation using the official
    ComponentModeler plugin for accurate S-matrix calculation.
    Supports multi-modal simulations and enhanced field visualization.

``solver_lumerical``
    Lumerical FDTD solver implementation with GPU acceleration support,
    layer builder integration, and S-parameter sweep configuration.

Key Features
------------

- **Modular Design**: Object-oriented architecture with pluggable solver backends
- **Multi-Modal Support**: Full TE/TM polarization and mode conversion analysis
- **Comprehensive Logging**: Detailed simulation tracking with file output
- **Field Visualization**: Enhanced field monitor system with solver-specific plotting
- **Technology Integration**: YAML-based technology files for material and layer definitions
- **S-Parameter Analysis**: Advanced S-parameter calculation, validation, and export

The modular architecture allows easy extension to new FDTD solvers while
maintaining a consistent interface. The examples demonstrate how these modules
work together to create, run, and analyze electromagnetic simulations of
photonic devices.
