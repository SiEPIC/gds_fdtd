FDTD Solvers
============

The ``gds_fdtd`` package provides a modular FDTD solver architecture with support for multiple electromagnetic simulation backends. The solver system is built around an object-oriented design that allows for easy extension and maintenance.

Base Solver Class
-----------------

The :class:`~gds_fdtd.solver.fdtd_solver` serves as the abstract base class for all FDTD solver implementations. It provides common functionality and defines the interface that all solvers must implement.

Key Features:
- Standardized port representation via :class:`~gds_fdtd.solver.fdtd_port` objects
- Comprehensive logging system
- Modular field monitor system
- Automatic simulation domain calculation
- Parameter validation and error handling

.. code-block:: python

    from gds_fdtd.solver import fdtd_solver
    
    class MyCustomSolver(fdtd_solver):
        def setup(self):
            # Implementation specific setup
            pass
            
        def run(self):
            # Implementation specific simulation execution
            pass

Common Solver Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

All solvers share these common initialization parameters:

- ``component``: The photonic component to simulate
- ``tech``: Technology definition (materials, layers, etc.)
- ``wavelength_start/end``: Simulation wavelength range (μm)
- ``wavelength_points``: Number of frequency points
- ``mesh``: Mesh resolution (cells per wavelength)
- ``modes``: List of mode indices for multi-modal simulation
- ``field_monitors``: Field monitoring axes ('x', 'y', 'z')
- ``working_dir``: Directory for simulation files and logs

Tidy3D Solver
-------------

The :class:`~gds_fdtd.solver_tidy3d.fdtd_solver_tidy3d` provides integration with the Tidy3D cloud-based FDTD platform using the official ComponentModeler plugin.

Features:
- Official Tidy3D ComponentModeler integration for accurate S-matrix calculation
- Multi-modal S-parameter support (TE, TM, mode conversion)
- Cloud-based simulation execution
- Enhanced field visualization with Tidy3D-specific plotting
- Automatic source and monitor setup with proper normalization

.. code-block:: python

    from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d
    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.simprocessor import load_component_from_tech
    from gds_fdtd.lyprocessor import load_cell
    
    # Load component and technology
    tech = parse_yaml_tech("tech_tidy3d.yaml")
    cell, layout = load_cell("device.gds", "crossing_te1550")
    component = load_component_from_tech(cell, tech)
    
    # Create Tidy3D solver
    solver = fdtd_solver_tidy3d(
        component=component,
        tech=tech,
        wavelength_start=1.5,
        wavelength_end=1.6,
        wavelength_points=100,
        modes=[1, 2],  # TE and TM modes
        field_monitors=["z"],
        visualize=True
    )
    
    # Run simulation
    solver.run()

Tidy3D Specific Parameters:
- ``visualize``: Enable/disable visualization during setup and results

Lumerical Solver
----------------

The :class:`~gds_fdtd.solver_lumerical.fdtd_solver_lumerical` provides integration with Lumerical FDTD for commercial-grade electromagnetic simulations.

Features:
- Full Lumerical FDTD integration
- GPU acceleration support
- Layer builder integration with technology files
- S-parameter sweep configuration
- Field monitor support

.. code-block:: python

    from gds_fdtd.solver_lumerical import fdtd_solver_lumerical
    
    # Create Lumerical solver
    solver = fdtd_solver_lumerical(
        component=component,
        tech=tech,
        wavelength_start=1.5,
        wavelength_end=1.6,
        wavelength_points=100,
        modes=[1, 2],
        gpu=True,  # Enable GPU acceleration
        boundary=["PML", "PML", "Metal"],
        symmetry=[0, 0, 1]  # Mirror symmetry in z
    )
    
    # Run simulation
    solver.run()

Lumerical Specific Parameters:
- ``gpu``: Enable GPU acceleration (Lumerical 2024 syntax)
- ``boundary``: Boundary conditions for each axis
- ``symmetry``: Symmetry conditions (0=none, 1=symmetric, -1=anti-symmetric)

Field Monitor System
--------------------

Both solvers feature a modular field monitor system for electromagnetic field visualization:

Base Field Monitor
^^^^^^^^^^^^^^^^^^

The :class:`~gds_fdtd.solver.fdtd_field_monitor` provides the base functionality:

.. code-block:: python

    # Access field monitors
    for monitor in solver.field_monitors_objs:
        print(f"Monitor: {monitor.name} ({monitor.monitor_type})")
        
        # Check if data is available
        if monitor.has_data():
            # Visualize fields
            monitor.visualize(freq=freq, field_component='E')
            
            # Get monitor information
            print(monitor.get_field_info())

Tidy3D Field Monitors
^^^^^^^^^^^^^^^^^^^^^

Tidy3D solvers use specialized field monitors with enhanced visualization:

.. code-block:: python

    # Visualize all field monitors
    solver.visualize_field_monitors()
    
    # Access individual monitors
    z_monitor = solver.get_field_monitor("z_field")
    if z_monitor and z_monitor.has_data():
        z_monitor.visualize(field_component='E')

Logging System
--------------

All solvers feature comprehensive logging to files in the working directory:

.. code-block:: python

    # Logging is automatic - check working directory for log files
    import os
    log_files = [f for f in os.listdir(solver.working_dir) if f.endswith('.log')]
    print(f"Log files: {log_files}")
    
    # Manual logging
    solver.logger.info("Custom log message")

Log files contain:
- Detailed simulation setup information
- Parameter validation results
- Simulation progress and timing
- Error messages and debugging information
- Field monitor and S-parameter processing details

Solver Comparison
-----------------

.. list-table:: Solver Feature Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Tidy3D Solver
     - Lumerical Solver
   * - **Platform**
     - Cloud-based
     - Local/Commercial license
   * - **S-matrix Calculation**
     - ComponentModeler plugin
     - Built-in S-parameter sweep
   * - **Multi-modal Support**
     - Full TE/TM/conversion
     - Full TE/TM/conversion
   * - **GPU Acceleration**
     - Cloud automatic
     - Local GPU support
   * - **Field Visualization**
     - Enhanced Tidy3D plots
     - Lumerical native
   * - **Cost**
     - Pay-per-simulation
     - License required
   * - **Setup Complexity**
     - Minimal (cloud)
     - Installation required

Extending the Solver System
----------------------------

To add support for a new FDTD solver:

1. **Inherit from the base class:**

.. code-block:: python

    from gds_fdtd.solver import fdtd_solver
    
    class fdtd_solver_newsolver(fdtd_solver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setup()

2. **Implement required abstract methods:**

.. code-block:: python

    def setup(self) -> None:
        """Setup the simulation."""
        self._validate_simulation_parameters()
        self._export_gds()
        # Solver-specific setup
        
    def run(self) -> None:
        """Run the simulation."""
        # Solver-specific execution
        
    def get_resources(self) -> None:
        """Get simulation resource requirements."""
        # Resource estimation
        
    def get_results(self) -> None:
        """Retrieve simulation results."""
        # Results processing
        
    def get_log(self) -> None:
        """Get simulation logs."""
        # Log retrieval

3. **Use modular components:**

.. code-block:: python

    # Create field monitors
    field_monitor = self.create_field_monitor_object("my_monitor", "z")
    
    # Access standardized ports
    for fdtd_port in self.fdtd_ports:
        # Port configuration using standardized interface
        pass

This modular approach ensures consistency across different solver backends while allowing for solver-specific optimizations and features. 