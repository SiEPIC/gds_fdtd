Setting Up Simulations
======================

This guide covers how to set up and run FDTD simulations using the ``gds_fdtd`` package, from loading GDS files to configuring simulation parameters and executing simulations.

Workflow Overview
-----------------

The typical simulation workflow follows these steps:

1. **Load GDS layout** and extract the target cell
2. **Load technology file** defining materials and layers
3. **Create component** from layout and technology
4. **Initialize solver** with simulation parameters
5. **Run simulation** and analyze results

Basic Simulation Setup
----------------------

Loading GDS and Technology
^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, load your GDS file and technology definition:

.. code-block:: python

    import os
    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.simprocessor import load_component_from_tech
    from gds_fdtd.lyprocessor import load_cell
    
    # Load technology file
    tech_path = "examples/tech_tidy3d.yaml"
    technology = parse_yaml_tech(tech_path)
    
    # Load GDS file and extract cell
    gds_file = "examples/devices.gds"
    cell, layout = load_cell(gds_file, top_cell='crossing_te1550')
    
    # Create component from cell and technology
    component = load_component_from_tech(cell=cell, tech=technology)

Component Analysis
^^^^^^^^^^^^^^^^^^

Inspect the loaded component:

.. code-block:: python

    # Component information
    print(f"Component name: {component.name}")
    print(f"Number of ports: {len(component.ports)}")
    print(f"Number of structures: {len(component.structures)}")
    
    # Port information
    for i, port in enumerate(component.ports):
        print(f"Port {i+1}: {port.name} at ({port.center[0]:.2f}, {port.center[1]:.2f}) μm")
        print(f"  Direction: {port.direction}° Width: {port.width:.2f} μm")
    
    # Simulation bounds
    print(f"Bounds: {component.bounds.x_span:.2f} × {component.bounds.y_span:.2f} μm")

Solver Configuration
^^^^^^^^^^^^^^^^^^^^

Configure the solver with appropriate parameters:

.. code-block:: python

    from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d
    
    solver = fdtd_solver_tidy3d(
        component=component,
        tech=technology,
        port_input=[component.ports[0]],  # Active ports for excitation
        wavelength_start=1.50,            # Start wavelength (μm)
        wavelength_end=1.60,              # End wavelength (μm)  
        wavelength_points=101,            # Number of wavelength points
        mesh=15,                          # Mesh cells per wavelength
        modes=[1, 2],                     # Mode indices (TE, TM)
        field_monitors=["z"],             # Field monitor axes
        working_dir="./simulation_results"
    )

Simulation Parameters
---------------------

Wavelength Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Configure the wavelength range and resolution:

.. code-block:: python

    # Broadband simulation
    solver = fdtd_solver_tidy3d(
        wavelength_start=1.45,  # Start wavelength
        wavelength_end=1.65,    # End wavelength
        wavelength_points=201,  # High resolution
        # ... other parameters
    )
    
    # Narrowband simulation
    solver = fdtd_solver_tidy3d(
        wavelength_start=1.549,  # Near target wavelength
        wavelength_end=1.551,    # Narrow range
        wavelength_points=21,    # Sufficient points
        # ... other parameters
    )

Mesh Configuration
^^^^^^^^^^^^^^^^^^

The mesh parameter controls simulation accuracy vs. computational cost:

.. code-block:: python

    # Coarse mesh (faster, less accurate)
    mesh=10  # 10 cells per wavelength
    
    # Standard mesh (balanced)
    mesh=15  # 15 cells per wavelength (recommended)
    
    # Fine mesh (slower, more accurate)
    mesh=20  # 20 cells per wavelength

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

Configure boundary conditions for different solver types:

.. code-block:: python

    # Tidy3D (automatic PML boundaries)
    solver_tidy3d = fdtd_solver_tidy3d(
        # Boundaries handled automatically
        symmetry=[0, 0, 0],  # No symmetry
        # ... other parameters
    )
    
    # Lumerical (explicit boundary configuration)
    solver_lumerical = fdtd_solver_lumerical(
        boundary=["PML", "PML", "PML"],     # PML on all boundaries
        symmetry=[0, 1, 0],                 # Mirror symmetry in Y
        # ... other parameters
    )

Simulation Domain
^^^^^^^^^^^^^^^^^

The simulation domain is automatically calculated but can be controlled:

.. code-block:: python

    solver = fdtd_solver_tidy3d(
        z_min=-1.0,        # Bottom of simulation domain (μm)
        z_max=1.5,         # Top of simulation domain (μm)
        buffer=1.0,        # Buffer around component (μm)
        width_ports=3.0,   # Port width (μm)
        depth_ports=2.0,   # Port depth in Z (μm)
        # ... other parameters
    )

Port Configuration
^^^^^^^^^^^^^^^^^^

Configure which ports to excite:

.. code-block:: python

    # Single port excitation
    solver = fdtd_solver_tidy3d(
        port_input=[component.ports[0]],  # Excite only first port
        # ... other parameters
    )
    
    # Multiple port excitation
    solver = fdtd_solver_tidy3d(
        port_input=[component.ports[0], component.ports[2]],  # Excite ports 1 and 3
        # ... other parameters
    )
    
    # All ports (for full S-matrix)
    solver = fdtd_solver_tidy3d(
        port_input=component.ports,  # Excite all ports
        # ... other parameters
    )

Field Monitoring
^^^^^^^^^^^^^^^^

Configure field monitors for visualization:

.. code-block:: python

    # Single axis monitoring
    field_monitors=["z"]           # Monitor Z-normal plane
    
    # Multiple axes
    field_monitors=["x", "y", "z"]  # Monitor all axes
    
    # No field monitoring (faster)
    field_monitors=[]              # No field monitors

Running Simulations
-------------------

Basic Execution
^^^^^^^^^^^^^^^

Run a simulation with standard settings:

.. code-block:: python

    # Setup is automatic during solver initialization
    print("Solver configured, ready to run...")
    
    # Check resource requirements (optional)
    solver.get_resources()
    
    # Run simulation
    solver.run()
    
    # Check results
    print(f"Simulation complete: {len(solver.sparameters.data)} S-parameters calculated")

Tidy3D Cloud Execution
^^^^^^^^^^^^^^^^^^^^^^^

For Tidy3D simulations, monitor cloud execution:

.. code-block:: python

    # Run with monitoring
    solver.run()  # Automatically handles cloud submission and monitoring
    
    # Results are automatically downloaded and processed
    print("S-parameters ready for analysis")

Error Handling
^^^^^^^^^^^^^^

Handle common simulation errors:

.. code-block:: python

    try:
        solver.run()
    except RuntimeError as e:
        print(f"Simulation failed: {e}")
        
        # Check logs for detailed error information
        solver.get_log()
        
        # Examine log files
        import os
        log_files = [f for f in os.listdir(solver.working_dir) if f.endswith('.log')]
        if log_files:
            print(f"Check log file: {log_files[0]}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

Working Directory Structure
---------------------------

The solver automatically creates a working directory structure:

.. code-block:: text

    simulation_results/
    ├── component_name/
    │   ├── component_name_timestamp.log    # Detailed simulation log
    │   ├── component_name.gds              # Exported GDS with extensions
    │   ├── component_name_sparams.dat      # S-parameters in .dat format
    │   └── field_data/                     # Field monitor data (if any)

Accessing the Working Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Working directory information
    print(f"Working directory: {solver.working_dir}")
    
    # List generated files
    import os
    for file in os.listdir(solver.working_dir):
        print(f"Generated file: {file}")
    
    # Access log files
    log_files = [f for f in os.listdir(solver.working_dir) if f.endswith('.log')]
    if log_files:
        with open(os.path.join(solver.working_dir, log_files[0]), 'r') as f:
            log_content = f.read()
            print("Simulation log preview:")
            print(log_content[:500] + "...")

Advanced Configuration
----------------------

Custom Working Directory
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import os
    from datetime import datetime
    
    # Custom working directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_dir = f"./simulations/{component.name}_{timestamp}"
    
    solver = fdtd_solver_tidy3d(
        working_dir=custom_dir,
        # ... other parameters
    )

Simulation Time Control
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Control simulation runtime
    solver = fdtd_solver_tidy3d(
        run_time_factor=5.0,    # Conservative (longer simulation)
        # run_time_factor=2.0,  # Aggressive (shorter simulation)
        # ... other parameters
    )

Memory and Performance
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # For large simulations, consider:
    
    # 1. Reduce wavelength points
    wavelength_points=51  # Instead of 101
    
    # 2. Coarser mesh
    mesh=12  # Instead of 15
    
    # 3. Smaller simulation domain
    buffer=0.5  # Instead of 1.0
    
    # 4. Fewer field monitors
    field_monitors=[]  # No field monitoring

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Port Detection Issues:**

.. code-block:: python

    # Check port extraction
    if len(component.ports) == 0:
        print("No ports found! Check:")
        print("- Port layer in technology file")
        print("- Port shapes in GDS file")
        print("- GDS layer mapping")

**Material Issues:**

.. code-block:: python

    # Verify materials are defined
    for structure in component.structures:
        if isinstance(structure, list):
            for s in structure:
                print(f"Structure {s.name}: material = {s.material}")
        else:
            print(f"Structure {structure.name}: material = {structure.material}")

**Simulation Domain Issues:**

.. code-block:: python

    # Check simulation bounds
    print(f"Component bounds: {component.bounds.x_span} × {component.bounds.y_span} μm")
    print(f"Simulation domain: {solver.span[0]} × {solver.span[1]} × {solver.span[2]} μm")
    print(f"Domain center: ({solver.center[0]}, {solver.center[1]}, {solver.center[2]}) μm")

Validation Checklist
^^^^^^^^^^^^^^^^^^^^^

Before running simulations, verify:

- [ ] GDS file loads without errors
- [ ] Technology file matches GDS layers
- [ ] Component has expected number of ports
- [ ] Materials are properly defined
- [ ] Wavelength range covers device operation
- [ ] Mesh resolution is appropriate
- [ ] Working directory is accessible
- [ ] For Tidy3D: Valid account credentials

Performance Tips
----------------

- Start with coarse parameters for testing, then refine
- Use symmetry when possible to reduce simulation size  
- Monitor field data only when needed for analysis
- Consider wavelength range vs. computational cost
- Use appropriate mesh resolution for accuracy requirements
- Enable logging to track simulation progress and debug issues 