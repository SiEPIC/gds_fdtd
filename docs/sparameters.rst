Working with S-Parameters
=========================

S-parameters (scattering parameters) are fundamental for characterizing the performance of photonic components. The ``gds_fdtd`` package provides comprehensive tools for calculating, analyzing, and exporting S-parameters from FDTD simulations.

S-Parameter Fundamentals
------------------------

S-parameters describe how electromagnetic waves are scattered by a multi-port network. For a photonic device:

- **S₁₁**: Reflection at port 1 when excited at port 1
- **S₂₁**: Transmission from port 1 to port 2  
- **S₁₂**: Transmission from port 2 to port 1
- **S₂₂**: Reflection at port 2 when excited at port 2

For multi-modal devices, S-parameters include mode indices:
- **S₁₁⁽ᵀᴱ→ᵀᴱ⁾**: TE mode reflection at port 1
- **S₂₁⁽ᵀᴱ→ᵀᴹ⁾**: TE to TM mode conversion from port 1 to port 2

Accessing S-Parameters
----------------------

After running a simulation, S-parameters are available through the solver's ``sparameters`` property:

Basic Access
^^^^^^^^^^^^

.. code-block:: python

    # Run simulation first
    solver.run()
    
    # Access S-parameters object
    sparams = solver.sparameters
    
    # Basic information
    print(f"Component: {sparams.name}")
    print(f"Number of S-parameter entries: {len(sparams.data)}")
    print(f"Frequency points: {len(sparams.wavelength)}")

Wavelength Information
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Get wavelength array
    wavelengths = solver.sparameters.wavelength  # in micrometers
    frequencies = solver.sparameters.frequency   # in Hz
    
    print(f"Wavelength range: {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} μm")
    print(f"Number of points: {len(wavelengths)}")

Individual S-Parameter Access
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Access specific S-parameters using the ``S()`` method:

.. code-block:: python

    # Single-mode device
    s21 = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)
    
    # Multi-modal device - different mode combinations
    s21_te_te = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)  # TE→TE
    s21_te_tm = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=2)  # TE→TM
    s21_tm_te = solver.sparameters.S(in_port=1, out_port=2, in_modeid=2, out_modeid=1)  # TM→TE
    s21_tm_tm = solver.sparameters.S(in_port=1, out_port=2, in_modeid=2, out_modeid=2)  # TM→TM

S-Parameter Properties
^^^^^^^^^^^^^^^^^^^^^^

Each S-parameter entry contains magnitude and phase information:

.. code-block:: python

    # Get specific S-parameter
    s21 = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)
    
    # Access properties
    magnitude = s21.s_mag      # Linear magnitude
    phase = s21.s_phase        # Phase in radians
    power = [abs(m)**2 for m in magnitude]  # Power transmission
    
    # Convert to dB
    import numpy as np
    transmission_db = 10 * np.log10(power)
    
    print(f"Peak transmission: {max(power):.4f} ({max(transmission_db):.2f} dB)")

Visualization and Analysis
--------------------------

Built-in Plotting
^^^^^^^^^^^^^^^^^

The package provides automatic plotting of all S-parameters:

.. code-block:: python

    # Plot all S-parameters
    solver.visualize_results()  # Includes S-parameter plots and .dat export
    
    # Or directly plot S-parameters
    solver.sparameters.plot()

Custom Plotting
^^^^^^^^^^^^^^^

Create custom plots for specific analysis:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get wavelength and S-parameters
    wavelengths = solver.sparameters.wavelength
    s21 = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)
    s31 = solver.sparameters.S(in_port=1, out_port=3, in_modeid=1, out_modeid=1)
    
    # Transmission plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s21.s_mag]), 
            label='Port 1→2', linewidth=2)
    ax.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s31.s_mag]), 
            label='Port 1→3', linewidth=2)
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Transmission (dB)')
    ax.set_title('Device Transmission Spectrum')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

Multi-Modal Analysis
^^^^^^^^^^^^^^^^^^^^

Analyze mode conversion and polarization effects:

.. code-block:: python

    # Multi-modal transmission analysis
    wavelengths = solver.sparameters.wavelength
    
    # Get all mode combinations for port 1→4
    s41_te_te = solver.sparameters.S(in_port=1, out_port=4, in_modeid=1, out_modeid=1)
    s41_te_tm = solver.sparameters.S(in_port=1, out_port=4, in_modeid=1, out_modeid=2)
    s41_tm_te = solver.sparameters.S(in_port=1, out_port=4, in_modeid=2, out_modeid=1)
    s41_tm_tm = solver.sparameters.S(in_port=1, out_port=4, in_modeid=2, out_modeid=2)
    
    # Plot multi-modal transmission
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s41_te_te.s_mag]), 
            label='TE→TE', linewidth=2)
    ax.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s41_tm_tm.s_mag]), 
            label='TM→TM', linewidth=2)
    ax.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s41_te_tm.s_mag]), 
            label='TE→TM (conversion)', linewidth=2, linestyle='--')
    ax.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s41_tm_te.s_mag]), 
            label='TM→TE (conversion)', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Transmission (dB)')
    ax.set_title('Multi-Modal S-Parameters: Port 1 → Port 4')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

Performance Metrics
^^^^^^^^^^^^^^^^^^^

Calculate common device performance metrics:

.. code-block:: python

    import numpy as np
    
    def calculate_metrics(solver, in_port=1, out_port=2):
        """Calculate common performance metrics."""
        wavelengths = solver.sparameters.wavelength
        
        # Get S-parameters
        s_trans = solver.sparameters.S(in_port=in_port, out_port=out_port, 
                                      in_modeid=1, out_modeid=1)
        s_refl = solver.sparameters.S(in_port=in_port, out_port=in_port, 
                                     in_modeid=1, out_modeid=1)
        
        # Calculate metrics
        transmission = [abs(m)**2 for m in s_trans.s_mag]
        reflection = [abs(m)**2 for m in s_refl.s_mag]
        insertion_loss = [-10*np.log10(t) for t in transmission]
        return_loss = [-10*np.log10(r) for r in reflection]
        
        # Find performance at specific wavelength
        target_wl = 1.55  # μm
        idx = np.argmin(np.abs(wavelengths - target_wl))
        
        print(f"Performance at {target_wl} μm:")
        print(f"  Transmission: {transmission[idx]:.4f} ({-insertion_loss[idx]:.2f} dB)")
        print(f"  Reflection: {reflection[idx]:.4f} ({-return_loss[idx]:.2f} dB)")
        print(f"  Insertion Loss: {insertion_loss[idx]:.2f} dB")
        print(f"  Return Loss: {return_loss[idx]:.2f} dB")
        
        return {
            'wavelength': wavelengths,
            'transmission': transmission,
            'reflection': reflection,
            'insertion_loss': insertion_loss,
            'return_loss': return_loss
        }
    
    # Calculate metrics
    metrics = calculate_metrics(solver, in_port=1, out_port=4)

Exporting S-Parameters
----------------------

.dat File Export  
^^^^^^^^^^^^^^^^

Export S-parameters to standard .dat format for use in circuit simulators:

.. code-block:: python

    # Automatic export (included in visualize_results)
    solver.visualize_results()  # Creates .dat file automatically
    
    # Manual export
    solver.export_sparameters_dat("my_device_sparams.dat")
    
    # Custom filepath
    import os
    dat_path = os.path.join(solver.working_dir, "custom_sparams.dat")
    solver.export_sparameters_dat(dat_path)

The .dat file format is compatible with most circuit simulators and contains:
- Frequency sweep information
- S-parameter magnitude and phase data
- Multi-port and multi-modal data

Custom Export Formats
^^^^^^^^^^^^^^^^^^^^^^

Export to custom formats for specific analysis tools:

.. code-block:: python

    import json
    import numpy as np
    
    def export_json(solver, filename):
        """Export S-parameters to JSON format."""
        data = {
            'component': solver.component.name,
            'wavelength_um': solver.sparameters.wavelength.tolist(),
            'frequency_hz': solver.sparameters.frequency.tolist(),
            'sparameters': []
        }
        
        for sparam in solver.sparameters.data:
            entry = {
                'in_port': sparam.in_port,
                'out_port': sparam.out_port,
                'in_mode': sparam.in_modeid,
                'out_mode': sparam.out_modeid,
                'magnitude': sparam.s_mag,
                'phase_rad': sparam.s_phase,
                'power': [abs(m)**2 for m in sparam.s_mag]
            }
            data['sparameters'].append(entry)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"S-parameters exported to {filename}")
    
    # Export to JSON
    export_json(solver, "device_sparams.json")

Touchstone Format
^^^^^^^^^^^^^^^^^

Export to Touchstone (.s2p, .s4p) format for RF/microwave tools:

.. code-block:: python

    def export_touchstone(solver, filename):
        """Export to Touchstone format."""
        import numpy as np
        
        # Get number of ports
        n_ports = len(solver.fdtd_ports)
        wavelengths = solver.sparameters.wavelength
        frequencies = solver.sparameters.frequency
        
        with open(filename, 'w') as f:
            # Header
            f.write(f"# Hz S MA R 50\n")
            f.write(f"! Exported from gds_fdtd for {solver.component.name}\n")
            
            # Data for each frequency
            for i, freq in enumerate(frequencies):
                f.write(f"{freq:.6e}")
                
                # Write S-parameters in order (S11, S12, S21, S22 for 2-port)
                for out_port in range(1, n_ports + 1):
                    for in_port in range(1, n_ports + 1):
                        s_param = solver.sparameters.S(in_port=in_port, out_port=out_port,
                                                      in_modeid=1, out_modeid=1)
                        mag = abs(s_param.s_mag[i])
                        phase_deg = np.degrees(s_param.s_phase[i])
                        f.write(f" {mag:.6e} {phase_deg:.6e}")
                
                f.write("\n")
        
        print(f"Touchstone file exported: {filename}")
    
    # Export 4-port device
    export_touchstone(solver, "device.s4p")

S-Parameter Validation
----------------------

Data Quality Checks
^^^^^^^^^^^^^^^^^^^^

Validate S-parameter data quality and physical consistency:

.. code-block:: python

    def validate_sparameters(solver):
        """Validate S-parameter data for physical consistency."""
        print("S-Parameter Validation:")
        print("=" * 40)
        
        # Check data completeness
        expected_combinations = len(solver.fdtd_ports)**2 * len(solver.modes)**2
        actual_combinations = len(solver.sparameters.data)
        print(f"S-parameter combinations: {actual_combinations}/{expected_combinations}")
        
        # Check energy conservation (for lossless devices)
        wavelengths = solver.sparameters.wavelength
        n_ports = len(solver.fdtd_ports)
        
        for i, wl in enumerate(wavelengths[::10]):  # Check every 10th point
            total_power = 0
            for out_port in range(1, n_ports + 1):
                s_param = solver.sparameters.S(in_port=1, out_port=out_port,
                                              in_modeid=1, out_modeid=1)
                power = abs(s_param.s_mag[i*10])**2
                total_power += power
            
            print(f"λ={wl:.3f}μm: Total power = {total_power:.4f}")
            if total_power > 1.01:
                print(f"  WARNING: Power > 1 (gain or numerical error)")
            elif total_power < 0.95:
                print(f"  INFO: Power < 1 (loss present)")
    
    # Run validation
    validate_sparameters(solver)

Reciprocity Check
^^^^^^^^^^^^^^^^^

For reciprocal devices, verify S₁₂ = S₂₁:

.. code-block:: python

    def check_reciprocity(solver, tolerance=0.01):
        """Check device reciprocity."""
        n_ports = len(solver.fdtd_ports)
        wavelengths = solver.sparameters.wavelength
        
        print("Reciprocity Check:")
        print("-" * 20)
        
        for i in range(1, n_ports + 1):
            for j in range(i + 1, n_ports + 1):
                s_ij = solver.sparameters.S(in_port=i, out_port=j, in_modeid=1, out_modeid=1)
                s_ji = solver.sparameters.S(in_port=j, out_port=i, in_modeid=1, out_modeid=1)
                
                # Compare magnitudes
                mag_diff = np.mean([abs(abs(m1) - abs(m2)) for m1, m2 in 
                                   zip(s_ij.s_mag, s_ji.s_mag)])
                
                print(f"S{i}{j} vs S{j}{i}: Avg magnitude difference = {mag_diff:.4f}")
                if mag_diff > tolerance:
                    print(f"  WARNING: Large reciprocity error (>{tolerance})")
    
    # Check reciprocity
    check_reciprocity(solver)

Advanced S-Parameter Analysis
-----------------------------

Frequency Domain Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze S-parameters in the frequency domain:

.. code-block:: python

    def analyze_bandwidth(solver, in_port=1, out_port=2, threshold_db=-3):
        """Calculate 3dB bandwidth."""
        s_param = solver.sparameters.S(in_port=in_port, out_port=out_port,
                                      in_modeid=1, out_modeid=1)
        wavelengths = solver.sparameters.wavelength
        
        # Convert to dB
        power_db = [10*np.log10(abs(m)**2) for m in s_param.s_mag]
        max_power_db = max(power_db)
        threshold = max_power_db + threshold_db  # -3dB from peak
        
        # Find bandwidth
        above_threshold = [p > threshold for p in power_db]
        if any(above_threshold):
            start_idx = above_threshold.index(True)
            end_idx = len(above_threshold) - above_threshold[::-1].index(True) - 1
            
            bandwidth = wavelengths[end_idx] - wavelengths[start_idx]
            center_wl = (wavelengths[start_idx] + wavelengths[end_idx]) / 2
            
            print(f"Device Bandwidth Analysis:")
            print(f"  Center wavelength: {center_wl:.3f} μm")
            print(f"  {-threshold_db}dB bandwidth: {bandwidth:.3f} μm")
            print(f"  Relative bandwidth: {bandwidth/center_wl*100:.2f}%")
            
            return center_wl, bandwidth
        else:
            print("No points above threshold found")
            return None, None
    
    # Analyze bandwidth
    center, bw = analyze_bandwidth(solver, threshold_db=-3)

Group Delay Analysis
^^^^^^^^^^^^^^^^^^^^

Calculate group delay from S-parameter phase:

.. code-block:: python

    def calculate_group_delay(solver, in_port=1, out_port=2):
        """Calculate group delay from S-parameter phase."""
        s_param = solver.sparameters.S(in_port=in_port, out_port=out_port,
                                      in_modeid=1, out_modeid=1)
        
        frequencies = solver.sparameters.frequency
        phases = s_param.s_phase
        
        # Unwrap phase for continuous derivative
        phases_unwrapped = np.unwrap(phases)
        
        # Calculate group delay: τ = -dφ/dω
        group_delay = -np.gradient(phases_unwrapped, frequencies)
        wavelengths = solver.sparameters.wavelength
        
        # Plot group delay
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Phase plot
        ax1.plot(wavelengths, np.degrees(phases), 'b-', linewidth=2)
        ax1.set_ylabel('Phase (degrees)')
        ax1.set_title('S-Parameter Phase and Group Delay')
        ax1.grid(True, alpha=0.3)
        
        # Group delay plot
        ax2.plot(wavelengths, group_delay * 1e12, 'r-', linewidth=2)  # Convert to ps
        ax2.set_xlabel('Wavelength (μm)')
        ax2.set_ylabel('Group Delay (ps)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return group_delay
    
    # Calculate group delay
    gd = calculate_group_delay(solver)

Best Practices
--------------

1. **Always validate S-parameter data** for physical consistency
2. **Use appropriate wavelength resolution** for your analysis needs
3. **Check reciprocity** for passive, reciprocal devices  
4. **Export data in multiple formats** for different analysis tools
5. **Document simulation parameters** with S-parameter files
6. **Verify convergence** with mesh and time step refinement
7. **Use multi-modal analysis** for polarization-sensitive devices

This comprehensive S-parameter functionality enables thorough characterization and analysis of photonic devices for both research and commercial applications. 