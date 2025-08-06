Multi-Modal Simulations
=======================

Multi-modal simulations enable analysis of devices that support multiple electromagnetic modes, such as TE (Transverse Electric) and TM (Transverse Magnetic) polarizations. This capability is essential for understanding polarization effects, mode conversion, and designing polarization-insensitive devices.

Understanding Multi-Modal Physics
----------------------------------

Optical Modes
^^^^^^^^^^^^^

In photonic waveguides, electromagnetic energy propagates in discrete modes:

- **TE Modes (Mode 1)**: Electric field primarily transverse to propagation direction
- **TM Modes (Mode 2)**: Magnetic field primarily transverse to propagation direction  
- **Higher-order modes**: Additional spatial field distributions

Mode Properties:
- Each mode has a different effective index
- Modes can couple and convert between each other
- Device geometry affects mode confinement and coupling

Multi-Modal Effects
^^^^^^^^^^^^^^^^^^^

Important phenomena in multi-modal devices:

- **Mode Conversion**: TE↔TM conversion due to asymmetry or birefringence
- **Polarization Rotation**: Rotation of linear polarization states
- **Differential Phase Shift**: Different phase velocities for TE and TM modes
- **Polarization Dependent Loss (PDL)**: Different losses for TE and TM modes

Enabling Multi-Modal Simulation
-------------------------------

Basic Multi-Modal Setup
^^^^^^^^^^^^^^^^^^^^^^^^

Enable multi-modal simulation by specifying multiple modes:

.. code-block:: python

    from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d
    
    # Multi-modal solver setup
    solver = fdtd_solver_tidy3d(
        component=component,
        tech=technology,
        modes=[1, 2],  # TE (mode 1) and TM (mode 2)
        wavelength_start=1.50,
        wavelength_end=1.60,
        wavelength_points=101,
        # ... other parameters
    )

Mode Configuration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Single mode (fundamental TE)
    modes=[1]
    
    # Dual polarization (TE and TM)
    modes=[1, 2]
    
    # Higher-order modes
    modes=[1, 2, 3, 4]  # Fundamental + higher-order modes

The solver automatically:
- Calculates all mode combinations for S-parameters
- Sets up appropriate mode sources and monitors
- Handles mode normalization and coupling analysis

Tidy3D Multi-Modal Implementation
----------------------------------

ComponentModeler Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Tidy3D solver uses the official ComponentModeler plugin for accurate multi-modal S-matrix calculation:

.. code-block:: python

    # Tidy3D automatically handles:
    # - Mode solver setup for each port
    # - Source excitation for each mode
    # - Monitor placement and mode decomposition
    # - S-parameter normalization
    
    solver = fdtd_solver_tidy3d(
        component=component,
        tech=technology,
        modes=[1, 2],  # ComponentModeler handles both modes automatically
        # ... parameters
    )
    
    solver.run()  # Full multi-modal S-matrix calculation

Port Mode Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Each port is configured for multi-modal operation:

.. code-block:: python

    # Inspect multi-modal port setup
    print(f"Solver modes: {solver.modes}")
    print(f"Total port-mode combinations: {len(solver.smatrix_ports) * len(solver.modes)}")
    
    for port in solver.smatrix_ports:
        print(f"Port {port.name}: {port.mode_spec.num_modes} modes configured")

Multi-Modal S-Parameter Analysis
---------------------------------

Accessing Multi-Modal S-Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After simulation, access all mode combinations:

.. code-block:: python

    # Run multi-modal simulation
    solver.run()
    
    # Access different mode combinations
    wavelengths = solver.sparameters.wavelength
    
    # Direct transmission (no mode conversion)
    s21_te_te = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)  # TE→TE
    s21_tm_tm = solver.sparameters.S(in_port=1, out_port=2, in_modeid=2, out_modeid=2)  # TM→TM
    
    # Cross-modal transmission (mode conversion)
    s21_te_tm = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=2)  # TE→TM
    s21_tm_te = solver.sparameters.S(in_port=1, out_port=2, in_modeid=2, out_modeid=1)  # TM→TE
    
    print(f"Available S-parameter combinations: {len(solver.sparameters.data)}")

Multi-Modal Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

Create multi-modal plots:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_multimodal_transmission(solver, in_port=1, out_port=2):
        """Plot all modal transmission combinations."""
        wavelengths = solver.sparameters.wavelength
        
        # Get all mode combinations
        s_te_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=1)
        s_tm_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=2)
        s_te_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=2)
        s_tm_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=1)
        
        # Create multi-panel plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Direct transmission (no conversion)
        ax1.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s_te_te.s_mag]), 
                'b-', linewidth=2, label='TE→TE')
        ax1.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s_tm_tm.s_mag]), 
                'r-', linewidth=2, label='TM→TM')
        ax1.set_title('Direct Transmission (No Conversion)')
        ax1.set_ylabel('Transmission (dB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mode conversion
        ax2.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s_te_tm.s_mag]), 
                'g--', linewidth=2, label='TE→TM')
        ax2.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s_tm_te.s_mag]), 
                'm--', linewidth=2, label='TM→TE')
        ax2.set_title('Mode Conversion')
        ax2.set_ylabel('Transmission (dB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Phase comparison
        ax3.plot(wavelengths, np.unwrap(s_te_te.s_phase), 'b-', linewidth=2, label='TE→TE')
        ax3.plot(wavelengths, np.unwrap(s_tm_tm.s_phase), 'r-', linewidth=2, label='TM→TM')
        ax3.set_title('Phase Response')
        ax3.set_xlabel('Wavelength (μm)')
        ax3.set_ylabel('Phase (rad)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Polarization extinction ratio
        ter_te = [10*np.log10(abs(t)**2 / abs(c)**2) for t, c in 
                  zip(s_te_te.s_mag, s_te_tm.s_mag)]
        ter_tm = [10*np.log10(abs(t)**2 / abs(c)**2) for t, c in 
                  zip(s_tm_tm.s_mag, s_tm_te.s_mag)]
        
        ax4.plot(wavelengths, ter_te, 'b-', linewidth=2, label='TE Extinction Ratio')
        ax4.plot(wavelengths, ter_tm, 'r-', linewidth=2, label='TM Extinction Ratio')
        ax4.set_title('Polarization Extinction Ratio')
        ax4.set_xlabel('Wavelength (μm)')
        ax4.set_ylabel('Extinction Ratio (dB)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Create multi-modal plot
    plot_multimodal_transmission(solver, in_port=1, out_port=4)

Multi-Modal Performance Metrics
-------------------------------

Polarization Dependent Loss (PDL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate PDL for device characterization:

.. code-block:: python

    def calculate_pdl(solver, in_port=1, out_port=2):
        """Calculate Polarization Dependent Loss."""
        # Get TE and TM transmission
        s_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=1)
        s_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=2)
        
        # Calculate power transmission
        power_te = [abs(m)**2 for m in s_te.s_mag]
        power_tm = [abs(m)**2 for m in s_tm.s_mag]
        
        # PDL = |10*log10(P_TE/P_TM)|
        pdl = [abs(10*np.log10(pte/ptm)) for pte, ptm in zip(power_te, power_tm)]
        
        wavelengths = solver.sparameters.wavelength
        max_pdl = max(pdl)
        avg_pdl = np.mean(pdl)
        
        print(f"Polarization Dependent Loss Analysis:")
        print(f"  Maximum PDL: {max_pdl:.2f} dB")
        print(f"  Average PDL: {avg_pdl:.2f} dB")
        
        # Plot PDL vs wavelength
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, pdl, 'k-', linewidth=2)
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('PDL (dB)')
        plt.title('Polarization Dependent Loss')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return pdl
    
    # Calculate PDL
    pdl_values = calculate_pdl(solver)

Mode Conversion Efficiency
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze mode conversion characteristics:

.. code-block:: python

    def analyze_mode_conversion(solver, in_port=1, out_port=2):
        """Analyze mode conversion efficiency."""
        wavelengths = solver.sparameters.wavelength
        
        # Get all mode combinations
        s_te_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=1)
        s_te_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=2)
        s_tm_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=1)
        s_tm_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=2)
        
        # Calculate conversion efficiencies
        # For TE input
        te_direct = [abs(m)**2 for m in s_te_te.s_mag]  # TE→TE (direct)
        te_conversion = [abs(m)**2 for m in s_te_tm.s_mag]  # TE→TM (conversion)
        te_conversion_eff = [c/(d+c) for d, c in zip(te_direct, te_conversion)]
        
        # For TM input  
        tm_direct = [abs(m)**2 for m in s_tm_tm.s_mag]  # TM→TM (direct)
        tm_conversion = [abs(m)**2 for m in s_tm_te.s_mag]  # TM→TE (conversion)
        tm_conversion_eff = [c/(d+c) for d, c in zip(tm_direct, tm_conversion)]
        
        # Find peak conversion
        max_te_conv = max(te_conversion_eff)
        max_tm_conv = max(tm_conversion_eff)
        
        print(f"Mode Conversion Analysis:")
        print(f"  Max TE→TM conversion: {max_te_conv:.1%}")
        print(f"  Max TM→TE conversion: {max_tm_conv:.1%}")
        
        # Plot conversion efficiency
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, [eff*100 for eff in te_conversion_eff], 
                'b-', linewidth=2, label='TE→TM Conversion')
        plt.plot(wavelengths, [eff*100 for eff in tm_conversion_eff], 
                'r-', linewidth=2, label='TM→TE Conversion')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Conversion Efficiency (%)')
        plt.title('Mode Conversion Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return te_conversion_eff, tm_conversion_eff
    
    # Analyze mode conversion
    te_conv, tm_conv = analyze_mode_conversion(solver)

Birefringence Analysis
^^^^^^^^^^^^^^^^^^^^^^

Calculate effective index difference between modes:

.. code-block:: python

    def calculate_birefringence(solver, in_port=1, out_port=2, device_length=None):
        """Calculate device birefringence from phase difference."""
        wavelengths = solver.sparameters.wavelength
        
        # Get phase for TE and TM modes
        s_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=1)
        s_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=2)
        
        # Unwrap phases for continuous analysis
        phase_te = np.unwrap(s_te.s_phase)
        phase_tm = np.unwrap(s_tm.s_phase)
        phase_diff = phase_te - phase_tm
        
        if device_length is not None:
            # Calculate effective index difference
            # Δn_eff = (φ_TE - φ_TM) * λ / (2π * L)
            delta_n_eff = [pd * wl / (2 * np.pi * device_length) 
                          for pd, wl in zip(phase_diff, wavelengths)]
            
            avg_birefringence = np.mean(delta_n_eff)
            print(f"Device Birefringence Analysis:")
            print(f"  Device length: {device_length} μm")
            print(f"  Average Δn_eff: {avg_birefringence:.6f}")
            
            # Plot birefringence vs wavelength
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, delta_n_eff, 'g-', linewidth=2)
            plt.xlabel('Wavelength (μm)')
            plt.ylabel('Effective Index Difference')
            plt.title('Device Birefringence')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            return delta_n_eff
        else:
            # Just show phase difference
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, phase_diff, 'purple', linewidth=2)
            plt.xlabel('Wavelength (μm)')
            plt.ylabel('Phase Difference (rad)')
            plt.title('TE-TM Phase Difference')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            return phase_diff
    
    # Calculate birefringence (specify device length if known)
    device_length = 10.0  # μm - replace with actual device length
    birefringence = calculate_birefringence(solver, device_length=device_length)

Device-Specific Multi-Modal Analysis
-------------------------------------

Polarization Beam Splitter (PBS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze PBS performance:

.. code-block:: python

    def analyze_pbs(solver):
        """Analyze Polarization Beam Splitter performance."""
        wavelengths = solver.sparameters.wavelength
        
        # Assume input port 1, TE output port 2, TM output port 3
        s_te_output = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)  # TE→TE
        s_tm_output = solver.sparameters.S(in_port=1, out_port=3, in_modeid=2, out_modeid=2)  # TM→TM
        
        # Cross-talk
        s_te_crosstalk = solver.sparameters.S(in_port=1, out_port=3, in_modeid=1, out_modeid=1)  # TE→TM port
        s_tm_crosstalk = solver.sparameters.S(in_port=1, out_port=2, in_modeid=2, out_modeid=2)  # TM→TE port
        
        # Calculate extinction ratios
        te_extinction = [10*np.log10(abs(sig)**2 / abs(xt)**2) 
                        for sig, xt in zip(s_te_output.s_mag, s_te_crosstalk.s_mag)]
        tm_extinction = [10*np.log10(abs(sig)**2 / abs(xt)**2) 
                        for sig, xt in zip(s_tm_output.s_mag, s_tm_crosstalk.s_mag)]
        
        print(f"PBS Performance Analysis:")
        print(f"  TE extinction ratio: {min(te_extinction):.1f} - {max(te_extinction):.1f} dB")
        print(f"  TM extinction ratio: {min(tm_extinction):.1f} - {max(tm_extinction):.1f} dB")
        
        # Plot PBS performance
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Transmission
        ax1.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s_te_output.s_mag]), 
                'b-', linewidth=2, label='TE Transmission')
        ax1.plot(wavelengths, 10*np.log10([abs(m)**2 for m in s_tm_output.s_mag]), 
                'r-', linewidth=2, label='TM Transmission')
        ax1.set_ylabel('Transmission (dB)')
        ax1.set_title('PBS Transmission')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Extinction ratio
        ax2.plot(wavelengths, te_extinction, 'b-', linewidth=2, label='TE Extinction')
        ax2.plot(wavelengths, tm_extinction, 'r-', linewidth=2, label='TM Extinction')
        ax2.set_xlabel('Wavelength (μm)')
        ax2.set_ylabel('Extinction Ratio (dB)')
        ax2.set_title('PBS Extinction Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

Polarization Rotator
^^^^^^^^^^^^^^^^^^^^

Analyze polarization rotation devices:

.. code-block:: python

    def analyze_polarization_rotator(solver, in_port=1, out_port=2):
        """Analyze polarization rotator performance."""
        wavelengths = solver.sparameters.wavelength
        
        # Get all mode combinations
        s_te_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=1)
        s_te_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=1, out_modeid=2)
        s_tm_te = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=1)
        s_tm_tm = solver.sparameters.S(in_port=in_port, out_port=out_port, in_modeid=2, out_modeid=2)
        
        # Calculate rotation angle
        # For a perfect rotator: cos²(θ) and sin²(θ) for converted/direct components
        rotation_angles = []
        for i in range(len(wavelengths)):
            # For TE input
            te_direct = abs(s_te_te.s_mag[i])**2
            te_converted = abs(s_te_tm.s_mag[i])**2
            total_power = te_direct + te_converted
            
            if total_power > 0:
                cos_theta_sq = te_direct / total_power
                theta = np.arccos(np.sqrt(cos_theta_sq)) * 180 / np.pi
                rotation_angles.append(theta)
            else:
                rotation_angles.append(0)
        
        avg_rotation = np.mean(rotation_angles)
        print(f"Polarization Rotator Analysis:")
        print(f"  Average rotation angle: {avg_rotation:.1f}°")
        
        # Plot rotation angle vs wavelength
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, rotation_angles, 'purple', linewidth=2)
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Rotation Angle (degrees)')
        plt.title('Polarization Rotation Angle')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return rotation_angles

Best Practices for Multi-Modal Simulations
------------------------------------------

1. **Mode Selection**
   - Include all relevant modes for your device
   - Higher-order modes may be needed for multimode devices
   - Consider computational cost vs. accuracy trade-offs

2. **Simulation Parameters**
   - Use adequate mesh resolution for mode confinement
   - Ensure simulation domain captures mode tails
   - Consider run time for mode beating phenomena

3. **Analysis Considerations**
   - Always check energy conservation across modes
   - Verify reciprocity for passive devices
   - Use appropriate normalization for mode coupling

4. **Validation**
   - Compare with analytical models when available
   - Cross-check with experimental data
   - Verify mesh convergence for mode calculations

Multi-modal simulations help you understand polarization effects and mode coupling in photonic devices. This is useful for designing devices where polarization control matters. 