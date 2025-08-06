#%%
"""
Comprehensive convergence study for Tidy3D simulations using the fdtd_solver_tidy3d class.
This script runs convergence studies for:
1. Mesh resolution
2. Port width 
3. Port depth
Each study plots S21 and S11 convergence and saves results.

@author: Mustafa Hammood
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from gds_fdtd.solver_tidy3d import fdtd_solver_tidy3d
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import load_component_from_tech
from gds_fdtd.lyprocessor import load_cell
import pickle
import time
from datetime import datetime

def create_working_directory():
    """Create a timestamped working directory for all convergence studies."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"convergence_study_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def run_mesh_convergence_study(base_working_dir):
    """Run mesh convergence study for different mesh resolutions."""
    
    # Mesh sizes to test
    mesh_sizes = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    # Storage for results
    results = {
        'mesh_sizes': mesh_sizes,
        's21_data': [],  # S21 (transmission)
        's11_data': [],  # S11 (reflection)
        'wavelengths': None,
        'simulation_times': []
    }
    
    # Load technology file - need to modify for Tidy3D materials
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # Load GDS component
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")
    cell, layout = load_cell(file_gds, top_cell='sbend_dontfabme')
    component = load_component_from_tech(cell=cell, tech=technology)
    
    print(f"\n{'='*60}")
    print(f"MESH CONVERGENCE STUDY")
    print(f"{'='*60}")
    print(f"Starting mesh convergence study with mesh sizes: {mesh_sizes}")
    print(f"Component: {cell.name}")
    print(f"Number of ports: {len(component.ports)}")
    
    for i, mesh_size in enumerate(mesh_sizes):
        print(f"\n--- Running simulation {i+1}/{len(mesh_sizes)} with mesh = {mesh_size} ---")
        start_time = time.time()
        
        try:
            # Create Tidy3D solver with current mesh size
            solver = fdtd_solver_tidy3d(
                component=component,
                tech=technology,
                port_input=[component.ports[0]],  # Input port(s)
                visualize=False,  # Set to True to show simulation setup plots
                wavelength_start=1.5,
                wavelength_end=1.6,
                wavelength_points=50,  # Fewer points for faster simulation
                mesh=mesh_size,  # Current mesh size being tested
                boundary=["PML", "PML", "PML"],
                symmetry=[0, 0, 0],
                z_min=-1.0,
                z_max=1.11,
                width_ports=2.0,  # Fixed port width
                depth_ports=1.5,  # Fixed port depth
                buffer=1.0,
                modes=[1],  # TE mode
                run_time_factor=50,
                field_monitors=None,
                working_dir=os.path.join(base_working_dir, "mesh_study", f"mesh{mesh_size}"),
            )
            
            # This requires Tidy3D cloud credentials and will consume simulation credits   
            solver.run()
            
            # Extract S-parameters
            wavelengths = solver.sparameters.wavelength
            
            # Get S21 (transmission from port 1 to port 2) and S11 (reflection at port 1)
            s21 = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)
            s11 = solver.sparameters.S(in_port=1, out_port=1, in_modeid=1, out_modeid=1)
            
            # Store results
            results['s21_data'].append(s21.s_mag)
            results['s11_data'].append(s11.s_mag)
            
            if results['wavelengths'] is None:
                results['wavelengths'] = wavelengths
                
            simulation_time = time.time() - start_time
            results['simulation_times'].append(simulation_time)
            
            print(f"Simulation completed in {simulation_time:.1f} seconds")
            print(f"S21 at 1.55μm: {np.interp(1.55, wavelengths, np.abs(s21.s_mag)):.4f}")
            print(f"S11 at 1.55μm: {np.interp(1.55, wavelengths, np.abs(s11.s_mag)):.4f}")
            
        except Exception as e:
            print(f"Error in simulation with mesh={mesh_size}: {str(e)}")
            # Store None values to maintain array consistency
            results['s21_data'].append(None)
            results['s11_data'].append(None)
            results['simulation_times'].append(None)
    
    return results

def run_port_width_convergence_study(base_working_dir):
    """Run port width convergence study."""
    
    # Port widths to test
    port_widths = [0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5]
    
    # Storage for results
    results = {
        'port_widths': port_widths,
        's21_data': [],  # S21 (transmission)
        's11_data': [],  # S11 (reflection)
        'wavelengths': None,
        'simulation_times': []
    }
    
    # Load technology file - need to modify for Tidy3D materials
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # Load GDS component
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")
    cell, layout = load_cell(file_gds, top_cell='sbend_dontfabme')
    component = load_component_from_tech(cell=cell, tech=technology)
    
    print(f"\n{'='*60}")
    print(f"PORT WIDTH CONVERGENCE STUDY")
    print(f"{'='*60}")
    print(f"Starting port width convergence study with widths: {port_widths}")
    print(f"Component: {cell.name}")
    print(f"Fixed depth: 1.5 μm")
    
    for i, port_width in enumerate(port_widths):
        print(f"\n--- Running simulation {i+1}/{len(port_widths)} with port_width = {port_width} ---")
        start_time = time.time()
        
        try:
            # Create Tidy3D solver with current port width
            solver = fdtd_solver_tidy3d(
                component=component,
                tech=technology,
                port_input=[component.ports[0]],  # Input port(s)
                visualize=False,  # Set to True to show simulation setup plots
                wavelength_start=1.5,
                wavelength_end=1.6,
                wavelength_points=50,  # Fewer points for faster simulation
                mesh=12,  # Fixed mesh size (from convergence results)
                boundary=["PML", "PML", "PML"],
                symmetry=[0, 0, 0],
                z_min=-1.0,
                z_max=1.11,
                width_ports=port_width,  # Current port width being tested
                depth_ports=1.5,  # Fixed port depth
                buffer=1.0,
                modes=[1],  # TE mode
                run_time_factor=50,
                field_monitors=None,
                working_dir=os.path.join(base_working_dir, "width_study", f"width{port_width}"),
            )
            
            # This requires Tidy3D cloud credentials and will consume simulation credits   
            solver.run()
            
            # Extract S-parameters
            wavelengths = solver.sparameters.wavelength
            
            # Get S21 (transmission from port 1 to port 2) and S11 (reflection at port 1)
            s21 = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)
            s11 = solver.sparameters.S(in_port=1, out_port=1, in_modeid=1, out_modeid=1)
            
            # Store results
            results['s21_data'].append(s21.s_mag)
            results['s11_data'].append(s11.s_mag)
            
            if results['wavelengths'] is None:
                results['wavelengths'] = wavelengths
                
            simulation_time = time.time() - start_time
            results['simulation_times'].append(simulation_time)
            
            print(f"Simulation completed in {simulation_time:.1f} seconds")
            print(f"S21 at 1.55μm: {np.interp(1.55, wavelengths, np.abs(s21.s_mag)):.4f}")
            print(f"S11 at 1.55μm: {np.interp(1.55, wavelengths, np.abs(s11.s_mag)):.4f}")
            
        except Exception as e:
            print(f"Error in simulation with port_width={port_width}: {str(e)}")
            # Store None values to maintain array consistency
            results['s21_data'].append(None)
            results['s11_data'].append(None)
            results['simulation_times'].append(None)
    
    return results

def run_port_depth_convergence_study(base_working_dir):
    """Run port depth convergence study."""
    
    # Port depths to test
    port_depths = [0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5]
    
    # Storage for results
    results = {
        'port_depths': port_depths,
        's21_data': [],  # S21 (transmission)
        's11_data': [],  # S11 (reflection)
        'wavelengths': None,
        'simulation_times': []
    }
    
    # Load technology file - need to modify for Tidy3D materials
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # Load GDS component
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")
    cell, layout = load_cell(file_gds, top_cell='sbend_dontfabme')
    component = load_component_from_tech(cell=cell, tech=technology)
    
    print(f"\n{'='*60}")
    print(f"PORT DEPTH CONVERGENCE STUDY")
    print(f"{'='*60}")
    print(f"Starting port depth convergence study with depths: {port_depths}")
    print(f"Component: {cell.name}")
    print(f"Fixed width: 2.5 μm")
    
    for i, port_depth in enumerate(port_depths):
        print(f"\n--- Running simulation {i+1}/{len(port_depths)} with port_depth = {port_depth} ---")
        start_time = time.time()
        
        try:
            # Create Tidy3D solver with current port depth
            solver = fdtd_solver_tidy3d(
                component=component,
                tech=technology,
                port_input=[component.ports[0]],  # Input port(s)
                visualize=False,  # Set to True to show simulation setup plots
                wavelength_start=1.5,
                wavelength_end=1.6,
                wavelength_points=50,  # Fewer points for faster simulation
                mesh=12,  # Fixed mesh size (from convergence results)
                boundary=["PML", "PML", "PML"],
                symmetry=[0, 0, 0],
                z_min=-1.0,
                z_max=1.11,
                width_ports=2.5,  # Fixed port width
                depth_ports=port_depth,  # Current port depth being tested
                buffer=1.0,
                modes=[1],  # TE mode
                run_time_factor=50,
                field_monitors=None,
                working_dir=os.path.join(base_working_dir, "depth_study", f"depth{port_depth}"),
            )
            
            # This requires Tidy3D cloud credentials and will consume simulation credits   
            solver.run()
            
            # Extract S-parameters
            wavelengths = solver.sparameters.wavelength
            
            # Get S21 (transmission from port 1 to port 2) and S11 (reflection at port 1)
            s21 = solver.sparameters.S(in_port=1, out_port=2, in_modeid=1, out_modeid=1)
            s11 = solver.sparameters.S(in_port=1, out_port=1, in_modeid=1, out_modeid=1)
            
            # Store results
            results['s21_data'].append(s21.s_mag)
            results['s11_data'].append(s11.s_mag)
            
            if results['wavelengths'] is None:
                results['wavelengths'] = wavelengths
                
            simulation_time = time.time() - start_time
            results['simulation_times'].append(simulation_time)
            
            print(f"Simulation completed in {simulation_time:.1f} seconds")
            print(f"S21 at 1.55μm: {np.interp(1.55, wavelengths, np.abs(s21.s_mag)):.4f}")
            print(f"S11 at 1.55μm: {np.interp(1.55, wavelengths, np.abs(s11.s_mag)):.4f}")
            
        except Exception as e:
            print(f"Error in simulation with port_depth={port_depth}: {str(e)}")
            # Store None values to maintain array consistency
            results['s21_data'].append(None)
            results['s11_data'].append(None)
            results['simulation_times'].append(None)
    
    return results

def plot_mesh_convergence_results(results, base_working_dir):
    """Plot mesh convergence results."""
    
    mesh_sizes = results['mesh_sizes']
    wavelengths = results['wavelengths']
    s21_data = results['s21_data']
    s11_data = results['s11_data']
    
    # Filter out None results (failed simulations)
    valid_indices = [i for i, s21 in enumerate(s21_data) if s21 is not None]
    valid_mesh_sizes = [mesh_sizes[i] for i in valid_indices]
    valid_s21_data = [s21_data[i] for i in valid_indices]
    valid_s11_data = [s11_data[i] for i in valid_indices]
    
    if not valid_indices:
        print("No valid simulation results to plot!")
        return
    
    # Find wavelength index closest to 1.55 μm for convergence plot
    target_wl = 1.55
    wl_idx = np.argmin(np.abs(wavelengths - target_wl))
    actual_wl = wavelengths[wl_idx]
    
    # Extract S-parameter values at target wavelength
    s21_at_target = [np.abs(s21[wl_idx]) for s21 in valid_s21_data]
    s11_at_target = [np.abs(s11[wl_idx]) for s11 in valid_s11_data]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: S21 vs wavelength for different mesh sizes
    ax1.set_prop_cycle('color', plt.cm.viridis(np.linspace(0, 1, len(valid_mesh_sizes))))
    for i, (mesh, s21) in enumerate(zip(valid_mesh_sizes, valid_s21_data)):
        ax1.plot(wavelengths, 10*np.log10(np.abs(s21)**2), 
                label=f'Mesh = {mesh}', linewidth=2)
    ax1.set_xlabel('Wavelength [μm]')
    ax1.set_ylabel('S21 Transmission [dB]')
    ax1.set_title('S21 vs Wavelength for Different Mesh Sizes')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: S11 vs wavelength for different mesh sizes
    ax2.set_prop_cycle('color', plt.cm.viridis(np.linspace(0, 1, len(valid_mesh_sizes))))
    for i, (mesh, s11) in enumerate(zip(valid_mesh_sizes, valid_s11_data)):
        ax2.plot(wavelengths, 10*np.log10(np.abs(s11)**2), 
                label=f'Mesh = {mesh}', linewidth=2)
    ax2.set_xlabel('Wavelength [μm]')
    ax2.set_ylabel('S11 Reflection [dB]')
    ax2.set_title('S11 vs Wavelength for Different Mesh Sizes')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Convergence of S21 at target wavelength
    ax3.plot(valid_mesh_sizes, s21_at_target, 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Mesh Size (cells per wavelength)')
    ax3.set_ylabel(f'|S21| at {actual_wl:.3f} μm')
    ax3.set_title(f'S21 Convergence at λ = {actual_wl:.3f} μm')
    ax3.grid(True, alpha=0.3)
    
    # Add convergence percentage text
    if len(s21_at_target) > 1:
        s21_change = abs(s21_at_target[-1] - s21_at_target[-2]) / s21_at_target[-1] * 100
        ax3.text(0.05, 0.95, f'Last change: {s21_change:.2f}%', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Convergence of S11 at target wavelength
    ax4.plot(valid_mesh_sizes, s11_at_target, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Mesh Size (cells per wavelength)')
    ax4.set_ylabel(f'|S11| at {actual_wl:.3f} μm')
    ax4.set_title(f'S11 Convergence at λ = {actual_wl:.3f} μm')
    ax4.grid(True, alpha=0.3)
    
    # Add convergence percentage text
    if len(s11_at_target) > 1:
        s11_change = abs(s11_at_target[-1] - s11_at_target[-2]) / s11_at_target[-1] * 100
        ax4.text(0.05, 0.95, f'Last change: {s11_change:.2f}%', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(base_working_dir, "mesh_convergence_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Mesh convergence plot saved to: {plot_path}")
    plt.show()
    
    # Print convergence summary
    print(f"\n--- Mesh Convergence Summary at λ = {actual_wl:.3f} μm ---")
    for i, mesh in enumerate(valid_mesh_sizes):
        print(f"Mesh {mesh:2d}: S21 = {s21_at_target[i]:.6f}, S11 = {s11_at_target[i]:.6f}")
        if i > 0:
            s21_change = abs(s21_at_target[i] - s21_at_target[i-1]) / s21_at_target[i] * 100
            s11_change = abs(s11_at_target[i] - s11_at_target[i-1]) / s11_at_target[i] * 100
            print(f"         Change: S21 = {s21_change:.2f}%, S11 = {s11_change:.2f}%")

def plot_port_width_convergence_results(results, base_working_dir):
    """Plot port width convergence results."""
    
    port_widths = results['port_widths']
    wavelengths = results['wavelengths']
    s21_data = results['s21_data']
    s11_data = results['s11_data']
    
    # Filter out None results (failed simulations)
    valid_indices = [i for i, s21 in enumerate(s21_data) if s21 is not None]
    valid_port_widths = [port_widths[i] for i in valid_indices]
    valid_s21_data = [s21_data[i] for i in valid_indices]
    valid_s11_data = [s11_data[i] for i in valid_indices]
    
    if not valid_indices:
        print("No valid simulation results to plot!")
        return
    
    # Find wavelength index closest to 1.55 μm for convergence plot
    target_wl = 1.55
    wl_idx = np.argmin(np.abs(wavelengths - target_wl))
    actual_wl = wavelengths[wl_idx]
    
    # Extract S-parameter values at target wavelength
    s21_at_target = [np.abs(s21[wl_idx]) for s21 in valid_s21_data]
    s11_at_target = [np.abs(s11[wl_idx]) for s11 in valid_s11_data]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: S21 vs wavelength for different port widths
    ax1.set_prop_cycle('color', plt.cm.plasma(np.linspace(0, 1, len(valid_port_widths))))
    for i, (width, s21) in enumerate(zip(valid_port_widths, valid_s21_data)):
        ax1.plot(wavelengths, 10*np.log10(np.abs(s21)**2), 
                label=f'Width = {width} μm', linewidth=2)
    ax1.set_xlabel('Wavelength [μm]')
    ax1.set_ylabel('S21 Transmission [dB]')
    ax1.set_title('S21 vs Wavelength for Different Port Widths')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: S11 vs wavelength for different port widths
    ax2.set_prop_cycle('color', plt.cm.plasma(np.linspace(0, 1, len(valid_port_widths))))
    for i, (width, s11) in enumerate(zip(valid_port_widths, valid_s11_data)):
        ax2.plot(wavelengths, 10*np.log10(np.abs(s11)**2), 
                label=f'Width = {width} μm', linewidth=2)
    ax2.set_xlabel('Wavelength [μm]')
    ax2.set_ylabel('S11 Reflection [dB]')
    ax2.set_title('S11 vs Wavelength for Different Port Widths')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Convergence of S21 at target wavelength
    ax3.plot(valid_port_widths, s21_at_target, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Port Width [μm]')
    ax3.set_ylabel(f'|S21| at {actual_wl:.3f} μm')
    ax3.set_title(f'S21 vs Port Width at λ = {actual_wl:.3f} μm')
    ax3.grid(True, alpha=0.3)
    
    # Add convergence percentage text
    if len(s21_at_target) > 1:
        s21_change = abs(s21_at_target[-1] - s21_at_target[-2]) / s21_at_target[-1] * 100
        ax3.text(0.05, 0.95, f'Last change: {s21_change:.2f}%', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 4: Convergence of S11 at target wavelength
    ax4.plot(valid_port_widths, s11_at_target, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Port Width [μm]')
    ax4.set_ylabel(f'|S11| at {actual_wl:.3f} μm')
    ax4.set_title(f'S11 vs Port Width at λ = {actual_wl:.3f} μm')
    ax4.grid(True, alpha=0.3)
    
    # Add convergence percentage text
    if len(s11_at_target) > 1:
        s11_change = abs(s11_at_target[-1] - s11_at_target[-2]) / s11_at_target[-1] * 100
        ax4.text(0.05, 0.95, f'Last change: {s11_change:.2f}%', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(base_working_dir, "port_width_convergence_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Port width convergence plot saved to: {plot_path}")
    plt.show()
    
    # Print convergence summary
    print(f"\n--- Port Width Convergence Summary at λ = {actual_wl:.3f} μm ---")
    for i, width in enumerate(valid_port_widths):
        print(f"Width {width:3.1f}: S21 = {s21_at_target[i]:.6f}, S11 = {s11_at_target[i]:.6f}")
        if i > 0:
            s21_change = abs(s21_at_target[i] - s21_at_target[i-1]) / s21_at_target[i] * 100
            s11_change = abs(s11_at_target[i] - s11_at_target[i-1]) / s11_at_target[i] * 100
            print(f"           Change: S21 = {s21_change:.2f}%, S11 = {s11_change:.2f}%")

def plot_port_depth_convergence_results(results, base_working_dir):
    """Plot port depth convergence results."""
    
    port_depths = results['port_depths']
    wavelengths = results['wavelengths']
    s21_data = results['s21_data']
    s11_data = results['s11_data']
    
    # Filter out None results (failed simulations)
    valid_indices = [i for i, s21 in enumerate(s21_data) if s21 is not None]
    valid_port_depths = [port_depths[i] for i in valid_indices]
    valid_s21_data = [s21_data[i] for i in valid_indices]
    valid_s11_data = [s11_data[i] for i in valid_indices]
    
    if not valid_indices:
        print("No valid simulation results to plot!")
        return
    
    # Find wavelength index closest to 1.55 μm for convergence plot
    target_wl = 1.55
    wl_idx = np.argmin(np.abs(wavelengths - target_wl))
    actual_wl = wavelengths[wl_idx]
    
    # Extract S-parameter values at target wavelength
    s21_at_target = [np.abs(s21[wl_idx]) for s21 in valid_s21_data]
    s11_at_target = [np.abs(s11[wl_idx]) for s11 in valid_s11_data]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: S21 vs wavelength for different port depths
    ax1.set_prop_cycle('color', plt.cm.inferno(np.linspace(0, 1, len(valid_port_depths))))
    for i, (depth, s21) in enumerate(zip(valid_port_depths, valid_s21_data)):
        ax1.plot(wavelengths, 10*np.log10(np.abs(s21)**2), 
                label=f'Depth = {depth} μm', linewidth=2)
    ax1.set_xlabel('Wavelength [μm]')
    ax1.set_ylabel('S21 Transmission [dB]')
    ax1.set_title('S21 vs Wavelength for Different Port Depths')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: S11 vs wavelength for different port depths
    ax2.set_prop_cycle('color', plt.cm.inferno(np.linspace(0, 1, len(valid_port_depths))))
    for i, (depth, s11) in enumerate(zip(valid_port_depths, valid_s11_data)):
        ax2.plot(wavelengths, 10*np.log10(np.abs(s11)**2), 
                label=f'Depth = {depth} μm', linewidth=2)
    ax2.set_xlabel('Wavelength [μm]')
    ax2.set_ylabel('S11 Reflection [dB]')
    ax2.set_title('S11 vs Wavelength for Different Port Depths')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Convergence of S21 at target wavelength
    ax3.plot(valid_port_depths, s21_at_target, 'co-', linewidth=2, markersize=8)
    ax3.set_xlabel('Port Depth [μm]')
    ax3.set_ylabel(f'|S21| at {actual_wl:.3f} μm')
    ax3.set_title(f'S21 vs Port Depth at λ = {actual_wl:.3f} μm')
    ax3.grid(True, alpha=0.3)
    
    # Add convergence percentage text
    if len(s21_at_target) > 1:
        s21_change = abs(s21_at_target[-1] - s21_at_target[-2]) / s21_at_target[-1] * 100
        ax3.text(0.05, 0.95, f'Last change: {s21_change:.2f}%', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # Plot 4: Convergence of S11 at target wavelength
    ax4.plot(valid_port_depths, s11_at_target, 'yo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Port Depth [μm]')
    ax4.set_ylabel(f'|S11| at {actual_wl:.3f} μm')
    ax4.set_title(f'S11 vs Port Depth at λ = {actual_wl:.3f} μm')
    ax4.grid(True, alpha=0.3)
    
    # Add convergence percentage text
    if len(s11_at_target) > 1:
        s11_change = abs(s11_at_target[-1] - s11_at_target[-2]) / s11_at_target[-1] * 100
        ax4.text(0.05, 0.95, f'Last change: {s11_change:.2f}%', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(base_working_dir, "port_depth_convergence_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Port depth convergence plot saved to: {plot_path}")
    plt.show()
    
    # Print convergence summary
    print(f"\n--- Port Depth Convergence Summary at λ = {actual_wl:.3f} μm ---")
    for i, depth in enumerate(valid_port_depths):
        print(f"Depth {depth:3.1f}: S21 = {s21_at_target[i]:.6f}, S11 = {s11_at_target[i]:.6f}")
        if i > 0:
            s21_change = abs(s21_at_target[i] - s21_at_target[i-1]) / s21_at_target[i] * 100
            s11_change = abs(s11_at_target[i] - s11_at_target[i-1]) / s11_at_target[i] * 100
            print(f"           Change: S21 = {s21_change:.2f}%, S11 = {s11_change:.2f}%")

def print_timing_summary(results, study_name):
    """Print timing summary for a convergence study."""
    valid_times = [t for t in results['simulation_times'] if t is not None]
    if valid_times:
        print(f"\n--- {study_name} Timing Summary ---")
        print(f"Total simulation time: {sum(valid_times):.1f} seconds")
        print(f"Average time per simulation: {np.mean(valid_times):.1f} seconds")
        print(f"Fastest simulation: {min(valid_times):.1f} seconds")
        print(f"Slowest simulation: {max(valid_times):.1f} seconds")

if __name__ == "__main__":
    # Create timestamped working directory
    base_working_dir = create_working_directory()
    print(f"All results will be saved to: {base_working_dir}")
    
    # Run all convergence studies
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE CONVERGENCE STUDY")
    print(f"{'='*80}")
    
    # 1. Mesh convergence study
    mesh_results = run_mesh_convergence_study(base_working_dir)
    
    # Save mesh results
    mesh_results_file = os.path.join(base_working_dir, "mesh_convergence_results.pkl")
    with open(mesh_results_file, 'wb') as f:
        pickle.dump(mesh_results, f)
    print(f"Mesh results saved to {mesh_results_file}")
    
    # Plot mesh results
    plot_mesh_convergence_results(mesh_results, base_working_dir)
    print_timing_summary(mesh_results, "Mesh Convergence")
    
    # 2. Port width convergence study
    width_results = run_port_width_convergence_study(base_working_dir)
    
    # Save width results
    width_results_file = os.path.join(base_working_dir, "port_width_convergence_results.pkl")
    with open(width_results_file, 'wb') as f:
        pickle.dump(width_results, f)
    print(f"Port width results saved to {width_results_file}")
    
    # Plot width results
    plot_port_width_convergence_results(width_results, base_working_dir)
    print_timing_summary(width_results, "Port Width Convergence")
    
    # 3. Port depth convergence study
    depth_results = run_port_depth_convergence_study(base_working_dir)
    
    # Save depth results
    depth_results_file = os.path.join(base_working_dir, "port_depth_convergence_results.pkl")
    with open(depth_results_file, 'wb') as f:
        pickle.dump(depth_results, f)
    print(f"Port depth results saved to {depth_results_file}")
    
    # Plot depth results
    plot_port_depth_convergence_results(depth_results, base_working_dir)
    print_timing_summary(depth_results, "Port Depth Convergence")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"CONVERGENCE STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {base_working_dir}")
    print(f"Generated plots:")
    print(f"  - mesh_convergence_plot.png")
    print(f"  - port_width_convergence_plot.png") 
    print(f"  - port_depth_convergence_plot.png")
    print(f"Generated data files:")
    print(f"  - mesh_convergence_results.pkl")
    print(f"  - port_width_convergence_results.pkl")
    print(f"  - port_depth_convergence_results.pkl")
    
                                                                                                                                                                  # %%
