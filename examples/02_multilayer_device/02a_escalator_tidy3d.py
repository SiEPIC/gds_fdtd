# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mustafa Hammood
"""
import tidy3d as td
import gds_fdtd as gtd
import os

# Define the path to the GDS file
file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")

# define 3D geometry paramters not captured in the 2D layout
thickness_si = 0.22
thickness_sin = 0.4
gap_si_sin = 0.1
thickness_sub = 2
thickness_super = 3
sidewall_angle = 88  # sidewall angle of the structures

# define simulation parameters
z_span = 3  # Z-span of the simulation

# define materials structures
mat_si = td.material_library["cSi"]["Li1993_293K"]
mat_sin = td.material_library["Si3N4"]["Philipp1973Sellmeier"]
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

# frequency and bandwidth of pulsed excitation
in_pol = "TE"  # input polarization state (options are TE, TM, TETM)
wavl_min = 1.45  # simulation wavelength start (microns)
wavl_max = 1.65  # simulation wavelength end (microns)
wavl_pts = 101

# define symmetry across Z axis (TE mode) - set to -1 for anti symmetric
symmetry = (0, 0, 0)

# %% load and process the layout file
layout = gtd.lyprocessor.load_layout(file_gds, top_cell="si_sin_escalator_te1550")

# load all the ports in the device and (optional) initialize each to have a center
ports_si = gtd.lyprocessor.load_ports(layout, layer=[1, 10])

ports_sin = gtd.lyprocessor.load_ports(layout, layer=[1, 11])

# load the device simulation region
bounds = gtd.lyprocessor.load_region(
    layout, layer=[68, 0], z_center=thickness_si / 2, z_span=z_span
)

# load the silicon structures in the device in layer (1,0)
device_si = gtd.lyprocessor.load_structure(
    layout,
    name="Si",
    layer=[1, 0],
    z_base=0,
    z_span=thickness_si,
    material=mat_si
)

device_sin = gtd.lyprocessor.load_structure(
    layout,
    name="SiN",
    layer=[1, 5],
    z_base=thickness_si + gap_si_sin,
    z_span=thickness_sin,
    material=mat_sin,
)

# make the superstrate and substrate based on device bounds
# this information isn't typically captured in a 2D layer stack
device_super = gtd.lyprocessor.load_structure_from_bounds(
    bounds, name="Superstrate", z_base=0, z_span=thickness_super, material=mat_super
)
device_sub = gtd.lyprocessor.load_structure_from_bounds(
    bounds, name="Substrate", z_base=0, z_span=-thickness_sub, material=mat_sub
)

# create the device by loading the structures
device = gtd.core.component(
    name=layout.name,
    structures=[device_sub, device_super, device_si, device_sin],
    ports=ports_si + ports_sin,
    bounds=bounds,
)

# %%
simulation = gtd.simprocessor.make_sim(
    device=device,
    wavl_min=wavl_min,
    wavl_max=wavl_max,
    wavl_pts=wavl_pts,
    symmetry=symmetry,
    z_span=z_span,
    field_monitor_axis="y",
)
# %% upload and run the simulation
# create job, upload sim to server to begin running
simulation.upload()
# %% run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
simulation.execute()
# %% visualize the results
simulation.visualize_results()
