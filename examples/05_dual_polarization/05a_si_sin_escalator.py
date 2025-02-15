# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mustafa Hammood
Dual polarization simulation.
"""
import gds_fdtd as gtd
import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import os

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech.yaml")
    technology = gtd.core.parse_yaml_tech(tech_path)

    # Define the path to the GDS file
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")

    layout = gtd.lyprocessor.load_layout(file_gds, top_cell='si_sin_escalator_te1550')

    simulation = gtd.simprocessor.build_sim_from_tech(
        tech=technology,
        layout=layout,
        in_port=0,
        wavl_min=1.5,
        wavl_max=1.6,
        wavl_pts=101,
        symmetry=(
            0,
            0,
            0,
        ),  # ensure structure is symmetric across symmetry axis before triggering this!
        z_span=4,
        mode_index=[0,1],
        num_modes=2,
    )

    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    simulation.execute()


    #  visualize the results
    simulation.visualize_results()

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Wavelength [microns]")
    ax.set_ylabel("Transmission [dB]")
    te_mode = simulation.s_parameters.entries_in_mode(mode_in=0, mode_out=0)
    tm_mode = simulation.s_parameters.entries_in_mode(mode_in=1, mode_out=1)

    idx=1
    mag_te = [10 * np.log10(abs(s_value) ** 2) for s_value in te_mode[idx].s]
    ax.plot(td.C_0 / te_mode[idx].freq, mag_te, label=f"{te_mode[idx].label} (TE-TE)")

    mag_tm = [10 * np.log10(abs(s_value) ** 2) for s_value in tm_mode[idx].s]
    ax.plot(td.C_0 / tm_mode[idx].freq, mag_tm, label=f"{tm_mode[idx].label} (TM-TM)")

    ax.legend()
# %%
