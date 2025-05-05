# %%
"""
Dual polarization simulation of a Bragg grating device.
@author: Mustafa Hammood
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from gds_fdtd.core import parse_yaml_tech
from gds_fdtd.simprocessor import make_t3d_sim, load_component_from_tech
from gds_fdtd.lyprocessor import load_cell

c0_um = 299792458000000.0
if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech_tidy3d.yaml")
    technology = parse_yaml_tech(tech_path)

    # Define the path to the GDS file
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")

    cell, layout = load_cell(file_gds, top_cell='bragg_te1550')

    device = load_component_from_tech(cell=cell, tech=technology, z_span=5)

    simulation = make_t3d_sim(
        device=device,
        in_port=device.ports[0],
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
    ax.plot(c0_um / te_mode[idx].freq, mag_te, label=f"{te_mode[idx].label} (TE-TE)")

    mag_tm = [10 * np.log10(abs(s_value) ** 2) for s_value in tm_mode[idx].s]
    ax.plot(c0_um / tm_mode[idx].freq, mag_tm, label=f"{tm_mode[idx].label} (TM-TM)")

    ax.legend()
# %%
