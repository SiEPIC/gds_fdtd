"""
gds_fdtd simulation toolbox.

Tidy3d interface module.
@author: Mustafa Hammood, 2025
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from .core import s_parameters, sparam, port

class sim_tidy3d:
    def __init__(
        self, in_port, device, wavl_min=1.45, wavl_max=1.65, wavl_pts=101, sim_jobs=None
    ):
        self.in_port = in_port
        self.device = device
        self.wavl_min = wavl_min
        self.wavl_max = wavl_max
        self.wavl_pts = wavl_pts
        self.sim_jobs = sim_jobs
        self.results = None

    def upload(self):
        from tidy3d import web

        # divide between job and sim, how to attach them?
        for sim_job in self.sim_jobs:
            sim = sim_job["sim"]
            name = sim_job["name"]
            sim_job["job"] = web.Job(simulation=sim, task_name=name)

    def execute(self):

        def get_directions(ports):
            directions = []
            for p in ports:
                if p.direction in [0, 90]:
                    directions.append("+")
                else:
                    directions.append("-")
            return tuple(directions)

        def get_source_direction(port):
            if port.direction in [0, 90]:
                return "-"
            else:
                return "+"

        def get_port_name(port):
            return [int(i) for i in port if i.isdigit()][0]

        def measure_transmission(in_port: port, in_mode_idx: int, out_mode_idx: int):
            """
            Constructs a "row" of the scattering matrix.
            """
            num_ports = np.size(self.device.ports)

            if isinstance(self.results, list):
                if len(self.results) == 1:
                    results = self.results[0]
                else:
                    # TBD: Handle the case where self.results is a list with more than one item
                    logging.warning(
                        "Multiple results handler is WIP, using first results entry"
                    )
                    results = self.results[0]
            else:
                results = self.results

            input_amp = results[in_port.name].amps.sel(
                direction=get_source_direction(in_port),
                mode_index=in_mode_idx,
            )
            amps = np.zeros((num_ports, self.wavl_pts), dtype=complex)
            directions = get_directions(self.device.ports)
            for i, (monitor, direction) in enumerate(
                zip(results.simulation.monitors[:num_ports], directions)
            ):
                amp = results[monitor.name].amps.sel(
                    direction=direction, mode_index=out_mode_idx
                )
                amp_normalized = amp / input_amp
                amps[i] = np.squeeze(amp_normalized.values)

            return amps

        self.s_parameters = s_parameters()  # initialize empty s parameters

        self.results = []
        for sim_job in self.sim_jobs:
            if not os.path.exists(self.device.name):
                os.makedirs(self.device.name)
            self.results.append(
                sim_job["job"].run(
                    path=os.path.join(self.device.name, f"{sim_job['name']}.hdf5")
                )
            )
            for mode in range(sim_job["num_modes"]):
                amps_arms = measure_transmission(
                    in_port=sim_job["in_port"],
                    in_mode_idx=sim_job["source"].mode_index,
                    out_mode_idx=mode,
                )

                logging.info("Mode amplitudes in each port: \n")
                wavl = np.linspace(self.wavl_min, self.wavl_max, self.wavl_pts)
                for amp, monitor in zip(
                    amps_arms, self.results[-1].simulation.monitors
                ):
                    logging.info(f'\tmonitor     = "{monitor.name}"')
                    logging.info(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
                    logging.info(
                        f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n"
                    )

                    self.s_parameters.add_param(
                        sparam(
                            idx_in=sim_job["in_port"].idx,
                            idx_out=get_port_name(monitor.name),
                            mode_in=sim_job["source"].mode_index,
                            mode_out=mode,
                            freq=td.C_0 / (wavl),
                            s=amp,
                        )
                    )
        if isinstance(self.results, list) and len(self.results) == 1:
            self.results = self.results[0]

    def visualize_results(self):
        self.s_parameters.plot()

        try:
            if isinstance(self.results, list):
                for job_result in self.results:
                    fig, ax = plt.subplots(1, 1, figsize=(16, 3))
                    job_result.plot_field(
                        "field",
                        "Ey",
                        freq=td.C_0 / ((self.wavl_max + self.wavl_min) / 2),
                        ax=ax,
                    )
                    fig.show()
            else:
                fig, ax = plt.subplots(1, 1, figsize=(16, 3))
                self.results.plot_field(
                    "field",
                    "Ey",
                    freq=td.C_0 / ((self.wavl_max + self.wavl_min) / 2),
                    ax=ax,
                )
                fig.show()
        except:
            return