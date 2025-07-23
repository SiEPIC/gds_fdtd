# -*- coding: utf-8 -*-
"""
gds_fdtd simulation toolbox.

S-parameter module.
@author: Mustafa Hammood, 2025
"""

import numpy as np
import logging
import re
import os
import matplotlib.pyplot as plt

logger = logging.getLogger("dreamcompiler")


class s:
    """Component's single in-out s-parameter dataset class."""

    def __init__(
        self,
        f: list,
        s_mag: list,
        s_phase: list,
        in_port: int = 1,
        out_port: int = 1,
        mode_label: int = 1,
        in_modeid: int = 1,
        out_modeid: int = 1,
        data_type: int = 1,
        group_delay: float = 0.0,
    ):
        self.in_port = in_port
        self.out_port = out_port
        self.mode_label = mode_label
        self.in_modeid = in_modeid
        self.out_modeid = out_modeid
        self.data_type = data_type
        self.group_delay = group_delay
        self.f = f
        self.s_mag = s_mag
        self.s_phase = s_phase
        return

    @property
    def wavl(self):
        c = 299792458
        return c / np.array(self.f)

    @property
    def idn_ports(self):
        in_idn = "".join(char for char in self.in_port if char.isdigit())
        out_idn = "".join(char for char in self.out_port if char.isdigit())
        return f"{out_idn}{in_idn}"

    @property
    def idn_modes(self):
        in_idn = "".join(char for char in self.in_modeid if char.isdigit())
        out_idn = "".join(char for char in self.out_modeid if char.isdigit())
        return f"{out_idn}{in_idn}"

    @property
    def idn(self):
        return f"{self.idn_ports}_{self.idn_modes}"

    def plot(self):
        c = 299792458
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("Wavelength [microns]")
        ax.set_ylabel("Transmission [dB]")

        mag = [10 * np.log10(abs(i) ** 2) for i in self.s_mag]
        phase = [np.angle(i) ** 2 for i in self.s_phase]
        ax.plot(1e6 * c / np.array(self.f), mag, label="S_" + self.idn)
        ax.legend()
        return fig, ax


class port:
    """Component port abstraction class."""

    def __init__(self, name, direction):
        self.name = name
        self.direction = direction


class sparameters:
    """Component s-parameter abstraction class."""

    def __init__(self, name: str):
        """Component s-parameter abstraction class.

        Args:
            name (_type_): _description_
        """
        self.name = name
        self.ports = []
        self.data = []
        return

    def add_port(self, port_name: str, port_direction: str):
        """
        Add a port to the component s-parameters.

        Parameters
        ----------
        port_name : string
            Port's name.
        port_direction : string
            Port direction (LEFT, RIGHT, TOP, BOTTOM).

        Returns
        -------
        None.

        """
        self.ports.append(port(port_name, port_direction))

    def add_data(
        self,
        in_port: str,
        out_port: str,
        mode_label: int,
        in_modeid: int,
        out_modeid: int,
        data_type: str,
        group_delay: float,
        f: list,
        s_mag: list,
        s_phase: list,
    ):
        """
        Add an S-parameter dataset.

        Parameters
        ----------
        in_port : string
            Input port name.
        out_port : string
            Output port name.
        mode_label : string
            Mode label.
        in_modeid : int
            Input mode ID (or othogonal identifier).
        out_modeid : int
            Output mode ID (or othogonal identifier).
        data_type : string
            S-parameter data type. Typically "Transmission" unless "Modulation"
        group_delay : float
            Isolated group delay offset.
        f : List of floats
            Frequency data points.
        s_mag : List of floats
            S magnitude data points.
        s_phase : List of floats
            S phase data points.

        Returns
        -------
        None.

        """
        data = s(
            in_port=in_port,
            out_port=out_port,
            mode_label=mode_label,
            in_modeid=in_modeid,
            out_modeid=out_modeid,
            data_type=data_type,
            group_delay=group_delay,
            f=f,
            s_mag=s_mag,
            s_phase=s_phase,
        )
        self.data.append(data)

    def S(
        self,
        in_port: int = 1,
        out_port: int = 1,
        in_modeid: int = 1,
        out_modeid: int = 1,
    ) -> s:
        """fetches the specified S parameter entry

        Args:
            in_port (int, optional): input port index. Defaults to 1.
            out_port (int, optional): output port index. Defaults to 1.
            in_modeid (int, optional): input mode index. Defaults to 1.
            out_modeid (int, optional): output mode index. Defaults to 1.

        Returns:
            s: s_parameter entry
        """
        for d in self.data:
            if d.idn == f"{out_port}{in_port}_{out_modeid}{in_modeid}":
                return d
        logger.warning("Cannot find specified S-parameter entry.")

    def plot(self, plot_type: str = "log"):
        valid_plots = ["log", "phase", "linear"]
        if plot_type not in valid_plots:
            logging.warning(
                f"Not a valid plot type. Options: {valid_plots}. defaulting to: {valid_plots[0]}"
            )
            plot_type = valid_plots[0]

        if self.data:
            c = 299792458
            fig, ax = plt.subplots()
            legends = []
            for data in self.data:
                label = f"S{data.idn}"
                if plot_type == "log":
                    ax.plot(
                        c * 1e9 / np.array(data.f),
                        10 * np.log10(np.array(data.s_mag) ** 2),
                        label=label,
                    )
                    ax.set_ylabel("Transmission [dB]")
                elif plot_type == "linear":
                    ax.plot(c * 1e9 / np.array(data.f), data.s_mag, label=label)
                    ax.set_ylabel("Magnitude [normalized]")
                elif plot_type == "phase":
                    ax.plot(c * 1e9 / np.array(data.f), data.s_phase, label=label)
                    ax.set_ylabel("Phase [rad]")
                legends.append(label)
            ax.set_xlabel("Wavelength [nm]")
            ax.set_title(f"{self.name} S-Parameters")

            ax.legend(legends, loc="upper left", bbox_to_anchor=(1, 1))
            fig.show()
        else:
            logging.error("No valid data to visualize")


def process_dat(file_path: str, name: str | None = None, verbose: bool = True):
    """
    Process a .dat s-parameters file into a sparameters object.

    Parameters
    ----------
    file_path : string
        File path containing the s-parameters data.
    verbose : Boolean, optional
        Logging flag. The default is True.

    Returns
    -------
    sparams : dpcmgenerator sparameters object.
        Parsed sparameters object.

    """
    if not name:
        name = os.path.basename(file_path)
    spar = sparameters(name=name)
    port_pattern = re.compile(r'\["(.*?)","(.*?)"\]')
    data_pattern = re.compile(r'\("(.*?)","(.*?)",(\d+),"(.+?)",(\d+),"(.+?)",?(.*?)\)')

    with open(file_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            port_match = port_pattern.match(line)
            data_match = data_pattern.match(line)
            if port_match:
                # find the available ports from the file headers
                port_name, port_direction = port_match.groups()
                if verbose:
                    logger.debug(f"Found port: name={port_name} , direction={port_direction}")
                spar.add_port(port_name, port_direction)
            elif data_match:
                # parse an S-parameter dataset header
                (
                    out_port,
                    mode_label,
                    out_modeid,
                    in_port,
                    in_modeid,
                    data_type,
                    group_delay,
                ) = data_match.groups()
                
                # Handle missing group_delay
                if not group_delay:
                    group_delay = "0.0"
                
                if verbose:
                    logger.debug(
                        f"Found S-param dataset: out_port={out_port}, mode_label={mode_label}, "
                        f"out_modeid={out_modeid}, in_port={in_port}, in_modeid={in_modeid}, "
                        f"data_type={data_type}, group_delay={float(group_delay):.2e}"
                    )

                # parse the data set
                i += 1
                num_points, _ = map(int, lines[i].strip().strip("()").split(","))
                freq_data = []
                for _ in range(num_points):
                    i += 1
                    f, s_mag, s_phase = map(float, lines[i].strip().split())
                    freq_data.append((f, s_mag, s_phase))
                
                f = [i[0] for i in freq_data]  # first column in dat
                s_mag = [i[1] for i in freq_data]  # second column in dat
                s_phase = [i[2] for i in freq_data]  # third column in dat
                
                spar.add_data(
                    in_port,
                    out_port,
                    mode_label,
                    in_modeid,
                    out_modeid,
                    data_type,
                    group_delay,
                    f,
                    s_mag,
                    s_phase,
                )

            i += 1
    return spar

class s_paramet_writer():
    """Object that writes data to Lumerical INTERCONNECT S-parameters .dat format
    """
    def __init__(self):
        self.name = 'sparams'
        self.wavl = [1500e-9, 1600e-9, 20e-9] # start, stop, resolution
        self.n_ports = 2
        self.data = 0
        self.encoding = 'utf-8'

    def make_file(self):
        f = open(self.name+'.dat',"wb")
        return f

    def npoints(self):
        self.nPts = int((self.wavl[1]-self.wavl[0])/self.wavl[2] +1)
        return self.nPts

    def get_index(self, idx):
        return str(int(idx/self.n_ports)+1)+str(idx%self.n_ports+1)

    def visualize(self):
        wavelength = np.linspace(self.wavl[0], self.wavl[1], self.npoints())
        for idx, S in enumerate(self.data):
            txt = 'S'+self.get_index(idx)
            plt.plot(wavelength*1e9, 10*np.log10(S[0]), label = txt)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Transmission [dB]')
        plt.tight_layout()
        plt.legend()
        return 0

    def write_header(self, file):
        for i in range(self.n_ports):
            text = bytes('["port %d",""]\n' % (i+1), encoding = self.encoding)
            file.write(text)
        return 0

    def write_S_header(self, file, S):
        port1 = S[0]; port2 = S[1]; OID1 = S[2]; OID2 = S[3]
        text = bytes('("port %d","mode 1",%d,"port %d",%d,"transmission")\n' % (port1, OID1, port2, OID2), encoding = 'utf-8')
        file.write(text)
        text = bytes('(%d, 3)\n' % self.nPts, encoding = self.encoding)
        file.write(text)
        return 0

    def write_S_data(self, file, data):
        c = 299792458 #m/s
        wavelength = np.linspace(self.wavl[0], self.wavl[1], self.nPts)
        freq = c/wavelength; freq = np.flip(freq)
        for idx, i in enumerate(freq):
            text = bytes('%d %f %f\n' % (i, data[0][idx], data[1][idx]), encoding = self.encoding)
            file.write(text)
        return 0

    def write_S(self):
        self.npoints()
        f = self.make_file()
        self.write_header(f)
        idx = 0
        for i in range(self.n_ports):
            for k in range(self.n_ports):
                S = [i+1,k+1,1,1] # out_port, in_port2, out_OID1, in_OID2
                self.write_S_header(f, S)
                self.write_S_data(f, self.data[idx])
                idx+=1
        f.close()
        return 0