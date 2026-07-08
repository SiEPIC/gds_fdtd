"""
gds_fdtd simulation toolbox.

S-parameter module.
@author: Mustafa Hammood, 2025
"""

import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _number_of(value) -> int:
    """Derive a port/mode number from an int or a name with trailing digits.

    "opt1" -> 1, "port 12" -> 12, 3 -> 3. Trailing digits only — never digit
    concatenation, which made port indices >= 10 ambiguous (B6).
    """
    if isinstance(value, int):
        return value
    m = re.search(r"(\d+)\s*$", str(value).strip())
    if m is None:
        raise ValueError(f"Cannot derive a port/mode number from {value!r}")
    return int(m.group(1))


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
    def in_port_num(self) -> int:
        return _number_of(self.in_port)

    @property
    def out_port_num(self) -> int:
        return _number_of(self.out_port)

    @property
    def in_mode_num(self) -> int:
        return _number_of(self.in_modeid)

    @property
    def out_mode_num(self) -> int:
        return _number_of(self.out_modeid)

    @property
    def idn_ports(self):
        return f"{self.out_port_num}{self.in_port_num}"

    @property
    def idn_modes(self):
        return f"{self.out_mode_num}{self.in_mode_num}"

    @property
    def idn(self):
        return f"{self.idn_ports}_{self.idn_modes}"

    def plot(self):
        c = 299792458
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel("Wavelength [microns]")
        ax.set_ylabel("Transmission [dB]")

        mag = [10 * np.log10(abs(i) ** 2) for i in self.s_mag]
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
            Input mode ID (or orthogonal identifier).
        out_modeid : int
            Output mode ID (or orthogonal identifier).
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

    @property
    def wavelength(self):
        c = 299792458
        # fetch wavelength from first available s parameter
        for d in self.data:
            return 1e6 * c / np.array(d.f)  # convert to microns

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
            if (
                d.in_port_num == _number_of(in_port)
                and d.out_port_num == _number_of(out_port)
                and d.in_mode_num == _number_of(in_modeid)
                and d.out_mode_num == _number_of(out_modeid)
            ):
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

    def to_smatrix(self, name: str | None = None):
        """Convert to the canonical gds_fdtd.smatrix.SMatrix (WP2.4b).

        Magnitude/phase entries become complex amplitudes; port names are
        carried through as-is; unmeasured paths stay NaN in the result.
        """
        import numpy as np

        from .smatrix import SMatrix

        entries = []
        for d in self.data:
            s_complex = np.asarray(d.s_mag, dtype=float) * np.exp(
                1j * np.asarray(d.s_phase, dtype=float)
            )
            entries.append(
                (str(d.in_port), str(d.out_port), d.in_mode_num, d.out_mode_num, d.f, s_complex)
            )
        return SMatrix.from_entries(entries, name=name or self.name)

    def analyze_excitations(self, threshold: float = 1e-10, verbose: bool = True):
        """
        Analyze S-parameter data to identify zero vs non-zero entries.

        Parameters
        ----------
        threshold : float, optional
            Magnitude threshold below which entries are considered zero. Default is 1e-10.
        verbose : bool, optional
            Whether to print detailed analysis. Default is True.

        Returns
        -------
        dict
            Dictionary containing analysis results with keys:
            'zero_entries', 'non_zero_entries', 'input_port_summary'
        """
        zero_entries = []
        non_zero_entries = []
        input_port_groups = {}

        # Categorize entries
        for data in self.data:
            # Group by input port
            if data.in_port not in input_port_groups:
                input_port_groups[data.in_port] = {"zero": [], "non_zero": []}

            # Check if entry is effectively zero
            max_mag = max(abs(mag) for mag in data.s_mag) if data.s_mag else 0
            is_zero = max_mag < threshold

            if is_zero:
                zero_entries.append(data)
                input_port_groups[data.in_port]["zero"].append(data)
            else:
                non_zero_entries.append(data)
                input_port_groups[data.in_port]["non_zero"].append(data)

        if verbose:
            print(f"=== S-Parameter Excitation Analysis for {self.name} ===")
            print(f"Total entries: {len(self.data)}")
            print(f"Non-zero entries: {len(non_zero_entries)}")
            print(f"Zero entries: {len(zero_entries)}")

            print("\nBy Input Port:")
            for input_port in sorted(input_port_groups.keys()):
                zero_count = len(input_port_groups[input_port]["zero"])
                non_zero_count = len(input_port_groups[input_port]["non_zero"])
                total_count = zero_count + non_zero_count

                print(
                    f"  {input_port}: {total_count} total ({non_zero_count} non-zero, {zero_count} zero)"
                )

                if zero_count > 0:
                    zero_idns = [d.idn for d in input_port_groups[input_port]["zero"]]
                    print(
                        f"    ⚠️  WARNING: {zero_count} zero entries found for input port {input_port}"
                    )
                    if zero_count <= 8:  # Show all if not too many
                        print(f"         Zero IDNs: {sorted(zero_idns)}")
                    else:  # Show sample if many
                        print(f"         Sample zero IDNs: {sorted(zero_idns)[:8]}...")

        return {
            "zero_entries": zero_entries,
            "non_zero_entries": non_zero_entries,
            "input_port_summary": input_port_groups,
        }

    def get_expected_excitations(self, excited_ports: list, excited_modes: list):
        """
        Get the expected S-parameter IDNs if only specific ports and modes were excited.

        Parameters
        ----------
        excited_ports : list
            List of port names that should be excited (e.g., ['opt1'])
        excited_modes : list
            List of mode IDs that should be excited (e.g., [1, 2])

        Returns
        -------
        list
            List of expected S-parameter IDNs
        """
        expected_idns = []

        # Get all unique output ports from the data
        output_ports = sorted({data.out_port for data in self.data})
        # Get all unique output modes from the data
        output_modes = sorted({data.out_modeid for data in self.data})

        for excited_port in excited_ports:
            for excited_mode in excited_modes:
                for output_port in output_ports:
                    for output_mode in output_modes:
                        # Extract port numbers for IDN
                        in_port_num = "".join(char for char in str(excited_port) if char.isdigit())
                        out_port_num = "".join(char for char in str(output_port) if char.isdigit())
                        idn = f"{out_port_num}{in_port_num}_{output_mode}{excited_mode}"
                        expected_idns.append(idn)

        return expected_idns

    def validate_excitations(
        self,
        expected_excited_ports: list,
        expected_excited_modes: list,
        threshold: float = 1e-10,
        verbose: bool = True,
    ):
        """
        Validate which ports and modes were actually excited vs expected.

        Parameters
        ----------
        expected_excited_ports : list
            List of port names that should have been excited (e.g., ['opt1'])
        expected_excited_modes : list
            List of mode IDs that should have been excited (e.g., [1, 2])
        threshold : float, optional
            Magnitude threshold for considering entries as zero. Default is 1e-10.
        verbose : bool, optional
            Whether to print detailed validation results. Default is True.

        Returns
        -------
        dict
            Validation results dictionary
        """
        analysis = self.analyze_excitations(threshold=threshold, verbose=False)
        expected_idns = self.get_expected_excitations(
            expected_excited_ports, expected_excited_modes
        )

        actual_non_zero_idns = [data.idn for data in analysis["non_zero_entries"]]
        actual_zero_idns = [data.idn for data in analysis["zero_entries"]]

        # Categorize results
        expected_and_found = [idn for idn in expected_idns if idn in actual_non_zero_idns]
        expected_but_zero = [idn for idn in expected_idns if idn in actual_zero_idns]
        unexpected_non_zero = [idn for idn in actual_non_zero_idns if idn not in expected_idns]

        if verbose:
            print(f"=== Excitation Validation for {self.name} ===")
            print(
                f"Expected excitations: {expected_excited_ports} with modes {expected_excited_modes}"
            )
            print(f"Expected IDNs: {len(expected_idns)}")
            print(f"Actual non-zero IDNs: {len(actual_non_zero_idns)}")

            print(f"\nExpected and found non-zero: {len(expected_and_found)}")
            if expected_and_found and len(expected_and_found) <= 16:
                print(f"   {sorted(expected_and_found)}")
            elif expected_and_found:
                print(f"   Sample: {sorted(expected_and_found)[:8]}...")

            print(f"\n❌ Expected non-zero but found zero: {len(expected_but_zero)}")
            if expected_but_zero and len(expected_but_zero) <= 16:
                print(f"   {sorted(expected_but_zero)}")
            elif expected_but_zero:
                print(f"   Sample: {sorted(expected_but_zero)[:8]}...")

            print(f"\n⚠️  Unexpected non-zero entries: {len(unexpected_non_zero)}")
            if unexpected_non_zero and len(unexpected_non_zero) <= 16:
                print(f"   {sorted(unexpected_non_zero)}")
            elif unexpected_non_zero:
                print(f"   Sample: {sorted(unexpected_non_zero)[:8]}...")

            if unexpected_non_zero:
                print("\n⚠️  WARNING: Found excitations at unexpected ports/modes!")
                print("   This suggests the simulation included more inputs than expected.")
            else:
                print("\nAll non-zero entries match expected excitations.")

        return {
            "expected_idns": expected_idns,
            "expected_and_found": expected_and_found,
            "expected_but_zero": expected_but_zero,
            "unexpected_non_zero": unexpected_non_zero,
            "total_expected": len(expected_idns),
            "total_actual_non_zero": len(actual_non_zero_idns),
        }


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
    data_pattern = re.compile(r'\("(.*?)","(.*?)",(\d+),"(.+?)",(\d+),"(.+?)"\)')

    with open(file_path) as f:
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
                # Format: ("output_port", "output_mode_name", output_mode_id, "input_port", input_mode_id, "data_type")
                (
                    output_port,  # Port where measurement is taken
                    output_mode_name,  # Mode name at output port (e.g., "mode 1")
                    output_modeid,  # Mode ID at output port
                    input_port,  # Port where excitation is applied
                    input_modeid,  # Mode ID at input port
                    data_type,  # Data type (e.g., "transmission")
                ) = data_match.groups()

                if verbose:
                    logger.debug(
                        f"Found S-param dataset: output_port={output_port}, output_mode_name={output_mode_name}, "
                        f"output_modeid={output_modeid}, input_port={input_port}, input_modeid={input_modeid}, "
                        f"data_type={data_type}"
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
                    input_port,  # Input port (excitation)
                    output_port,  # Output port (measurement)
                    output_mode_name,  # Mode label
                    int(input_modeid),  # Convert to integer
                    int(output_modeid),  # Convert to integer
                    data_type,
                    0.0,  # Group delay set to 0.0 (not provided in this format)
                    f,
                    s_mag,
                    s_phase,
                )

            i += 1
    return spar


def write_dat(spar: sparameters, file_path: str) -> str:
    """Write a sparameters object to Lumerical INTERCONNECT .dat format.

    One header + data block is written per entry, carrying that entry's ACTUAL
    in/out port and mode ids and its own frequency vector. (The previous writer
    class assumed a dense n_ports x n_ports ordering and regenerated frequencies
    from a wavelength range, mislabeling ports whenever the entry order differed
    or the matrix was partial; bug B14.)

    The format is the same one process_dat() parses; write_dat -> process_dat is
    a lossless round trip (frequencies are written as full-precision floats).

    Args:
        spar: The S-parameter collection to write.
        file_path: Destination path (conventionally .dat).

    Returns:
        str: file_path.
    """
    # port header lines: declared ports first, then any others found in the data
    port_names: list[str] = [p.name for p in spar.ports]
    for d in spar.data:
        for name in (f"port {d.out_port_num}", f"port {d.in_port_num}"):
            if name not in port_names:
                port_names.append(name)

    with open(file_path, "w", encoding="utf-8") as f:
        directions = {p.name: p.direction for p in spar.ports}
        for name in port_names:
            f.write(f'["{name}","{directions.get(name, "")}"]\n')

        for d in spar.data:
            if not (len(d.f) == len(d.s_mag) == len(d.s_phase)):
                raise ValueError(
                    f"S-parameter entry {d.idn} has inconsistent lengths: "
                    f"f={len(d.f)}, s_mag={len(d.s_mag)}, s_phase={len(d.s_phase)}"
                )
            f.write(
                f'("port {d.out_port_num}","mode {d.out_mode_num}",{d.out_mode_num},'
                f'"port {d.in_port_num}",{d.in_mode_num},"{d.data_type}")\n'
            )
            f.write(f"({len(d.f)}, 3)\n")
            for freq, mag, phase in zip(d.f, d.s_mag, d.s_phase, strict=True):
                f.write(f"{freq:.10e} {mag:.10e} {phase:.10e}\n")

    return file_path
