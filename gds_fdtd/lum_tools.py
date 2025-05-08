"""
gds_fdtd simulation toolbox.

Lumerical tools interface module.
@author: Mustafa Hammood, 2025
"""

from gds_fdtd.core import structure, component
import logging
import lumapi
import numpy as np
import os
import shutil

m_to_um = 1e-6

def structure_to_lum_poly(
    s: structure,
    lum: lumapi.FDTD,
    alpha: float = 1.0,
    group: bool = False,
    group_name: str = "group",
):
    """import a structure objecto to a lumerical instance.

    Args:
        s (structure): structure to instantiate
        lum (lumapi.FDTD): lumerical instance
        alpha (float, optional): transperancy setting. Defaults to 1..
        group (bool, optional): flag to add the structure to a given group. Defaults to False.
        group_name (str, optional): group name, if group is True. Defaults to 'group'.
    """
    if s.z_span < 0:
        bounds = (s.z_base + s.z_span, s.z_base)
    else:
        bounds = (s.z_base, s.z_base + s.z_span)

    poly = lum.addpoly(
        vertices=m_to_um * np.array(s.polygon),
        x=0,
        y=0,
        z_min=m_to_um * bounds[0],
        z_max=m_to_um * bounds[1],
        name=s.name,
        material=s.material["lum"] if isinstance(s.material, dict) else s.material,
        alpha=alpha,
    )

    if group:
        lum.addtogroup(group_name)
    lum.eval(f"?'Polygons {s.name} added';")

    return poly


def to_lumerical(c: component, lum: lumapi.FDTD, buffer: float = 2.0) -> list:
    """Add an input component with a given tech to a lumerical instance.

    Args:
        c (component): input component.
        lum (lumapi.FDTD): lumerical FDTD instance.
    """

    # TODO fix box tox handling here
    structures = []
    for s in c.structures:
        # if structure is a list then its a device (could have multiple polygons inside)
        if type(s) == list:
            for i in s:
                structures.append(
                    structure_to_lum_poly(s=i, lum=lum, group=True, group_name="device")
                )

        # if structure is not a list then its a region
        else:
            structures.append(structure_to_lum_poly(s=s, lum=lum, alpha=0.5))

    # extend ports beyond sim region
    for p in c.ports:
        structures.append(
            lum.addpoly(
                vertices=m_to_um * np.array(p.polygon_extension(buffer=buffer)),
                x=0,
                y=0,
                z_min=m_to_um * (p.center[2] - p.height / 2),
                z_max=m_to_um * (p.center[2] + p.height / 2),
                name=p.name,
                material=(
                    p.material["lum"] if isinstance(p.material, dict) else p.material
                ),
            )
        )

        lum.addtogroup("ports")
        lum.eval(f"?'port {p.name} added';")

    return structures


def make_sim_lum(
    c: component,
    lum: lumapi.FDTD,
    wavl_min: float = 1.45,
    wavl_max: float = 1.65,
    wavl_pts: int = 101,
    width_ports: float = 3.0,
    depth_ports: float = 2.0,
    symmetry: tuple[int, int, int] = (0, 0, 0),
    pol: str = "TE",
    num_modes: int = 1,
    boundary: str = "pml",
    mesh: int = 2,
    run_time_factor: float = 50,
    z_span: float | None = None,
    field_monitor_axis: str | None = None,
    visualize: bool = True,
    export_plot_file: str = 'sparam.png', 
    gpu: bool = False,
) -> dict[str, list[float]]:

    # send component to lumerical instance
    to_lumerical(c=c, lum=lum)

    if z_span == None:
        sim_size = 1e-6 * np.array([c.bounds.x_span, c.bounds.y_span, c.bounds.z_span])
    else:
        sim_size = 1e-6 * np.array([c.bounds.x_span, c.bounds.y_span, z_span])

    run_time = run_time_factor * max(sim_size) / 3e8  # 85/fwidth  # sim. time in secs

    lum.addfdtd(
        x=c.bounds.x_center * 1e-6,
        y=c.bounds.y_center * 1e-6,
        z=c.bounds.z_center * 1e-6,
        x_span=sim_size[0],
        y_span=sim_size[1],
        z_span=sim_size[2],
        simulation_time=run_time,
        mesh_accuracy=mesh,
        x_min_bc=boundary,
        y_min_bc=boundary,
        z_min_bc=boundary,
        x_max_bc=boundary,
        y_max_bc=boundary,
        z_max_bc=boundary,
    )

    if gpu:
        lum.setnamed("FDTD", "express mode", True)  # for GPU acceleration
    else:
        lum.setnamed("FDTD", "express mode", False)

    lum.setglobalsource("wavelength start", wavl_min*1e-6)
    lum.setglobalsource("wavelength stop", wavl_max*1e-6)
    lum.setglobalmonitor("frequency points", wavl_pts)

    if pol.lower() == "te":
        polarization = "fundamental TE mode"
    elif pol.lower() == "tm":
        polarization = "fundamental TM mode"
    else:
        raise ValueError(f"Polarization {pol} not supported")

    buffer = 1e-6
    for p in c.ports:
        port = lum.addport()
        lum.set("name", p.name)

        if p.direction in [90, 270]:
            if p.direction == 90:
                lum.set("y", p.center[1]*1e-6 + buffer)
                lum.set("direction", "Backward")
            else:
                lum.set("y", p.center[1]*1e-6 - buffer)
                lum.set("direction", "Forward")

            lum.set("z", p.center[2]*1e-6)
            lum.set("x", p.center[0]*1e-6)
            lum.set("injection axis", "y-axis")
            lum.set("x span", width_ports*1e-6)
            lum.set("z span", depth_ports*1e-6)
        elif p.direction in [180, 0]:
            if p.direction == 180:
                lum.set("x", p.center[0]*1e-6 - buffer)
                lum.set("direction", "Forward")
            else:
                lum.set("x", p.center[0]*1e-6 + buffer)
                lum.set("direction", "Backward")

            lum.set("y", p.center[1]*1e-6)
            lum.set("z", p.center[2]*1e-6)
            lum.set("injection axis", "x-axis")
            lum.set("y span", width_ports*1e-6)
            lum.set("z span", depth_ports*1e-6)
        else:
            raise ValueError(f"Port direction {p.direction} not supported")

        lum.set("mode selection", polarization)
        lum.set("number of field profile samples", num_modes)

    lum.save(f"{c.name}.fsp")
    lum.addsweep(3)
    lum.setsweep("s-parameter sweep", "name", "sparams")
    #input("Press Enter to continue...")
    if gpu:
        lum.runsweep("sparams", "GPU")
    else:
        lum.runsweep("sparams")

    lum.exportsweep("sparams", f"{c.name}.dat")
    sparams_sweep = lum.getsweepresult("sparams", "S parameters")

    sparams = {}
    sparams["wavl"] = [i[0] for i in sparams_sweep["lambda"]]

    ports = [port.name for port in c.ports]  # Extract port names (e.g., ['opt2', 'opt1'])
    for p1 in ports:
        for p2 in ports:
            key = f"S{ports.index(p1) + 1}{ports.index(p2) + 1}"  # Generates keys like "s11", "s12", "s21", "s22"
            sparams[f"s{ports.index(p1) + 1}_{ports.index(p2) + 1}"] = sparams_sweep[key]  # Maps to corresponding S-parameter value
    
    if visualize or export_plot_file:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for param, values in sparams.items():
            if param == "wavl":
                continue  # Skip the wavelength key
            # Power in dB: 10 * log10(|S|^2)
            power_db = 10 * np.log10(np.abs(values) ** 2)
            
            plt.plot(1e6*np.array(sparams["wavl"]), power_db, label=f"|{param}|² (dB)")
        
        plt.ylabel("Transmission [dB]")
        plt.xlabel("Wavelength [μm]")
        plt.title("S-Parameters Power Response")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        if export_plot_file:
            plt.savefig(export_plot_file) # Saves the plot 
        if visualize:
            plt.show()

    return sparams

def Update_halfring_CML(device,CML,sparam_file,gap,rad,width,thickness,CoupleLength):
    """
    Moves the halfring sparam_file to the local EBeam CML folder found by querying interconnect. 

    Parameters:
     device (str): CML device name. ex: ebeam_dc_halfring_straight
     CML (str): The CML library to add the sparam file to. ex: EBeam
     sparam_file (str): The name/path of the sparam file you are adding to the CML
     gap (int): The halfring coupling gap in nanometers
     rad (int): The halfring radius in microns
     width (int): The halfring waveguide width in nanometers
     thickness (int): The halfring waveguide thickness in nanometers
     CoupleLength (int): The halfring coupler length in microns

    Example: 
    Update_halfring_CML("ebeam_dc_halfring_straight","EBeam","sparams.dat",gap=2,rad=20,width=2,thickness=220,CoupleLength=2)
    """
    
    # Ensure sparam_file contains .dat
    if not('.dat' in sparam_file):
        sparam_file = sparam_file + ".dat"
        
    to_check = [gap,rad,width,thickness,CoupleLength]

    for prm in to_check:
        if not isinstance(prm, int):
            if prm in [rad,CoupleLength]:
                raise TypeError(f"Parameter '{prm}' must be an integer (in microns), got {type(prm).__name__}.")
            else:
                if prm in [rad,CoupleLength]:
                    raise TypeError(f"Parameter '{prm}' must be an integer (in nanometers), got {type(prm).__name__}.")
    
    # Query Lumerical INTERCONNECT to find the path for the specific design kit
    intc = lumapi.open('interconnect')

    # Get all the library elements
    command = 'elements=library;'
    lumapi.evalScript(intc, command)
    intc_elements = lumapi.getVar(intc, "elements")
    intc_elements = intc_elements.split('\n')
    
    # find the ones that match the requested Design Kit name
    j=[i for i in intc_elements if '::design kits::'+CML.lower() in i]
    if not j:
        raise Exception('No elements in the Design Kit "%s" found.' % CML)
    
    # find the first element in the root folder
    intc_element=[i for i in j if len(i.split('::'))==4][0]
    
    # get the CML path
    command = f'addelement("{intc_element}"); \n'
    command += 'path=get("local path");'
    lumapi.evalScript(intc, command)
    CML_path = lumapi.getVar(intc, "path")
    
    path_halfring = os.path.join(CML_path, 'source_data/' + device)
    filename = f"te_ebeam_dc_halfring_straight_gap={gap}nm_radius={rad}um_width={width}nm_thickness={thickness}nm_CoupleLength={CoupleLength}um.dat"
    destination = os.path.join(path_halfring, filename)
    print(destination)

    # Get the source data file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    source = os.path.join(dir_path, sparam_file)
                            
    # copy the file
    shutil.copyfile(source, destination)
    return


