"""
gds_fdtd integration toolbox.

Layout processing module.
@author: Mustafa Hammood, 2024
"""

from .core import layout, port, structure, region
import logging
import klayout.db as pya

def dilate(vertices, extension=1.3):
    """grow or shrink a rectangle defined as [[x1,y1],[x2,y2]]

    Args:
        vertices (list): list defining rectangle: [[x1,y1],[x2,y2]]
        extension (int, optional): Growth amount. Defaults to 1.

    Returns:
        list: dilated rectangle.
    """
    import numpy as np

    x_min = np.min([i[0] for i in vertices]) - extension
    x_max = np.max([i[0] for i in vertices]) + extension
    y_min = np.min([i[1] for i in vertices]) - extension
    y_max = np.max([i[1] for i in vertices]) + extension

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def dilate_1d(vertices, extension=1, dim="y"):
    if dim == "x":
        if vertices[0][0] < vertices[1][0]:
            sign = 1
        else:
            sign = -1
        return [[vertices[0][0] - abs(extension)*sign, vertices[0][1]], [vertices[1][0] + abs(extension)*sign, vertices[1][1]]]
    elif dim == "y":
        if vertices[0][1] < vertices[1][1]:
            sign = 1
        else:
            sign = -1
        return [[vertices[0][0], vertices[0][1] - abs(extension)]*sign, [vertices[1][0], vertices[1][1] + abs(extension)*sign]]
    elif dim == "xy":
        if vertices[0][1] < vertices[1][1]:
            sign = 1
        else:
            sign = -1
        return [[vertices[0][0], vertices[0][1] - abs(extension)]*sign, [vertices[1][0] + abs(extension)*sign, vertices[1][1] + abs(extension)*sign]]
    
    else:
        raise ValueError("Dimension must be 'x' or 'y' or 'xy'")

def apply_prefab(fname, top_cell, MODEL_NAME="ANT_NanoSOI_ANF1_d9"):
    import prefab as pf
    device = pf.read.from_gds(gds_path=fname, cell_name=top_cell)

    prediction = device.predict(model=pf.models[MODEL_NAME])
    prediction_bin = prediction.binarize()

    prediction_bin.to_gds(gds_path=fname, cell_name=top_cell, gds_layer=(1, 0))
    return

def load_device(fname: str, tech, top_cell: str = None, z_span:float=3.0, z_center:float|None=None, prefab=None):
    from .simprocessor import load_component_from_tech
    ly = pya.Layout()
    ly.read(fname)
    
    if top_cell is None:
        if len(ly.top_cells()) > 1:
            err_msg = "More than one top cell found, ensure only 1 top cell exists. Otherwise, specify the cell using the top_cell argument."
            logging.error(err_msg)
            raise ValueError(err_msg)
        else:
            cell = ly.top_cell()
            name = cell.name
    else:
        cell = ly.cell(top_cell)
        if cell is None:
            err_msg = f"Top cell with name {top_cell} not found."
            logging.error(err_msg)
            raise ValueError(err_msg)
        name = cell.name

    c = load_component_from_tech(ly = layout(name, ly, cell), tech=tech, z_span=z_span, z_center=z_center)

    dbu = ly.dbu  # Get the database unit (dbu) from the layout

    for p in c.ports:
        polygon = p.polygon_extension(buffer=2)
        layer_index = ly.layer(pya.LayerInfo(1, 0))
        # Convert polygon vertices from um to dbu
        polygon_dbu = [pya.Point(int(pt[0] / dbu), int(pt[1] / dbu)) for pt in polygon]
        cell.shapes(layer_index).insert(pya.Polygon(polygon_dbu))
    
    new_layout_path = fname.replace(".gds", "_with_extensions.gds")
    ly.write(new_layout_path)
    
    if prefab is not None:
        apply_prefab(fname=new_layout_path, top_cell=top_cell, MODEL_NAME=prefab)

def load_layout(fname: str, top_cell: str = None) -> layout:
    """
    Load a GDS layout and return a layout object.

    Args:
        fname (str): Path to the GDS file.
        top_cell (str, optional): Name of the top cell. If None, the function will attempt to find a single top cell. Defaults to None.

    Returns:
        layout: A layout object containing the name, layout, and top cell.

    Raises:
        ValueError: If more than one top cell is found and top_cell is not specified, or if the specified top cell is not found.
    """
    ly = pya.Layout()
    ly.read(fname)
    
    if top_cell is None:
        if len(ly.top_cells()) > 1:
            err_msg = "More than one top cell found, ensure only 1 top cell exists. Otherwise, specify the cell using the top_cell argument."
            logging.error(err_msg)
            raise ValueError(err_msg)
        else:
            cell = ly.top_cell()
            name = cell.name
    else:
        cell = ly.cell(top_cell)
        if cell is None:
            err_msg = f"Top cell with name {top_cell} not found."
            logging.error(err_msg)
            raise ValueError(err_msg)
        name = cell.name

    return layout(name, ly, cell)


def load_region(cell: pya.Cell, layer: list[int, int] = [68, 0], z_center: float = 0., z_span: float = 5., extension: float = 1.3):
    """
    Get device bounds.

    Args:
        cell (pya.Cell): SiEPIC Tidy3d cell type to extract the polygons from.
        layer (list[int, int]): Layer to detect the devrec object from. Defaults to [68, 0].
        z_center (float): Z-center of the layout in microns. Defaults to 0.
        z_span (float): Z-span of the layout in microns. Defaults to 5.
        extension (float): Amount of extended region to retrieve beyond the specified region. Defaults to 1.3.

    Returns:
        region: Region object type.
    """

    def get_kdb_layer(layer):
        return cell.layout().layer(layer[0], layer[1])

    dbu = cell.layout().dbu
    layer = get_kdb_layer(layer)
    iter1 = cell.begin_shapes_rec(layer)
    # DevRec must be either a Box or a Polygon:
    if iter1.shape().is_box():
        box = iter1.shape().box.transformed(iter1.itrans())
        polygon = pya.Polygon(box)  # Save the component outline polygon
        DevRec_polygon = pya.Polygon(iter1.shape().box)
    if iter1.shape().is_polygon():
        polygon = iter1.shape().polygon.transformed(
            iter1.itrans()
        )  # Save the component outline polygon
        DevRec_polygon = iter1.shape().polygon
    polygons_vertices = [
        [[vertex.x * dbu, vertex.y * dbu] for vertex in p.each_point()]
        for p in [p.to_simple_polygon() for p in [DevRec_polygon]]
    ][0]

    if extension != 0:
        polygons_vertices = dilate(polygons_vertices, extension)
    return region(vertices=polygons_vertices, z_center=z_center, z_span=z_span)


def load_structure(cell, name, layer, z_base, z_span, material, sidewall_angle=90):
    """
    Extract polygons from a given cell on a given layer.

    Parameters
    ----------
    cell : klayout.db (pya) Cell type
        Cell to extract the polygons from.
    layer : klayout.db (pya) layout.layer() type
        Layer to place the pin object into.
    dbu : Float, optional
        Layout's database unit (in microns). The default is 0.001 (1 nm)

    Returns
    -------
    polygons_vertices : list [lists[x,y]]
        list of polygons from the cell.

    """

    def get_kdb_layer(layer):
        return cell.layout().layer(layer[0], layer[1])

    dbu = cell.layout().dbu
    layer = get_kdb_layer(layer)

    r = pya.Region()
    s = cell.begin_shapes_rec(layer)
    while not (s.at_end()):
        if s.shape().is_polygon() or s.shape().is_box() or s.shape().is_path():
            r.insert(s.shape().polygon.transformed(s.itrans()))
        s.next()

    r.merge()
    polygons = [p for p in r.each_merged()]
    polygons_vertices = [
        [[vertex.x * dbu, vertex.y * dbu] for vertex in p.each_point()]
        for p in [p.to_simple_polygon() for p in polygons]
    ]
    structures = []
    for idx, s in enumerate(polygons_vertices):
        name = f"{name}_{idx}"
        structures.append(
            structure(
                name=name,
                polygon=s,
                z_base=z_base,
                z_span=z_span,
                material=material,
                sidewall_angle=sidewall_angle,
            )
        )
    return structures


def load_structure_from_bounds(bounds, name, z_base, z_span, material, extension=2.0):
    """Load a structure from a region definition

    Args:
        bounds (core.region): Input region to use to generate structure.
        name (_type_): Name of structure.
        z_base (float): Z base of structure.
        z_span (float): Z span (thickness) of structure, can be negative for downward growth.
        material (tidy3d.Medium): Material of structure
        extension (float, optional): Growth (or shrinkage), in um, of structure defintion relative to bounds. Defaults to 2 um.

    Returns:
        core.structure: Structure generated from input region.
    """
    return structure(
        name=name,
        polygon=dilate(bounds.vertices, extension=extension),
        z_base=z_base,
        z_span=z_span,
        material=material,
    )


def load_ports(cell: pya.Cell, layer: list[int, int]=[1, 10]):
    """Load ports from cell.

    Args:
        cell (pya.Cell): Input cell object
        layer (list, optional): Ports layer identifier. Defaults to [1, 10].

    Returns:
        list: List of extracted port objects.
    """
    import klayout.db as pya

    def get_kdb_layer(layer):
        return cell.layout().layer(layer[0], layer[1])

    def get_direction(path):
        """Determine orientation of a pin path."""
        if path.points > 2:
            return ValueError("Number of points in a pin path are > 2.")
        p = path.each_point()
        p1 = p.__next__()
        p2 = p.__next__()
        if p1.x == p2.x:
            if p1.y > p2.y:  # north/south
                return 270
            else:
                return 90
        elif p1.y == p2.y:  # east/west
            if p1.x > p2.x:
                return 180
            else:
                return 0

    def get_center(path, dbu):
        """Determine center of a pin path."""
        p = path.each_point()
        p1 = p.__next__()
        p2 = p.__next__()
        direction = get_direction(path)
        if direction in [0, 180]:
            x = dbu * (p1.x + p2.x) / 2
            y = dbu * p1.y
        elif direction in [90, 270]:
            x = dbu * p1.x
            y = dbu * (p1.y + p2.y) / 2
        return x, y

    def get_name(c, x, y, dbu):
        s = c.begin_shapes_rec(get_kdb_layer(layer))
        while not (s.at_end()):
            if s.shape().is_text():
                label_x = s.shape().text.x * dbu
                label_y = s.shape().text.y * dbu
                if label_x == x and label_y == y:
                    return s.shape().text.string
            s.next()

    ports = []
    s = cell.begin_shapes_rec(get_kdb_layer(layer))
    while not (s.at_end()):
        if s.shape().is_path():
            width = s.shape().path_dwidth
            direction = get_direction(s.shape().path)
            # initialize Z center with none. Z center is identified in component init
            center = list(get_center(s.shape().path, cell.layout().dbu)) + [None]
            name = get_name(cell, center[0], center[1], cell.layout().dbu)
            ports.append(
                port(
                    name=name,
                    center=center,
                    width=width,
                    direction=direction,
                )
            )
        s.next()
    return ports
