"""
gds_fdtd simulation toolbox.

Layout processing module.
@author: Mustafa Hammood, 2025
"""

import logging

import klayout.db as pya

from .geometry import Port, Region, Structure


def dilate(vertices, extension=1.0):
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
    """Extend a 2-point segment/rectangle outward along the given dimension(s).

    Args:
        vertices: two corner points [[x1, y1], [x2, y2]] in any order.
        extension: growth amount (its absolute value is applied outward).
        dim: "x", "y", or "xy".

    Returns:
        The two corner points, moved apart by ``abs(extension)`` along the
        selected dimension(s). Point order is preserved.
    """
    (x1, y1), (x2, y2) = vertices
    ex = abs(extension)
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    if dim == "x":
        return [[x1 - ex * sx, y1], [x2 + ex * sx, y2]]
    if dim == "y":
        return [[x1, y1 - ex * sy], [x2, y2 + ex * sy]]
    if dim == "xy":
        return [[x1 - ex * sx, y1 - ex * sy], [x2 + ex * sx, y2 + ex * sy]]
    raise ValueError("Dimension must be 'x' or 'y' or 'xy'")


def apply_prefab(gds_in: str, gds_out: str, top_cell: str, model: str = "ANT_NanoSOI_ANF1_d9"):
    """Run a PreFab lithography prediction and write the result to a NEW file.

    Args:
        gds_in: Path of the input GDS (never modified).
        gds_out: Path to write the predicted GDS to (must differ from gds_in).
        top_cell: Name of the cell to predict.
        model: PreFab model name.

    Returns:
        str: gds_out.
    """
    import os

    import prefab as pf

    if os.path.abspath(gds_in) == os.path.abspath(gds_out):
        raise ValueError("gds_out must differ from gds_in — apply_prefab never overwrites its input")

    device = pf.read.from_gds(gds_path=gds_in, cell_name=top_cell)
    prediction = device.predict(model=pf.models[model])
    prediction_bin = prediction.binarize()
    prediction_bin.to_gds(gds_path=gds_out, cell_name=top_cell, gds_layer=(1, 0))
    return gds_out


def load_device(
    fname: str,
    tech,
    top_cell: str | None = None,
    z_span: float = 3.0,
    z_center: float | None = None,
    prefab_model: str | None = None,
    output_dir: str | None = None,
):
    """Load a GDS device into a component, writing derived GDS files to output_dir.

    Writes ``<cell>_with_extensions.gds`` (the layout with port extension stubs)
    and, when ``prefab_model`` is given, ``<cell>_predicted.gds`` into
    ``output_dir`` (a temporary directory by default). The input file is never
    modified (previously: derived files were written next to the input, the
    built component was discarded, and the function returned None; bug B15).

    Returns:
        core.component: The parsed component.
    """
    import os
    import tempfile

    from .simprocessor import load_component_from_tech

    cell, ly = load_cell(fname, top_cell=top_cell)

    c = load_component_from_tech(cell=cell, tech=tech, z_span=z_span, z_center=z_center)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="gds_fdtd_")
    os.makedirs(output_dir, exist_ok=True)

    # add port extension stubs so sources sit on straight waveguide
    dbu = ly.dbu
    layer_index = ly.layer(pya.LayerInfo(1, 0))
    for p in c.ports:
        polygon = p.polygon_extension(buffer=2)
        polygon_dbu = [pya.Point(int(pt[0] / dbu), int(pt[1] / dbu)) for pt in polygon]
        cell.shapes(layer_index).insert(pya.Polygon(polygon_dbu))

    extended_path = os.path.join(output_dir, f"{cell.name}_with_extensions.gds")
    ly.write(extended_path)
    logging.info("Wrote extended layout to %s", extended_path)

    if prefab_model is not None:
        predicted_path = os.path.join(output_dir, f"{cell.name}_predicted.gds")
        apply_prefab(
            gds_in=extended_path,
            gds_out=predicted_path,
            top_cell=cell.name,
            model=prefab_model,
        )
        logging.info("Wrote PreFab prediction to %s", predicted_path)

    return c


def load_cell(fname: str, top_cell: str = None) -> pya.Cell:
    """
    Load a GDS layout and return a cell object.

    Args:
        fname (str): Path to the GDS file.
        top_cell (str, optional): Name of the top cell. If None, the function will attempt to find a single top cell. Defaults to None.

    Returns:
        pya.Cell: A cell object containing the name, layout, and top cell.

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

    return cell, ly


def load_region(
    cell: pya.Cell,
    layer: list[int, int] = [68, 0],
    z_center: float = 0.0,
    z_span: float = 5.0,
    extension: float = 0.0,
):
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

    dbu = cell.layout().dbu
    layer_spec = list(layer)
    layer_idx = cell.layout().layer(layer_spec[0], layer_spec[1])

    # Collect every box/polygon DevRec shape on the layer (previously only the
    # first shape was inspected, and a non-box/non-polygon first shape left the
    # polygon variables undefined -> NameError; bug B8).
    devrec_polygons = []
    it = cell.begin_shapes_rec(layer_idx)
    while not it.at_end():
        shape = it.shape()
        if shape.is_box():
            devrec_polygons.append(pya.Polygon(shape.box))
        elif shape.is_polygon():
            devrec_polygons.append(shape.polygon)
        it.next()

    if not devrec_polygons:
        raise ValueError(
            f"No DevRec box/polygon found on layer {layer_spec[0]}:{layer_spec[1]} "
            f"in cell {cell.name!r}. Check the technology 'devrec' layer."
        )

    devrec_polygon = devrec_polygons[0]
    if len(devrec_polygons) > 1:
        logging.warning(
            "Multiple DevRec shapes found on layer %s:%s in cell %r; using the "
            "bounding box of their union.",
            layer_spec[0],
            layer_spec[1],
            cell.name,
        )
        union = pya.Region()
        for p in devrec_polygons:
            union.insert(p)
        union.merge()
        bbox = union.bbox()
        devrec_polygon = pya.Polygon(bbox)

    polygons_vertices = [
        [vertex.x * dbu, vertex.y * dbu]
        for vertex in devrec_polygon.to_simple_polygon().each_point()
    ]

    if extension != 0:
        polygons_vertices = dilate(polygons_vertices, extension)
    return Region(vertices=polygons_vertices, z_center=z_center, z_span=z_span)


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

    dbu = cell.layout().dbu
    layer_idx = cell.layout().layer(layer[0], layer[1])

    r = pya.Region()
    s = cell.begin_shapes_rec(layer_idx)
    while not (s.at_end()):
        if s.shape().is_polygon() or s.shape().is_box() or s.shape().is_path():
            r.insert(s.shape().polygon.transformed(s.itrans()))
        s.next()

    r.merge()
    polygons = list(r.each_merged())
    polygons_vertices = [
        [[vertex.x * dbu, vertex.y * dbu] for vertex in p.each_point()]
        for p in [p.to_simple_polygon() for p in polygons]
    ]
    structures = []
    for idx, s in enumerate(polygons_vertices):
        structure_name = f"{name}_{idx}"
        structures.append(
            Structure(
                name=structure_name,
                polygon=s,
                z_base=z_base,
                z_span=z_span,
                material=material,
                sidewall_angle=sidewall_angle,
                layer=layer,  # Pass the layer information to the structure
            )
        )
    return structures


def load_structure_from_bounds(bounds, name, z_base, z_span, material, extension=0.0, layer=[1, 0]):
    """Load a structure from a region definition

    Args:
        bounds (core.region): Input region to use to generate structure.
        name (_type_): Name of structure.
        z_base (float): Z base of structure.
        z_span (float): Z span (thickness) of structure, can be negative for downward growth.
        material (tidy3d.Medium): Material of structure
        extension (float, optional): Growth (or shrinkage), in um, of structure defintion relative to bounds. Defaults to 2 um.
        layer (list[int], optional): GDS layer specification as [layer_number, datatype]. Defaults to [1, 0].

    Returns:
        core.structure: Structure generated from input region.
    """
    return Structure(
        name=name,
        polygon=dilate(bounds.vertices, extension=extension),
        z_base=z_base,
        z_span=z_span,
        material=material,
        layer=layer,
    )


def load_ports(cell: pya.Cell, layer: list[int, int] = [1, 10]):
    """Load ports from cell.

    Args:
        cell (pya.Cell): Input cell object
        layer (list, optional): Ports layer identifier. Defaults to [1, 10].

    Returns:
        list: List of extracted port objects.
    """

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
        s = c.begin_shapes_rec(cell.layout().layer(layer[0], layer[1]))
        while not (s.at_end()):
            if s.shape().is_text():
                label_x = s.shape().text.x * dbu
                label_y = s.shape().text.y * dbu
                if label_x == x and label_y == y:
                    return s.shape().text.string
            s.next()

    ports = []
    s = cell.begin_shapes_rec(cell.layout().layer(layer[0], layer[1]))
    while not (s.at_end()):
        if s.shape().is_path():
            width = s.shape().path_dwidth
            direction = get_direction(s.shape().path)
            # initialize Z center with none. Z center is identified in component init
            center = list(get_center(s.shape().path, cell.layout().dbu)) + [None]
            name = get_name(cell, center[0], center[1], cell.layout().dbu)
            ports.append(
                Port(
                    name=name,
                    center=center,
                    width=width,
                    direction=direction,
                )
            )
        s.next()
    return ports
