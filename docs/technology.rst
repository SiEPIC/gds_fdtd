Technology Configuration
========================

Technology files define the physical properties of your photonic devices, including materials, layer definitions, and fabrication parameters. The ``gds_fdtd`` package uses YAML-based technology files that are compatible with both Tidy3D and Lumerical solvers.

Schema v2: named materials (recommended)
----------------------------------------

Define each material ONCE and reference it by name — one technology file
serves every solver:

.. code-block:: yaml

    technology:
      name: "EBeam"
      schema_version: 2

      materials:
        Si:
          nk: 3.476                      # neutral constant (beamz, grid/modes)
          tidy3d: [cSi, Li1993_293K]     # dispersive model on tidy3d
          lumerical: Si (Silicon) - Palik
        SiO2:
          nk: 1.444
          tidy3d: 1.444
          lumerical: SiO2 (Glass) - Palik

      substrate: {z_base: 0.0, z_span: -2, material: SiO2}
      superstrate: {z_base: 0.0, z_span: 3, material: SiO2}
      pinrec: [{layer: [1, 10]}]
      devrec: [{layer: [68, 0]}]
      device:
        - {layer: [1, 0], z_base: 0.0, z_span: 0.22, material: Si, sidewall_angle: 85}

Migrate a v1 file (per-layer inline materials) with::

    gds-fdtd convert-tech old.yaml

The two schemas are equivalent by construction — v2 expands into v1 before
validation. ``examples/tech.yaml`` is the reference v2 file.

.. _material-sources:

Material sources: ``nk`` vs the engine model vs refractiveindex.info
--------------------------------------------------------------------

A material may name up to **three** sources of optical constants. Each engine
picks exactly one when you run it — so the *same* material serves every solver.

.. list-table::
   :header-rows: 1
   :widths: 12 40 24

   * - source
     - what it is (tech-file key)
     - engines
   * - ``eda``
     - the engine's *own* database model — ``tidy3d`` / ``lumerical`` in the
       tech file (dispersive, vendor-maintained)
     - tidy3d, Lumerical
   * - ``rii``
     - a `refractiveindex.info <https://refractiveindex.info>`_ page (``rii:``)
       — dispersive, engine-independent, measured
     - all three
   * - ``nk``
     - a single neutral constant index (``nk:``)
     - all three

You do **not** have to specify all three. A material with only ``tidy3d`` and
``lumerical`` gets each vendor's dispersive model; add ``rii`` for one
engine-independent measured model across all engines, or just ``nk`` for a quick
constant.

Selection rule (per material, per engine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. If the material sets ``source:`` explicitly (``eda`` / ``rii`` / ``nk``),
   that source is used — and it is an **error** if that source is not defined
   for the engine you are running.
2. Otherwise the default precedence is **eda → rii → nk**: the first one that is
   defined (and applies to the engine) wins.
3. If *none* applies, a clear :class:`~gds_fdtd.errors.MaterialSourceError` is
   raised — for example a Lumerical-only technology run on tidy3d.

``beamz`` has no vendor material database, so its ``eda`` slot is always empty;
it uses ``rii`` (if present) or ``nk``.

.. code-block:: yaml

    materials:
      Si:
        nk: 3.476                           # constant fallback (and beamz)
        tidy3d: [cSi, Li1993_293K]          # tidy3d's own dispersive model
        lumerical: Si (Silicon) - Palik     # Lumerical's own dispersive model
        rii: {shelf: main, book: Si, page: Salzberg}   # refractiveindex.info
        source: rii     # OPTIONAL: force every engine to use the rii model
                        # (omit -> tidy3d/Lumerical use their model, beamz uses rii)

How each engine uses the chosen source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **tidy3d** — ``eda`` and ``rii`` both become *dispersive* media (``rii`` is
  fitted to a pole-residue medium via
  :meth:`gds_fdtd.materials.rii.RiiMaterial.to_tidy3d_medium`); ``nk`` becomes a
  constant ``td.Medium``.
- **Lumerical** — ``eda`` is the vendor database name (its own dispersion);
  ``rii`` / ``nk`` are emitted as a constant ``(n,k)`` material in the ``.lsf``
  (sampled at the band center — use the ``lumerical`` model when you want
  Lumerical's own dispersion).
- **beamz** — a single constant index from ``rii`` (sampled at band center) or
  ``nk``.

``rii`` pages are read **offline** from a local copy of the refractiveindex.info
database; point ``GDS_FDTD_RII_DB`` at its ``data`` directory (or pass
``db_dir=``). See :doc:`examples` notebook ``02b`` for feeding one rii model —
its full complex ``n(λ)+ik(λ)`` — into all three engines.

Technology File Structure (schema v1)
-------------------------------------

Basic Structure
^^^^^^^^^^^^^^^

A technology file contains the following main sections:

.. code-block:: yaml

    technology:
      name: "YourTechnology"

      substrate:
        # Substrate definition

      superstrate:
        # Superstrate definition

      device:
        # Device layer definitions

      pinrec:
        # Port layer definition

      devrec:
        # Device region definition

Each section defines different aspects of the device stack and simulation setup.

Material Definitions
--------------------

Tidy3D Materials
^^^^^^^^^^^^^^^^

For Tidy3D simulations, materials can be defined using:

1. **Material Database**: Pre-defined materials from Tidy3D's material library
2. **Constant Index**: Simple refractive index specification
3. **Custom Models**: Dispersive material models

.. code-block:: yaml

    # Tidy3D material database
    material:
      tidy3d_db:
        model: [cSi, Li1993_293K]  # [material, model]

    # Constant refractive index
    material:
      tidy3d:
        nk: 3.48  # Real refractive index (lossless)

    # Complex refractive index
    material:
      tidy3d:
        nk: 3.48 + 0.01j  # Real + imaginary parts

Popular Tidy3D Materials:
- ``[cSi, Li1993_293K]``: Crystalline silicon
- ``[Si3N4, Luke2015PMLStable]``: Silicon nitride
- ``[SiO2, Horiba]``: Silicon dioxide
- ``[Al2O3, Malitson1962]``: Aluminum oxide

Lumerical Materials
^^^^^^^^^^^^^^^^^^^

For Lumerical simulations, materials reference the Lumerical material database:

.. code-block:: yaml

    # Lumerical material database
    material:
      lum_db:
        model: Si (Silicon) - Palik

    # Other Lumerical materials:
    # - SiO2 (Glass) - Palik
    # - Si3N4 (Silicon Nitride) - Luke
    # - Al2O3 (Alumina) - Malitson

Layer Definitions
-----------------

Substrate Layer
^^^^^^^^^^^^^^^

The substrate forms the bottom layer of your device stack:

.. code-block:: yaml

    substrate:
      z_base: 0.0      # Base z-position (μm)
      z_span: -2.0     # Thickness (negative = downward growth)
      material:
        tidy3d_db:
          nk: 1.44     # SiO2 refractive index

Key parameters:
- ``z_base``: Starting z-position in micrometers
- ``z_span``: Layer thickness (negative for downward growth)
- ``material``: Material definition for the solver

Superstrate Layer
^^^^^^^^^^^^^^^^^

The superstrate forms the top cladding layer:

.. code-block:: yaml

    superstrate:
      z_base: 0.0      # Starting from substrate top
      z_span: 3.0      # Upward growth
      material:
        tidy3d_db:
          nk: 1.0      # Air cladding

Device Layers
^^^^^^^^^^^^^

Device layers contain the patterned structures from your GDS file:

.. code-block:: yaml

    device:
      # Silicon waveguide layer
      - layer: [1, 0]           # GDS layer [layer_number, datatype]
        z_base: 0.0             # Base z-position
        z_span: 0.22            # Layer thickness
        material:
          tidy3d_db:
            model: [cSi, Li1993_293K]
        sidewall_angle: 85      # Sidewall angle (degrees)

      # Silicon nitride layer
      - layer: [4, 0]
        z_base: 0.3
        z_span: 0.4
        material:
          tidy3d_db:
            model: [Si3N4, Luke2015PMLStable]
        sidewall_angle: 83

Device layer parameters:
- ``layer``: GDS layer specification [layer_number, datatype]
- ``z_base``: Base z-position of the layer
- ``z_span``: Layer thickness
- ``material``: Material definition
- ``sidewall_angle``: Etch angle (90° = vertical, <90° = tapered)

Port and Region Definitions
---------------------------

Port Layer (PinRec)
^^^^^^^^^^^^^^^^^^^

Defines which GDS layer contains port information:

.. code-block:: yaml

    pinrec:
      - layer: [1, 10]  # GDS layer for port shapes

The port layer should contain:
- Path objects defining port positions and orientations
- Text labels for port names
- Proper width specification for mode calculation

Device Region (DevRec)
^^^^^^^^^^^^^^^^^^^^^^

Defines the simulation region boundary:

.. code-block:: yaml

    devrec:
      - layer: [68, 0]  # GDS layer for device boundary

The device region layer should contain:
- Box or polygon defining the device extent
- Used for automatic simulation domain calculation

Complete Technology Examples
----------------------------

Tidy3D Technology File
^^^^^^^^^^^^^^^^^^^^^^

Complete example for Tidy3D simulations:

.. code-block:: yaml

    technology:
      name: "SiPhotonics_Tidy3D"

      substrate:
        z_base: 0.0
        z_span: -2.0
        material:
          tidy3d_db:
            nk: 1.44  # SiO2 substrate

      superstrate:
        z_base: 0.0
        z_span: 3.0
        material:
          tidy3d_db:
            nk: 1.0   # Air cladding

      pinrec:
        - layer: [1, 10]  # Port layer

      devrec:
        - layer: [68, 0]  # Device region layer

      device:
        # Silicon device layer
        - layer: [1, 0]
          z_base: 0.0
          z_span: 0.22
          material:
            tidy3d_db:
              model: [cSi, Li1993_293K]  # Crystalline silicon
          sidewall_angle: 85

        # Silicon nitride layer (if present)
        - layer: [4, 0]
          z_base: 0.3
          z_span: 0.4
          material:
            tidy3d_db:
              model: [Si3N4, Luke2015PMLStable]
          sidewall_angle: 83

Lumerical Technology File
^^^^^^^^^^^^^^^^^^^^^^^^^

Equivalent technology file for Lumerical:

.. code-block:: yaml

    technology:
      name: "SiPhotonics_Lumerical"

      substrate:
        z_base: 0.0
        z_span: -2.0
        material:
          lum_db:
            model: SiO2 (Glass) - Palik

      superstrate:
        z_base: 0.0
        z_span: 3.0
        material:
          lum_db:
            model: SiO2 (Glass) - Palik  # Same as substrate for cladding

      pinrec:
        - layer: [1, 10]

      devrec:
        - layer: [68, 0]

      device:
        - layer: [1, 0]
          z_base: 0.0
          z_span: 0.22
          material:
            lum_db:
              model: Si (Silicon) - Palik
          sidewall_angle: 85

        - layer: [4, 0]
          z_base: 0.3
          z_span: 0.4
          material:
            lum_db:
              model: Si3N4 (Silicon Nitride) - Luke
          sidewall_angle: 83

More Configuration Options
--------------------------

Multi-Layer Stacks
^^^^^^^^^^^^^^^^^^^

Define complex layer stacks:

.. code-block:: yaml

    device:
      # Bottom silicon layer
      - layer: [1, 0]
        z_base: 0.0
        z_span: 0.22
        material:
          tidy3d_db:
            model: [cSi, Li1993_293K]
        sidewall_angle: 85

      # Intermediate oxide
      - layer: [2, 0]
        z_base: 0.22
        z_span: 0.5
        material:
          tidy3d_db:
            nk: 1.44
        sidewall_angle: 90

      # Top silicon nitride
      - layer: [3, 0]
        z_base: 0.72
        z_span: 0.3
        material:
          tidy3d_db:
            model: [Si3N4, Luke2015PMLStable]
        sidewall_angle: 83

Custom Materials
^^^^^^^^^^^^^^^^

Define custom material properties:

.. code-block:: yaml

    # Lossy silicon (for testing)
    material:
      tidy3d_db:
        nk: 3.48 + 0.01j  # n + ik format

    # Temperature-dependent material (conceptual)
    material:
      tidy3d_db:
        model: [cSi, Li1993_300K]  # Different temperature model

Different Substrate Types
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # Silicon-on-insulator (SOI)
    substrate:
      z_base: 0.0
      z_span: -2.0
      material:
        tidy3d_db:
          nk: 1.44  # SiO2 BOX layer

    # Silicon-on-sapphire (SOS)
    substrate:
      z_base: 0.0
      z_span: -5.0
      material:
        tidy3d_db:
          model: [Al2O3, Malitson1962]  # Sapphire substrate

Loading and Using Technology Files
----------------------------------

Loading Technology
^^^^^^^^^^^^^^^^^^

Load technology files in your simulation scripts:

.. code-block:: python

    from gds_fdtd.technology import Technology

    # Load technology file
    tech_path = "examples/tech.yaml"
    technology = Technology.from_yaml(tech_path)

    # Inspect loaded technology
    print(f"Technology name: {technology.name}")
    print(f"Device layers: {len(technology.device)}")

Technology Validation
^^^^^^^^^^^^^^^^^^^^^

Validate technology definitions:

.. code-block:: python

    def validate_technology(tech):
        """Validate technology file completeness."""
        print("Technology Validation:")
        print("-" * 30)

        # Check required sections
        required_sections = ['substrate', 'superstrate', 'device', 'pinrec', 'devrec']
        for section in required_sections:
            if hasattr(tech, section) and getattr(tech, section) is not None:
                print(f"✓ {section} defined")
            else:
                print(f"✗ {section} missing")

        # Check device layers
        if hasattr(tech, 'device') and tech.device:
            print(f"Device layers: {len(tech.device)}")
            for i, layer in enumerate(tech.device):
                print(f"  Layer {i}: GDS {layer.layer} at z={layer.z_base}μm")

        print()

Technology Debugging
^^^^^^^^^^^^^^^^^^^^^

Debug material and layer issues:

.. code-block:: python

    def debug_materials(component):
        """Debug material assignments in loaded component."""
        print("Material Debug:")
        print("-" * 20)

        for structure in component.structures:
            print(f"Structure {structure.name} (role={structure.role}):")
            print(f"  Material: {structure.material}")
            print(f"  Layer: {structure.layer}")

Creating Technology Files
-------------------------

Design Guidelines
^^^^^^^^^^^^^^^^^

When creating technology files:

1. **Start with substrate**: Define the bottom layer first
2. **Layer ordering**: Define layers from bottom to top
3. **Material consistency**: Use appropriate materials for your platform
4. **Sidewall angles**: Match your fabrication process (typically 80-90°)
5. **Layer thickness**: Use realistic fabrication values
6. **GDS layer mapping**: Ensure GDS layers match your layout

Template Creation
^^^^^^^^^^^^^^^^^

Create a template for your technology:

.. code-block:: python

    def create_tech_template(name, substrate_material="SiO2", device_material="Si"):
        """Create a basic technology template."""
        template = f"""technology:
      name: "{name}"

      substrate:
        z_base: 0.0
        z_span: -2.0
        material:
          tidy3d_db:
            nk: 1.44  # {substrate_material}

      superstrate:
        z_base: 0.0
        z_span: 3.0
        material:
          tidy3d_db:
            nk: 1.0  # Air

      pinrec:
        - layer: [1, 10]  # Port layer

      devrec:
        - layer: [68, 0]  # Device region

      device:
        - layer: [1, 0]  # Main device layer
          z_base: 0.0
          z_span: 0.22
          material:
            tidy3d_db:
              model: [cSi, Li1993_293K]  # {device_material}
          sidewall_angle: 85
    """

        with open(f"{name.lower()}_tech.yaml", 'w') as f:
            f.write(template)

        print(f"Technology template created: {name.lower()}_tech.yaml")

    # Create a template
    create_tech_template("MyPhotonics", "SiO2", "Silicon")

Best Practices
--------------

1. **Version Control**: Keep technology files in version control
2. **Documentation**: Comment complex material definitions
3. **Validation**: Always validate loaded technology files
4. **Solver Compatibility**: Maintain separate files for different solvers
5. **Realistic Parameters**: Use fabrication-realistic layer thicknesses and angles
6. **Material Database**: Prefer validated material models over custom definitions
7. **Layer Naming**: Use descriptive GDS layer assignments
8. **Testing**: Validate technology files with simple test structures first

Common Issues and Solutions
---------------------------

**Material Not Found**:
- Check material name spelling in database
- Verify solver-specific material format
- Use fallback constant index materials for testing

**Layer Mapping Issues**:
- Verify GDS layer numbers match technology file
- Check datatype specifications
- Ensure port and device region layers exist in GDS

**Simulation Domain Problems**:
- Check z_base and z_span values for physical consistency
- Ensure positive z_span for upward growth, negative for downward
- Verify substrate extends below device layers

**Port Detection Failures**:
- Verify pinrec layer contains proper path objects
- Check port width and text label formatting
- Ensure port positions align with device features

Technology files are crucial for accurate simulations - spend time getting them right for your specific fabrication process and material system.
