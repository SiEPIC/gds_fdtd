Examples
========

The ``examples`` directory contains a variety of small scripts that
show how to use the library with different simulation engines and
workflows.  Each script can be executed directly with Python once the
package and optional dependencies are installed.

.. list-table:: Available examples
   :header-rows: 1

   * - Directory
     - Description

   * - ``01_basics``
     - Simple directional coupler and crossing simulations using
       :mod:`tidy3d`.
   * - ``02_multilayer_device``
     - Multilayer structures and more complex port definitions.
   * - ``03_import_technology``
     - Importing technology YAML files and performing convergence
       studies.
   * - ``04_dual_polarization``
     - Running a Bragg grating with both TE and TM polarisations.
   * - ``05_gdsfactory``
     - Integration with :mod:`gdsfactory` component definitions.
   * - ``06_lumerical``
     - Exporting structures to Lumerical's FDTD solver.
   * - ``07_prefab``
     - Example workflow with the PreFab lithography package.
   * - ``08_siepic``
     - Utilities for the SiEPIC ecosystem.

To run an example simply execute for instance::

    python examples/01_basics/01a_directional_coupler_tidy3d.py

Each script is well commented and mirrors the high level API provided by
``gds_fdtd``.
