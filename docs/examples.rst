Examples
========

The ``examples/`` directory covers every feature; each script runs directly
with Python. Scripts follow one standardized flow: geometry plot (ports +
FDTD region + port extensions) → offline setup (free) → S-parameters →
field profile.

.. list-table::
   :header-rows: 1

   * - Directory
     - Shows
   * - ``01_basics``
     - First contact: load & inspect any GDS with no engine installed
       (``01b``); the free offline setup flow on tidy3d (``01a``).
   * - ``02_lumerical``
     - The standard flow on Lumerical FDTD; ``02b`` is a mesh-convergence
       study — identical code to ``03b`` except the engine string.
   * - ``03_tidy3d``
     - The same flow on the tidy3d cloud (``03a`` with curated
       thru/crosstalk/reflection plots); ``03b`` mesh convergence.
   * - ``04_solvers``
     - The engine-agnostic registry (``04a``); convergence sweeps + job
       caching (``04b``); three-engine cross-validation on recorded real
       results — runs offline (``04c``).
   * - ``05_gdsfactory``
     - gdsfactory (>= 9) components into any solver.
   * - ``06_beamz``
     - The free, open-source engine: straight (``06a``), S-bend
       (``06b``), and a zero-cost mesh-convergence study (``06c``).
   * - ``07_prefab``
     - Lithography-predicted geometry via PreFab.
   * - ``08_siepic``
     - SiEPIC EBeam PDK cells on tidy3d / Lumerical.
   * - ``09_smatrix``
     - SMatrix I/O on recorded real data: .dat, Touchstone, npz/HDF5,
       physics checks, plotting — fully offline.
   * - ``10_materials``
     - Validated technology YAML (schema v2, named materials) +
       refractiveindex.info sources.

Run any of them like::

    python examples/04_solvers/04c_cross_solver_validation.py

Examples that call ``solver.run()`` on tidy3d spend FlexCredits and say so
in their docstrings; Lumerical examples need a local license; every beamz
example is free.
