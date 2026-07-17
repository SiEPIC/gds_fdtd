API Reference
=============

Every public module, class, and function in ``gds_fdtd``. Each module page
below is generated from the source docstrings and links to the individual
classes and functions it defines.

If you are looking for a task-oriented introduction instead, start with
:doc:`simulations` (GDS → S-parameters), :doc:`technology` (materials and the
layer stack), or the :doc:`examples` gallery.

Core data types
---------------

The objects that flow through every simulation, independent of engine.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.geometry
   gds_fdtd.smatrix
   gds_fdtd.spec

Technology and materials
------------------------

The layer stack, named materials, and the three optical-constant sources
(engine model / refractiveindex.info / constant ``nk``). See :doc:`technology`.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.technology
   gds_fdtd.materials.rii
   gds_fdtd.materials.select

Layout frontends
----------------

Turning a layout (GDS, gdsfactory, SiEPIC/KLayout) into a ``Component``. See
:doc:`frontends`.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.lyprocessor
   gds_fdtd.simprocessor
   gds_fdtd.layout.gdsfactory

Solvers
-------

The engine-agnostic ``Solver`` contract, the registry (``get_solver``), and the
tidy3d / Lumerical / beamz adapters. See :doc:`solvers` and
:doc:`adding_a_solver`.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.solvers
   gds_fdtd.solvers.base
   gds_fdtd.solvers.tidy3d
   gds_fdtd.solvers.lumerical
   gds_fdtd.solvers.beamz

Fields, modes, and meshing
--------------------------

The kernel-engine pipeline: permittivity rasterization, local mode solving, and
mode-overlap S-parameter extraction.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.grid
   gds_fdtd.modes
   gds_fdtd.extraction

Analysis and visualization
--------------------------

Reading, plotting, converging, and cross-validating results. See
:doc:`sparameters`.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.plotting
   gds_fdtd.viewer3d
   gds_fdtd.convergence
   gds_fdtd.validation
   gds_fdtd.caching

Execution and CLI
-----------------

Serializable jobs, local/subprocess backends, and the ``gds-fdtd`` command-line
interface. See :doc:`remote_compute`.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.execution.jobspec
   gds_fdtd.execution.backends
   gds_fdtd.cli

Configuration and errors
------------------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gds_fdtd.errors
   gds_fdtd.settings
   gds_fdtd.logging_config
