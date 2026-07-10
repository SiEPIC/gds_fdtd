.. gds_fdtd documentation master file, created by
   sphinx-quickstart on Tue Sep 05 2023.

Welcome to gds_fdtd's documentation!
=====================================

``gds_fdtd`` is a Python package that helps you set up FDTD simulations for photonic devices. One component + one technology file + one ``SimulationSpec`` runs on any engine — tidy3d (cloud), Lumerical (local), or beamz (free/JAX) — behind a single ``get_solver(name)`` contract.

What you can do with it:
- Run simulations on tidy3d, Lumerical, or beamz via ``get_solver(name)``
- Handle TE and TM polarizations in the same simulation
- Extract and analyze S-parameters from your results
- Monitor electromagnetic fields during simulation
- Configure materials and layers using YAML files
- Load GDS layouts and automatically detect ports

The sections below will show you how to install and use the package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   adding_a_solver
   solvers
   simulations
   sparameters
   multimodal
   technology
   examples
   remote_compute
   self_hosted_runner
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
