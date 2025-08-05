.. gds_fdtd documentation master file, created by
   sphinx-quickstart on Tue Sep 05 2023.

Welcome to gds_fdtd's documentation!
=====================================

``gds_fdtd`` is a Python package that helps you set up FDTD simulations for photonic devices. It works with both Tidy3D and Lumerical solvers.

What you can do with it:
- Run simulations using either Tidy3D or Lumerical
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
   solvers
   simulations
   sparameters
   multimodal
   technology
   examples
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
