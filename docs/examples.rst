Examples
========

A guided path from *"load a layout"* to *"run it on any FDTD engine and read the
S-parameters."* Each entry below is a real Jupyter notebook, executed and
committed **with its outputs**, so the plots and numbers you see are the actual
solver output.

Every notebook reproduces **for free**: the live runs use
`beamz <https://github.com/beamzorg/beamz>`_ (Apache-2.0 JAX FDTD, on CPU) or
tidy3d's free *local* mode solver, and the cross-engine comparisons load recorded
tidy3d/Lumerical artifacts. No cloud account, license, or GPU is required to run
any of them.

The source of truth for each notebook is a paired ``.py`` (jupytext *percent*
format) under ``examples/``; see ``examples/README.md`` for how to run them.

.. list-table::
   :header-rows: 1
   :widths: 8 30 62

   * - #
     - Notebook
     - You'll learn
   * - 00
     - :doc:`_notebooks/00_quickstart`
     - Layout → S-matrix in ten lines, on the free engine.
   * - 01
     - :doc:`_notebooks/01_layout_to_component`
     - Load a GDS / gdsfactory cell, auto-detect ports, read the geometry view.
   * - 02
     - :doc:`_notebooks/02_technology`
     - Materials, the vertical layer stack, and (in ``02b``) ``refractiveindex.info`` sources across engines.
   * - 03
     - :doc:`_notebooks/03_first_simulation`
     - The full flow end-to-end: geometry → permittivity → build → run → S-params → fields.
   * - 04
     - :doc:`_notebooks/04_reading_results`
     - ``SMatrix``: insertion loss, crosstalk, phase, reciprocity/passivity, Touchstone/HDF5/npz I/O.
   * - 05
     - :doc:`_notebooks/05_fields_and_modes`
     - Waveguide mode profiles, effective indices, and permittivity cross-sections.
   * - 05b
     - :doc:`_notebooks/05b_field_monitors`
     - Field monitors: axes, pinned positions, recorded wavelengths, and ``plot_monitor_planes`` — with the Si→SiN escalator's side view.
   * - 06
     - :doc:`_notebooks/06_convergence_and_caching`
     - Mesh-convergence sweeps, ``run_cached`` (repeat runs free), and cross-engine validation where *converged ≠ correct*.
   * - 07
     - :doc:`_notebooks/07_choosing_an_engine`
     - The identical job on beamz / tidy3d / Lumerical, and how they agree.
   * - 08
     - :doc:`_notebooks/08_frontends`
     - Any EDA in, any engine out: gdsfactory / SiEPIC / raw-GDS frontends, then the full frontend × engine matrix.
   * - 09
     - :doc:`_notebooks/09_cli_and_jobs`
     - The ``gds-fdtd`` CLI and serializable ``JobSpec`` for remote/batch compute.
   * - 10
     - :doc:`_notebooks/10_cookbook`
     - Reference devices with known-good S-params — the **Si→SiN escalator** on the free engine.
   * - 10b
     - :doc:`_notebooks/10b_polarization`
     - Polarization splitter and splitter-rotator from gdsfactory: TE/TM modes, multi-mode S-params, and per-polarization fields on two engines.
   * - 11
     - :doc:`_notebooks/11_bragg_grating`
     - A 95 µm SiEPIC Bragg grating on tidy3d: the stopband spectrum, and the same field monitor in-band (reflected) vs out-of-band (transmitted).

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Example notebooks

   _notebooks/00_quickstart
   _notebooks/01_layout_to_component
   _notebooks/02_technology
   _notebooks/02b_rii_to_engines
   _notebooks/03_first_simulation
   _notebooks/04_reading_results
   _notebooks/05_fields_and_modes
   _notebooks/05b_field_monitors
   _notebooks/06_convergence_and_caching
   _notebooks/07_choosing_an_engine
   _notebooks/08_frontends
   _notebooks/09_cli_and_jobs
   _notebooks/10_cookbook
   _notebooks/10b_polarization
   _notebooks/11_bragg_grating
