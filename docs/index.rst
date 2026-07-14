gds_fdtd
========

**EDA- and solver-agnostic 3D FDTD simulation framework for photonic layouts:
GDS in, S-parameters, fields, and compact models out - tidy3d, Lumerical, or
beamz behind one API.**

``gds_fdtd`` takes one component, one technology file, and one
``SimulationSpec``, and runs the identical job on any engine: `tidy3d
<https://github.com/flexcompute/tidy3d>`_ (cloud), Ansys Lumerical (local), or
`beamz <https://github.com/beamzorg/beamz>`_ (free, JAX), behind a single
``get_solver(name)`` contract. The three engines agree within 0.052 dB on an
identical job (tidy3d and Lumerical within 0.0033 dB).

.. code-block:: python

    from gds_fdtd.technology import Technology
    from gds_fdtd.layout.gdsfactory import from_gdsfactory
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec
    import gdsfactory as gf

    tech = Technology.from_yaml("tech.yaml")
    gf.gpdk.PDK.activate()
    component = from_gdsfactory(gf.components.mmi1x2(), tech)   # any frontend

    solver = get_solver("beamz")(component, tech, SimulationSpec())  # any engine
    smatrix = solver.run()          # the only call that spends money / a license / GPU
    smatrix.to_touchstone("mmi.s3p")

.. figure:: images/three_engine_agreement.png
   :width: 85%
   :align: center

   The same y-branch, three engines: all land within a few hundredths of a dB of
   each other and the −3 dB ideal. Reproduced in :doc:`_notebooks/07_choosing_an_engine`.

Start here
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Examples gallery
      :link: examples
      :link-type: doc

      Twelve executed notebooks, from a ten-line quickstart to polarization
      splitters. Every plot is real solver output.

   .. grid-item-card:: API reference
      :link: modules
      :link-type: doc

      Every public module, class, and function, grouped by topic.

   .. grid-item-card:: Setting up simulations
      :link: simulations
      :link-type: doc

      GDS to S-parameters in four steps, on any engine.

   .. grid-item-card:: Technology and materials
      :link: technology
      :link-type: doc

      The layer stack and the three optical-constant sources
      (engine model, refractiveindex.info, or a constant).

   .. grid-item-card:: Layout frontends
      :link: frontends
      :link-type: doc

      gdsfactory, SiEPIC/KLayout, raw GDS, PreFab, and how to add your own.

   .. grid-item-card:: Bring your own engine
      :link: adding_a_solver
      :link-type: doc

      Any FDTD engine becomes a gds_fdtd solver in four methods.

What you can do with it
-----------------------

- Run the same job on tidy3d, Lumerical, or beamz via ``get_solver(name)``,
  and :doc:`cross-validate <solvers>` them against each other.
- Load layouts from :doc:`gdsfactory, SiEPIC/KLayout, or raw GDS <frontends>`
  with **auto-detected ports**.
- Configure the :doc:`layer stack and materials <technology>` in one YAML,
  including dispersive `refractiveindex.info <https://refractiveindex.info>`_
  models that feed every engine.
- Solve :doc:`TE and TM modes <multimodal>` in one simulation.
- Extract, check (reciprocity / passivity), plot, and export
  :doc:`S-parameters <sparameters>` to Touchstone, INTERCONNECT ``.dat``, HDF5,
  or npz.
- Preview cost offline and only spend on ``run()``; ship jobs to
  :doc:`remote/batch compute <remote_compute>` with enforced budgets.

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   simulations
   technology
   frontends
   sparameters
   multimodal

.. toctree::
   :maxdepth: 2
   :caption: Engines & compute
   :hidden:

   solvers
   adding_a_solver
   remote_compute
   self_hosted_runner

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   architecture
   examples
   modules
