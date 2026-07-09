Solvers
=======

Every engine sets up the same way — swap engines by changing one string:

.. code-block:: python

    from gds_fdtd.core import parse_yaml_tech
    from gds_fdtd.lyprocessor import load_cell
    from gds_fdtd.simprocessor import load_component_from_tech
    from gds_fdtd.solvers import get_solver
    from gds_fdtd.spec import SimulationSpec

    tech = parse_yaml_tech("tech.yaml")            # ONE tech, every engine
    cell, layout = load_cell("devices.gds", top_cell="crossing_te1550")
    component = load_component_from_tech(cell=cell, tech=tech)

    solver = get_solver("tidy3d")(                 # or "lumerical" / "beamz"
        component,
        technology=tech,
        spec=SimulationSpec(wavelength_points=51, mesh=10, z_min=-1.0, z_max=1.11),
    )

The lifecycle contract
----------------------

.. list-table::
   :header-rows: 1

   * - method
     - contract
   * - ``validate() -> list[str]``
     - every problem with the job as human-readable strings; ``[]`` = runnable
   * - ``build() -> SetupArtifacts``
     - engine-native scene, **offline and deterministic** — no network, no license
   * - ``estimate() -> ResourceEstimate``
     - offline cost hints (cells, memory, number of simulations)
   * - ``run() -> SMatrix``
     - the **only** method that spends money, license seats, or GPU time

.. code-block:: python

    assert solver.validate() == []
    artifacts = solver.build()      # free: script/scene generated locally
    print(solver.estimate())
    smatrix = solver.run()          # cloud credits / license / local compute

Available engines
-----------------

.. list-table::
   :header-rows: 1

   * - engine
     - execution
     - cost
     - install
   * - `Tidy3D <https://github.com/flexcompute/tidy3d>`_ >= 2.11
     - cloud
     - FlexCredits
     - ``pip install gds_fdtd[tidy3d]`` + ``TIDY3D_API_KEY``
   * - Ansys Lumerical FDTD 2024/2025
     - local
     - license
     - Lumerical install with ``lumapi`` on path
   * - `beamz <https://github.com/beamzorg/beamz>`_ >= 0.4
     - local (JAX CPU/GPU)
     - free
     - ``pip install gds_fdtd[beamz]``

``gds-fdtd solvers`` (CLI) lists every registered engine with availability
and, when unavailable, the reason. Per-engine verification against the real
engines lives in ``SOLVER_STATUS.md`` — the three engines agree within
0.052 dB (tidy3d ↔ Lumerical within 0.0033 dB) on an identical job.

Standard visualization flow
---------------------------

Every example follows the same four steps:

.. code-block:: python

    from gds_fdtd.plotting import plot_component, plot_smatrix

    plot_component(component, spec=solver.spec)   # 1: geometry + ports + FDTD region
    solver.build()                                # 2: offline setup (free)
    plot_smatrix(smatrix, kind="db")              # 3: S-parameters
    solver.plot_fields(axis="z")                  # 4: field profile (after run())

Beyond one engine
-----------------

.. code-block:: python

    from gds_fdtd.convergence import sweep
    from gds_fdtd.validation import validate_across

    # principled mesh choice; with cache_dir reruns are free
    report = sweep(get_solver("tidy3d"), component, tech, spec,
                   field="mesh", values=[6, 8, 10], cache_dir=".cache")
    report.recommend(tol_db=0.05)

    # the agnosticism payoff: identical job, several engines, worst |dS| in dB
    report = validate_across(
        [get_solver("tidy3d"), get_solver("lumerical"), get_solver("beamz")],
        component, tech, spec, cache_dir=".cache",
    )

Bring your own engine
---------------------

Any FDTD engine becomes a gds_fdtd solver by implementing the four methods
above — see :doc:`adding_a_solver` for the full guide, including the
conformance test suite your adapter inherits for free.

.. note::

    The pre-0.5 class interface (``fdtd_solver_tidy3d`` /
    ``fdtd_solver_lumerical`` with per-solver keyword arguments) still works
    but is deprecated and will be removed at v1.0. New code should use
    ``get_solver(name)(component, tech, spec)``.
