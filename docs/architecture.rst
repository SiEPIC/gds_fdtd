Architecture Overview
=====================

GDS in, S-parameters out — with every engine behind one contract.

.. code-block:: text

    layout (GDS / gdsfactory / SiEPIC)
        │  lyprocessor · layout.gdsfactory        ports auto-detected
        ▼
    Component  ─────────  Technology (YAML, schema v2: named materials)
        │
        ▼
    Solver ABC (solvers.base) ── registry + entry points
        │        validate() · build() · estimate()   [offline, free]
        │        run()                                [the only spending step]
        ▼
    SMatrix ── checks (reciprocity/passivity) · .dat / Touchstone / HDF5 / npz
        │
        ▼
    plotting · convergence.sweep · validation.validate_across · caching

Core modules
------------

``geometry``
    ``Component``, ``Structure`` (flat, role-tagged: device / substrate /
    superstrate), ``Port`` (with port-extension stubs through the boundary),
    ``Region``.

``technology`` / ``materials.rii``
    Pydantic-validated technology files. Schema v2 defines named materials
    once (neutral ``nk``/``rii`` + per-engine hints) and layers reference
    them; ``gds-fdtd convert-tech`` migrates v1 files.

``lyprocessor`` / ``simprocessor`` / ``layout.gdsfactory``
    GDS loading via KLayout (SiEPIC pin conventions), component assembly,
    and gdsfactory (>= 9) conversion.

``spec``
    ``SimulationSpec`` — every numeric simulation setting, validated, in
    package-wide units (µm / degrees / Hz).

``smatrix``
    The canonical ``SMatrix`` container plus the internal Lumerical-format
    ``.dat`` reader/writer (``_sparams``) it interoperates with.

Solver layer
------------

``solvers.base``
    The ``Solver`` ABC and registry. Constructors are cheap and pure;
    ``validate``/``build``/``estimate`` are offline; only ``run()`` spends.
    Third-party engines register via the ``gds_fdtd.solvers`` entry-point
    group (see :doc:`adding_a_solver`).

``solvers.tidy3d`` / ``solvers.lumerical`` / ``solvers.beamz``
    The engine adapters. The tidy3d adapter drives an internal scene builder
    (``solvers/_tidy3d_base`` + ``solvers/_tidy3d_engine``); the Lumerical
    adapter generates ``.lsf`` directly; beamz builds a JAX FDTD grid. The
    pre-0.5 ``solver_tidy3d`` / ``solver_lumerical`` modules were removed in
    0.6.0.

``grid`` / ``modes`` / ``extraction``
    The kernel-engine pipeline: permittivity rasterization with sub-pixel
    averaging, local mode solving, and bidirectional mode-overlap
    extraction — for engines that only accept raw permittivity arrays.

Orchestration
-------------

``convergence`` / ``caching`` / ``validation``
    Field sweeps with converged-value recommendation; job-hash result
    caching (repeat runs are free); cross-engine agreement reports.

``execution`` / ``cli``
    Serializable ``JobSpec`` + local/subprocess backends and the
    ``gds-fdtd`` command-line interface — the remote-compute surface
    (see :doc:`remote_compute`).

``errors`` / ``settings`` / ``logging_config``
    One exception hierarchy (``GdsFdtdError``), ``GDS_FDTD_*`` environment
    configuration, and package-scoped logging (text or JSON lines).
