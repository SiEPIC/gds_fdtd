Architecture Overview
=====================

Here's how the different parts of ``gds_fdtd`` work together to run FDTD simulations.

Main Modules
------------

``core``
    Contains the basic data structures and functions that everything else uses.
    This is where ports, components, and geometry are defined.

``lyprocessor``
    Takes GDS files and turns them into something the simulation can understand.
    Uses KLayout to read the layout files and extract the shapes.

``simprocessor``
    Puts everything together. Takes your technology file and GDS layout
    and creates a simulation setup.

``sparams``
    Handles S-parameter calculations after the simulation runs. Can export
    results to different file formats.

``logging_config``
    Keeps track of what happens during simulations. Writes detailed logs
    to files so you can debug problems.

How the Solvers Work
--------------------

``solver`` (Base Class)
    The foundation that both Tidy3D and Lumerical solvers build on.
    Handles common tasks like port setup and parameter checking.

``solver_tidy3d``
    Runs simulations on Tidy3D's cloud platform. Uses their ComponentModeler
    to get accurate S-parameters. Works with multiple polarizations.

``solver_lumerical``
    Interfaces with Lumerical FDTD on your local machine or cluster.
    Can use GPU acceleration and handles complex layer stacks.

What Makes It Useful
--------------------

The package is built so you can easily switch between different solvers without rewriting your simulation setup. Both Tidy3D and Lumerical solvers use the same interface, so your code stays mostly the same.

You can run simulations with both TE and TM polarizations at once, which is handy for analyzing things like polarization beam splitters or mode converters.

Everything gets logged automatically, so when something goes wrong (and it will), you have detailed information to figure out what happened.

The examples show you how to use all these pieces together for real photonic device simulations.
