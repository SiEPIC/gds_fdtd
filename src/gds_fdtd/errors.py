"""
gds_fdtd simulation toolbox.

Exception hierarchy (WP7.6). Every gds_fdtd-raised error derives from
``GdsFdtdError`` so applications can catch the whole package with one
except-clause — while each subclass ALSO derives from the builtin type the
code historically raised (ValueError, RuntimeError, ...), so no existing
caller breaks.

The CLI maps this hierarchy onto exit codes:
JobValidationError -> 2 · SolverUnavailableError -> 3 ·
BudgetExceededError -> 4 · anything else -> 1.
"""

from __future__ import annotations


class GdsFdtdError(Exception):
    """Base class for every error gds_fdtd raises deliberately."""


class TechnologyError(GdsFdtdError, ValueError):
    """A technology file/model is invalid (bad layer stack, material, ...)."""


class LayoutError(GdsFdtdError, ValueError):
    """A layout could not be loaded or interpreted (missing cell, ports, ...)."""


class JobValidationError(GdsFdtdError, ValueError):
    """A simulation job failed validate() — fix the job, don't retry."""


class SolverError(GdsFdtdError, RuntimeError):
    """An engine adapter failed at run/build time.

    Adapters wrap native engine exceptions in this (``raise ... from e``)
    so callers see one type with the native message attached.
    """


class SolverUnavailableError(GdsFdtdError, RuntimeError):
    """The requested engine is not usable here (not installed, no license)."""


class BudgetExceededError(GdsFdtdError, PermissionError):
    """A spending limit (Budget) refused the operation before any cost."""
