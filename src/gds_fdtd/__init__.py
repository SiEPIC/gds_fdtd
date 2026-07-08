"""gds_fdtd Top-level package imports."""

import logging as _logging
from importlib.metadata import PackageNotFoundError, version

from . import (
    caching,
    convergence,
    core,
    geometry,
    grid,
    lyprocessor,
    simprocessor,
    sparams,
    validation,
)
from .core import technology
from .geometry import Component, LayoutSource, Port, Region, Structure
from .smatrix import SMatrix

# Library logging etiquette: emit nothing unless the application (or
# logging_config.setup_logging) configures handlers.
_logging.getLogger("gds_fdtd").addHandler(_logging.NullHandler())

__all__ = [
    "Component",
    "LayoutSource",
    "Port",
    "Region",
    "SMatrix",
    "Structure",
    "caching",
    "convergence",
    "core",
    "geometry",
    "grid",
    "lyprocessor",
    "simprocessor",
    "sparams",
    "technology",
    "validation",
]

__author__ = """Mustafa Hammood"""
__email__ = "mustafa@siepic.com"

try:
    __version__ = version("gds_fdtd")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"
