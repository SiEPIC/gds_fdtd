"""gds_fdtd Top-level package imports."""

from importlib.metadata import PackageNotFoundError, version

from . import core, lyprocessor, simprocessor, sparams
from .core import technology

__all__ = ["core", "lyprocessor", "simprocessor", "sparams", "technology"]

__author__ = """Mustafa Hammood"""
__email__ = "mustafa@siepic.com"

try:
    __version__ = version("gds_fdtd")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"
