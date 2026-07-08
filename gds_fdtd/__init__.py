"""gds_fdtd Top-level package imports."""

from . import core, lyprocessor, simprocessor, sparams
from .core import technology

__all__ = ["core", "lyprocessor", "simprocessor", "sparams", "technology"]

__author__ = """Mustafa Hammood"""
__email__ = "mustafa@siepic.com"
__version__ = "0.4.0"
