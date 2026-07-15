"""
Logging configuration for gds_fdtd package.

Provides centralized logging setup with file output to working directory.
@author: Mustafa Hammood, 2025
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """One JSON object per line — for Modal/AWS/Batch log aggregation.

    Selected with ``GDS_FDTD_LOG_FORMAT=json`` (see gds_fdtd.settings).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "func": record.funcName,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(working_dir: str = "./", component_name: str = "gds_fdtd") -> logging.Logger:
    """
    Set up file + console logging for the ``gds_fdtd`` package logger ONLY.

    A library must never reconfigure the root logger; the previous
    implementation cleared all root handlers and set root to DEBUG on every
    solver construction, hijacking the host application's logging (bug B16).
    This configures the ``gds_fdtd`` logger with its own handlers and disables
    propagation so records are not duplicated through root.

    Args:
        working_dir: Directory where the log file will be created.
        component_name: Name of the component (for the log filename).

    Returns:
        The ``gds_fdtd`` package logger.
    """
    # Ensure working directory exists
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{component_name}_{timestamp}.log"
    log_filepath = os.path.join(working_dir, log_filename)

    package_logger = logging.getLogger("gds_fdtd")
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False

    # Remove only OUR previous handlers (re-configuration between solver runs)
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
        handler.close()

    from .settings import settings

    if settings().log_format == "json":
        detailed_formatter: logging.Formatter = JsonFormatter()
        console_formatter: logging.Formatter = JsonFormatter()
    else:
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_formatter = logging.Formatter("%(levelname)-8s | %(name)-15s | %(message)s")

    file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    package_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings().log_level.upper(), logging.INFO))
    console_handler.setFormatter(console_formatter)
    package_logger.addHandler(console_handler)

    package_logger.info(f"Logging initialized - Log file: {log_filepath}")
    package_logger.info(f"Working directory: {os.path.abspath(working_dir)}")

    return package_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_separator(logger: logging.Logger, title: str = "") -> None:
    """
    Log a separator line for better log readability.

    Args:
        logger: Logger instance
        title: Optional title for the separator
    """
    separator = "=" * 60
    if title:
        logger.info(separator)
        logger.info(f"  {title}")
        logger.info(separator)
    else:
        logger.info(separator)


def log_dict(logger: logging.Logger, data: dict[str, Any], title: str = "Configuration") -> None:
    """
    Log dictionary data in a formatted way.

    Args:
        logger: Logger instance
        data: Dictionary to log
        title: Title for the data
    """
    logger.info(f"{title}:")
    for key, value in data.items():
        logger.info(f"  {key}: {value}")


def log_simulation_start(logger: logging.Logger, solver_type: str, component_name: str) -> None:
    """Log simulation start with details."""
    log_separator(logger, f"STARTING {solver_type.upper()} SIMULATION")
    logger.info(f"Component: {component_name}")
    logger.info(f"Solver: {solver_type}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def log_simulation_complete(logger: logging.Logger, solver_type: str) -> None:
    """Log simulation completion."""
    log_separator(logger, f"{solver_type.upper()} SIMULATION COMPLETE")
    logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
