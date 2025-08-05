"""
Logging configuration for gds_fdtd package.

Provides centralized logging setup with file output to working directory.
@author: Mustafa Hammood, 2025
"""
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(working_dir: str = "./", component_name: str = "gds_fdtd"):
    """
    Setup logging for the entire gds_fdtd package.
    
    Args:
        working_dir: Directory where log file will be created
        component_name: Name of the component (for log filename)
        
    Returns:
        Logger instance for the package
    """
    # Ensure working directory exists
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{component_name}_{timestamp}.log"
    log_filepath = os.path.join(working_dir, log_filename)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(name)-15s | %(message)s'
    )
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Get package logger
    package_logger = logging.getLogger('gds_fdtd')
    package_logger.info(f"Logging initialized - Log file: {log_filepath}")
    package_logger.info(f"Working directory: {os.path.abspath(working_dir)}")
    
    return package_logger

def get_logger(name: str):
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_separator(logger, title: str = ""):
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

def log_dict(logger, data: dict, title: str = "Configuration"):
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

def log_simulation_start(logger, solver_type: str, component_name: str):
    """Log simulation start with details."""
    log_separator(logger, f"STARTING {solver_type.upper()} SIMULATION")
    logger.info(f"Component: {component_name}")
    logger.info(f"Solver: {solver_type}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def log_simulation_complete(logger, solver_type: str):
    """Log simulation completion."""
    log_separator(logger, f"{solver_type.upper()} SIMULATION COMPLETE")
    logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 