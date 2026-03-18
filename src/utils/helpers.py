"""Utility functions and configuration."""
import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """Setup console logger with optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(levelname)-8s %(name)s: %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
