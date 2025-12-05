"""Centralized logging configuration for the project."""
from __future__ import annotations

import logging
from typing import Optional

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger once."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with shared formatting."""
    configure_logging()
    return logging.getLogger(name if name else __name__)
