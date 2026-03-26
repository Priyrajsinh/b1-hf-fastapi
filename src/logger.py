"""Centralized logging configuration for the B1 project.

Provides a factory function that returns a configured logger.
In production (ENV=production) uses JSON formatting via python-json-logger.
In development uses a human-readable format.
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Create and return a configured logger instance.

    In production (ENV=production), attaches a JSON formatter using
    pythonjsonlogger so logs can be parsed by log aggregators.
    In all other environments, uses a human-readable format.

    Checks for existing handlers to prevent duplicate log entries when
    called multiple times with the same logger name.

    Args:
        name: The logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    env = os.environ.get("ENV", "development")

    handler = logging.StreamHandler()
    formatter: logging.Formatter
    if env == "production":
        from pythonjsonlogger import jsonlogger

        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
        )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
