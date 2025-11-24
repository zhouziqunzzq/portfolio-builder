from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def configure_logging(base_output: Path, level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Configure application-wide logging.

    Args:
        base_output: Base output directory (logs will go under base_output / "logs").
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: If True, emit logs to a rotating file handler.

    Returns:
        The root logger.
    """
    level_value = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(level_value)

    # Prevent duplicate handlers when re-configuring
    _remove_existing_handlers(logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level_value)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        logs_dir = base_output / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(logs_dir / "production.log", maxBytes=5_000_000, backupCount=3)
        fh.setLevel(level_value)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Reduce verbosity of noisy libraries
    for noisy in ["urllib3", "yfinance", "matplotlib", "numexpr"]:
        logging.getLogger(noisy).setLevel(max(level_value, logging.WARNING))

    logger.debug("Logging configured", extra={"level": level})
    return logger


def _remove_existing_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
