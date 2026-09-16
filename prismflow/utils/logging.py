"""Shared logging setup for PrismFlow."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: str = "prismflow", log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    If log_dir is given, also writes to <log_dir>/<name>.log (directory is
    created if needed). Safe to call multiple times for the same name.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir_path / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
