"""
Centralised logging setup.

Call ``setup_logging(settings)`` once at application startup.
All subsequent ``logging.getLogger(__name__)`` calls will inherit the
configured handlers and level automatically.

Usage
-----
    from cloudremoval.config import get_settings
    from cloudremoval.utils import setup_logging

    settings = get_settings()
    setup_logging(settings)

    import logging
    log = logging.getLogger(__name__)
    log.info("Application started")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloudremoval.config.settings import Settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(settings: "Settings") -> logging.Logger:
    """Configure root logger with console + rotating-file handlers.

    Parameters
    ----------
    settings:
        Application settings instance.  Uses ``settings.LOG_LEVEL``,
        ``settings.LOG_DIR``, and ``settings.LOG_FILE``.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    # Ensure log directory exists
    log_dir: Path = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    level: int = getattr(logging, settings.LOG_LEVEL, logging.INFO)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console handler -------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # --- File handler (rotating, max 10 MB, keep 5 backups) --------
    log_file: Path = log_dir / settings.LOG_FILE
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # --- Root logger -----------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    else:
        # Replace existing handlers (idempotent re-init)
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers at WARNING by default
    for noisy in ("rasterio", "fiona", "PIL", "urllib3", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.info(
        "Logging initialised | level=%s | file=%s",
        settings.LOG_LEVEL,
        log_file,
    )
    return root_logger
