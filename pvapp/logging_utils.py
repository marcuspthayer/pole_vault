import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = "pv",
    log_file: str | None = None,
    level: int = logging.INFO,
    max_bytes: int = 2_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """Create (or return) a configured logger.

    - Logs to console (Streamlit terminal) and optionally to a rotating file.
    - Safe to call multiple times; handlers won't be duplicated.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (Streamlit terminal)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Rotating file handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
            fh = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
