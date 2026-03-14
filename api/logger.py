"""
Centralized logger for the DS-Agent service.

Usage
-----
    from api.logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")

Log level is controlled by LOG_LEVEL in api/.env  (default: INFO).
Set LOG_LEVEL=DEBUG for verbose sandbox output.

Output goes to stderr so it appears alongside uvicorn's own access logs
in the same terminal window.
"""
from __future__ import annotations

import logging
import os
import sys

_ROOT = "ds_agent"
_FMT  = "%(asctime)s  %(levelname)-8s  %(message)s"
_DATE = "%H:%M:%S"


def _setup() -> None:
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    root.setLevel(level)
    # stderr — same stream uvicorn uses, so everything appears together in the terminal
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root.addHandler(handler)
    root.propagate = False


_setup()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ds_agent hierarchy."""
    short = name.removeprefix("api.").removeprefix("api/")
    return logging.getLogger(f"{_ROOT}.{short}")
