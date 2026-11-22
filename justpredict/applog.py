"""Log ke file agar crash bisa dilacak (aplikasi .app tidak punya jendela terminal)."""
from __future__ import annotations

import faulthandler
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import APP_NAME, data_dir

_fault_file = None


def log_dir() -> Path:
    if sys.platform == "darwin":
        p = Path.home() / "Library" / "Logs" / APP_NAME
    else:
        p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_file() -> Path:
    return log_dir() / "justpredict.log"


def setup_logging() -> None:
    global _fault_file
    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(log_file(), maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    # Crash tingkat rendah (segfault di Qt/LightGBM) juga dicatat.
    _fault_file = open(log_dir() / "crash.log", "a", encoding="utf-8")
    faulthandler.enable(_fault_file)
    logging.getLogger(__name__).info("Mulai %s, Python %s, argv=%s", APP_NAME, sys.version.split()[0], sys.argv)
