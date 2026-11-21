"""Konfigurasi aset, horizon, dan lokasi data."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "JustPredict"

# Horizon dalam jumlah hari bursa ke depan.
HORIZONS: dict[str, int] = {"harian": 1, "mingguan": 5, "bulanan": 21}

TROY_OUNCE_GRAMS = 31.1034768


@dataclass(frozen=True)
class Asset:
    key: str            # id internal, dipakai di database
    name: str           # nama tampilan
    ticker: str | None  # ticker Yahoo Finance; None untuk aset turunan
    currency: str
    derived_from: tuple[str, ...] = field(default_factory=tuple)


ASSETS: list[Asset] = [
    Asset("WTI", "Minyak WTI", "CL=F", "USD"),
    Asset("IHSG", "IHSG", "^JKSE", "IDR"),
    Asset("USDIDR", "Rupiah per Dolar AS", "IDR=X", "IDR"),
    Asset("GOLD_USD", "Emas (USD/oz)", "GC=F", "USD"),
    Asset("GOLD_IDR_G", "Emas (Rp/gram)", None, "IDR", derived_from=("GOLD_USD", "USDIDR")),
    Asset("BTC", "Bitcoin (USD)", "BTC-USD", "USD"),
    Asset("TLKM", "Telkom Indonesia", "TLKM.JK", "IDR"),
    Asset("ADRO", "Alamtri Resources (ADRO)", "ADRO.JK", "IDR"),
    Asset("BRPT", "Barito Pacific", "BRPT.JK", "IDR"),
]

ASSET_BY_KEY = {a.key: a for a in ASSETS}

# Aset yang ditampilkan dan diprediksi (GOLD_USD hanya bahan hitung emas per gram).
PREDICT_ASSETS = [a.key for a in ASSETS if a.key != "GOLD_USD"]

HISTORY_START = "2012-01-01"


def data_dir() -> Path:
    """Folder data pengguna. Bisa diganti lewat env JUSTPREDICT_HOME."""
    override = os.environ.get("JUSTPREDICT_HOME")
    if override:
        p = Path(override)
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        p = Path.home() / f".{APP_NAME.lower()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return data_dir() / "justpredict.sqlite"
