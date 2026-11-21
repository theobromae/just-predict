"""Pengambilan harga harian secara inkremental."""
from __future__ import annotations

import logging
from typing import Callable, Protocol

import pandas as pd

from ..config import ASSETS, HISTORY_START, TROY_OUNCE_GRAMS
from ..db import Store

log = logging.getLogger(__name__)


class PriceSource(Protocol):
    def fetch(self, ticker: str, start: str) -> pd.Series:
        """Harga penutupan harian sejak `start` (inklusif), index tanggal."""


class YahooSource:
    """Harga dari Yahoo Finance lewat yfinance (gratis, tanpa API key)."""

    def fetch(self, ticker: str, start: str) -> pd.Series:
        import yfinance as yf

        df = yf.download(ticker, start=start, progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance baru mengembalikan kolom MultiIndex
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return close.dropna().astype(float)


def _start_date(store: Store, asset: str, overlap_days: int = 7) -> str:
    last = store.last_price_date(asset)
    if last is None:
        return HISTORY_START
    # Ambil ulang beberapa hari terakhir: harga hari berjalan bisa berubah setelah penutupan.
    return (pd.Timestamp(last) - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")


def gold_idr_per_gram(gold_usd_oz: pd.Series, usdidr: pd.Series) -> pd.Series:
    """Emas Rp/gram = harga emas USD/troy oz × kurs ÷ 31,1035.

    Kurs diisi maju (ffill) untuk hari ketika pasar emas buka tapi kurs belum ada.
    """
    df = pd.concat({"g": gold_usd_oz, "fx": usdidr}, axis=1).sort_index()
    df["fx"] = df["fx"].ffill()
    df = df.dropna()
    return df["g"] * df["fx"] / TROY_OUNCE_GRAMS


def update_all(store: Store, source: PriceSource | None = None,
               progress: Callable[[str], None] = lambda m: None) -> dict[str, int]:
    """Perbarui semua aset. Mengembalikan jumlah baris yang ditulis per aset.

    Aset yang gagal diambil dilewati (dicatat di log); data lama tetap dipakai.
    """
    source = source or YahooSource()
    written: dict[str, int] = {}
    for asset in ASSETS:
        if asset.ticker is None:
            continue
        progress(f"Mengambil data {asset.name}…")
        try:
            s = source.fetch(asset.ticker, _start_date(store, asset.key))
        except Exception as exc:  # jaringan putus, ticker berubah, dll.
            log.warning("Gagal mengambil %s: %s", asset.key, exc)
            progress(f"Gagal mengambil {asset.name}, memakai data tersimpan.")
            continue
        written[asset.key] = store.upsert_prices(asset.key, s) if not s.empty else 0

    for asset in ASSETS:
        if asset.derived_from == ("GOLD_USD", "USDIDR"):
            derived = gold_idr_per_gram(store.prices("GOLD_USD"), store.prices("USDIDR"))
            written[asset.key] = store.upsert_prices(asset.key, derived) if not derived.empty else 0
    return written
