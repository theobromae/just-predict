"""Pengambilan harga harian secara inkremental."""
from __future__ import annotations

import logging
from typing import Callable, Protocol

import pandas as pd

from ..config import ASSETS, HISTORY_START, TROY_OUNCE_GRAMS
from ..db import Store

log = logging.getLogger(__name__)


class PriceSource(Protocol):
    def fetch(self, ticker: str, start: str) -> pd.Series | pd.DataFrame:
        """Harga harian sejak `start` (inklusif), index tanggal.

        Deret close, atau DataFrame berkolom open/high/low/close/volume.
        """


class YahooSource:
    """Harga dari Yahoo Finance lewat yfinance (gratis, tanpa API key)."""

    def fetch(self, ticker: str, start: str) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(ticker, start=start, progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if isinstance(df.columns, pd.MultiIndex):  # yfinance baru: kolom (field, ticker)
            df = df.xs(ticker, axis=1, level=-1) if ticker in df.columns.get_level_values(-1) else df.droplevel(-1, axis=1)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df.dropna(subset=["close"]).astype(float)


def _start_date(store: Store, asset: str, overlap_days: int = 7) -> str:
    last = store.last_price_date(asset)
    if last is None or store.missing_ohlc(asset):  # data lama tanpa OHLC: ambil ulang semua
        return HISTORY_START
    # Ambil ulang beberapa hari terakhir: harga hari berjalan bisa berubah setelah penutupan.
    return (pd.Timestamp(last) - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")


def gold_idr_per_gram(gold_usd_oz: pd.Series | pd.DataFrame, usdidr: pd.Series) -> pd.Series | pd.DataFrame:
    """Emas Rp/gram = harga emas USD/troy oz × kurs ÷ 31,1035.

    Kurs diisi maju (ffill) untuk hari ketika pasar emas buka tapi kurs belum ada.
    Kalau `gold_usd_oz` berupa OHLC, open/high/low/close dikonversi dengan kurs penutupan hari itu.
    """
    if isinstance(gold_usd_oz, pd.Series):
        df = pd.concat({"g": gold_usd_oz, "fx": usdidr}, axis=1).sort_index()
        df["fx"] = df["fx"].ffill()
        df = df.dropna()
        return df["g"] * df["fx"] / TROY_OUNCE_GRAMS
    fx = usdidr.reindex(gold_usd_oz.index.union(usdidr.index)).sort_index().ffill().reindex(gold_usd_oz.index)
    out = gold_usd_oz[["open", "high", "low", "close"]].mul(fx / TROY_OUNCE_GRAMS, axis=0)
    return out.dropna(subset=["close"])


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
        written[asset.key] = store.upsert_prices(asset.key, s) if len(s) else 0

    for asset in ASSETS:
        if asset.derived_from == ("GOLD_USD", "USDIDR"):
            derived = gold_idr_per_gram(store.ohlcv("GOLD_USD"), store.prices("USDIDR"))
            written[asset.key] = store.upsert_prices(asset.key, derived) if not derived.empty else 0
    return written
