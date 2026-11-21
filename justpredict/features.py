"""Fitur teknikal dari deret harga, dan target return ke depan.

Semua fitur pada baris t hanya memakai data sampai t (tanpa bocoran masa depan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "tech-v3-fund"


from .technicals import rsi as _rsi, ta_features  # noqa: E402


def technical_features(close: pd.Series) -> pd.DataFrame:
    close = close.astype(float)
    logp = np.log(close)
    r1 = logp.diff()
    f = pd.DataFrame(index=close.index)
    for k in (1, 2, 3, 5, 10, 21, 63):
        f[f"ret_{k}"] = logp.diff(k)
    for k in (5, 21, 63):
        f[f"vol_{k}"] = r1.rolling(k).std()
    for k in (5, 20, 50, 200):
        f[f"ma_ratio_{k}"] = close / close.rolling(k).mean() - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / close
    f["macd"] = macd
    f["macd_signal"] = macd - macd.ewm(span=9, adjust=False).mean()
    f["rsi_14"] = _rsi(close, 14) / 100
    roll_max = close.rolling(252, min_periods=21).max()
    roll_min = close.rolling(252, min_periods=21).min()
    f["dist_high_252"] = close / roll_max - 1
    f["dist_low_252"] = close / roll_min - 1
    f["dow"] = close.index.dayofweek
    return f


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return log dari t ke t+horizon hari bursa (NaN untuk baris terakhir)."""
    logp = np.log(close.astype(float))
    return logp.shift(-horizon) - logp


def build_dataset(close: pd.Series, horizon: int,
                  extra: pd.DataFrame | None = None,
                  ohlcv: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Kembalikan (X_latih, y_latih, X_terbaru).

    `ohlcv` = data harian lengkap untuk fitur analisis teknikal (pola candle, support/resistance, dll.).
    `extra` = fitur tambahan ber-index tanggal (mis. sentimen, fundamental), digabung apa adanya.
    """
    X = technical_features(close)
    if ohlcv is not None and not ohlcv.empty:
        X = X.join(ta_features(ohlcv.reindex(close.index)))
    valid = X.notna().all(axis=1)
    if extra is not None and not extra.empty:
        # Fitur tambahan boleh kosong (LightGBM menangani NaN), jadi tidak ikut memangkas baris latih.
        ex = extra.reindex(extra.index.union(X.index)).sort_index().ffill().reindex(X.index)
        X = X.join(ex.add_prefix("x_"))
    y = forward_return(close, horizon)
    X_latest = X.loc[valid].iloc[[-1]] if valid.any() else X.iloc[0:0]
    train_mask = valid & y.notna()
    return X.loc[train_mask], y.loc[train_mask], X_latest
