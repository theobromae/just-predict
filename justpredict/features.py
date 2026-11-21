"""Fitur teknikal dari deret harga, dan target return ke depan.

Semua fitur pada baris t hanya memakai data sampai t (tanpa bocoran masa depan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = "tech-v1"


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up / down  # down=0 -> inf -> RSI 100
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(~((up == 0) & (down == 0)), 50.0).where(delta.notna() | rsi.notna())


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
                  extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Kembalikan (X_latih, y_latih, X_terbaru).

    `extra` = fitur tambahan ber-index tanggal (mis. sentimen, fundamental), digabung apa adanya.
    """
    X = technical_features(close)
    if extra is not None and not extra.empty:
        X = X.join(extra.reindex(X.index).ffill())
    y = forward_return(close, horizon)
    valid = X.notna().all(axis=1)
    X_latest = X.loc[valid].iloc[[-1]] if valid.any() else X.iloc[0:0]
    train_mask = valid & y.notna()
    return X.loc[train_mask], y.loc[train_mask], X_latest
