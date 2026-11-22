"""Antarmuka model. Model baru cukup mewarisi `Forecaster` dan didaftarkan di `registry.py`.

Setiap model memprediksi distribusi return log ke depan:
  p_up (peluang naik), q10, q50, q90 (kuantil return).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ["p_up", "q10", "q50", "q90"]


class Forecaster(ABC):
    name: str = "base"
    version: str = "0"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """DataFrame dengan kolom OUTPUT_COLUMNS, index sama dengan X."""


class NaiveForecaster(Forecaster):
    """Baseline: harga tidak berubah (median return 0).

    Rentang dan peluang naik diambil dari distribusi return historis 2 tahun terakhir.
    Model ML hanya dianggap berguna kalau mengalahkan baseline ini.
    """

    name = "naive"
    version = "1"

    def __init__(self, window: int = 504):
        self.window = window

    def fit(self, X, y):
        recent = y.iloc[-self.window:]
        self.q10_, self.q90_ = (float(np.quantile(recent, 0.1)), float(np.quantile(recent, 0.9))) if len(recent) else (0.0, 0.0)
        self.p_up_ = float((recent > 0).mean()) if len(recent) else 0.5
        return self

    def predict(self, X):
        n = len(X)
        return pd.DataFrame(
            {"p_up": [self.p_up_] * n, "q10": [self.q10_] * n, "q50": [0.0] * n, "q90": [self.q90_] * n},
            index=X.index,
        )
