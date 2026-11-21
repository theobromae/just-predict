"""LightGBM: regresi kuantil untuk rentang return + klasifikasi arah."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster

_COMMON = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


class LightGBMForecaster(Forecaster):
    name = "lightgbm"
    version = "1"

    def __init__(self, recent_weight_halflife: int | None = 504, seed: int = 42, n_estimators: int = 300):
        # Bobot sampel menurun eksponensial ke masa lalu: data terbaru lebih berpengaruh.
        self.halflife = recent_weight_halflife
        self.seed = seed
        self.params = {**_COMMON, "n_estimators": n_estimators}

    def _weights(self, n: int) -> np.ndarray | None:
        if not self.halflife:
            return None
        age = np.arange(n)[::-1]
        return 0.5 ** (age / self.halflife)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import lightgbm as lgb

        w = self._weights(len(X))
        self.quantile_models_ = {}
        for q in (0.1, 0.5, 0.9):
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed, **self.params)
            m.fit(X, y, sample_weight=w)
            self.quantile_models_[q] = m
        self.clf_ = lgb.LGBMClassifier(objective="binary", random_state=self.seed, **self.params)
        self.clf_.fit(X, (y > 0).astype(int), sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        qs = np.column_stack([self.quantile_models_[q].predict(X) for q in (0.1, 0.5, 0.9)])
        qs.sort(axis=1)  # cegah kuantil tertukar (quantile crossing)
        p_up = self.clf_.predict_proba(X)[:, 1]
        return pd.DataFrame({"p_up": p_up, "q10": qs[:, 0], "q50": qs[:, 1], "q90": qs[:, 2]}, index=X.index)
