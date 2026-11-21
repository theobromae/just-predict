"""LightGBM: regresi kuantil untuk rentang return + klasifikasi arah."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster

# Parameter API asli LightGBM (tanpa scikit-learn, supaya aplikasi lebih kecil).
_COMMON = dict(
    learning_rate=0.03,
    num_leaves=15,
    min_data_in_leaf=40,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    lambda_l2=1.0,
    verbose=-1,
)


class LightGBMForecaster(Forecaster):
    name = "lightgbm"
    version = "2"

    def __init__(self, recent_weight_halflife: int | None = 504, seed: int = 42, n_estimators: int = 300):
        # Bobot sampel menurun eksponensial ke masa lalu: data terbaru lebih berpengaruh.
        self.halflife = recent_weight_halflife
        self.seed = seed
        self.n_estimators = n_estimators
        self.params = {**_COMMON, "seed": seed}

    def _weights(self, n: int) -> np.ndarray | None:
        if not self.halflife:
            return None
        age = np.arange(n)[::-1]
        return 0.5 ** (age / self.halflife)

    def _train(self, X: pd.DataFrame, y, w, **params):
        import lightgbm as lgb
        ds = lgb.Dataset(X, label=np.asarray(y, dtype=float), weight=w, free_raw_data=False)
        return lgb.train({**self.params, **params}, ds, num_boost_round=self.n_estimators)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        w = self._weights(len(X))
        self.quantile_models_ = {q: self._train(X, y, w, objective="quantile", alpha=q) for q in (0.1, 0.5, 0.9)}
        self.clf_ = self._train(X, (y > 0).astype(int), w, objective="binary")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        qs = np.column_stack([self.quantile_models_[q].predict(X) for q in (0.1, 0.5, 0.9)])
        qs.sort(axis=1)  # cegah kuantil tertukar (quantile crossing)
        p_up = self.clf_.predict(X)  # objective binary: langsung peluang
        return pd.DataFrame({"p_up": p_up, "q10": qs[:, 0], "q50": qs[:, 1], "q90": qs[:, 2]}, index=X.index)
