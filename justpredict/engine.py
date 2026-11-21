"""Siklus utama: update data -> evaluasi -> latih ulang -> prediksi."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import adaptive
from .config import ASSET_BY_KEY, HORIZONS, PREDICT_ASSETS
from .data.prices import PriceSource, update_all, update_intraday
from .db import Store
from .features import FEATURE_VERSION, build_dataset
from .models import make_models
from .models.gbm import LightGBMForecaster

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 300
BACKTEST_FRACTION = 0.2
ENGINE_VERSION = "0.2.0"

# Jalur prediksi: (jarak antar titik dalam hari bursa, jumlah titik) per tampilan.
PATH_POINTS: dict[str, tuple[int, int]] = {"harian": (1, 14), "mingguan": (5, 8), "bulanan": (21, 6)}
# Horizon yang benar-benar dilatih; titik di antaranya diinterpolasi.
PATH_ANCHORS: list[int] = [3, 10, 14, 20, 30, 40, 42, 63, 126]


@dataclass
class CycleReport:
    updated: dict[str, int] = field(default_factory=dict)
    evaluated: int = 0
    predicted: int = 0
    skipped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- evaluasi
def evaluate_matured(store: Store) -> int:
    """Bandingkan prediksi yang tanggal targetnya sudah lewat dengan harga aktual."""
    pending = store.pending_predictions()
    count = 0
    price_cache: dict[str, pd.Series] = {}
    for row in pending.itertuples():
        prices = price_cache.setdefault(row.asset, store.prices(row.asset))
        after = prices.loc[prices.index > pd.Timestamp(row.base_date)]
        if len(after) < row.horizon:
            continue  # belum jatuh tempo
        target_date = after.index[row.horizon - 1]
        actual = float(after.iloc[row.horizon - 1])
        actual_ret = math.log(actual / row.base_price)
        pred_price = row.base_price * math.exp(row.q50)
        raw_q50 = row.raw_q50 if row.raw_q50 is not None and not pd.isna(row.raw_q50) else row.q50
        store.save_evaluation(
            prediction_id=int(row.id),
            target_date=target_date.strftime("%Y-%m-%d"),
            actual_price=actual,
            actual_ret=actual_ret,
            error=actual_ret - row.q50,
            raw_error=actual_ret - raw_q50,
            abs_pct_error=abs(pred_price - actual) / actual,
            direction_hit=int((row.p_up > 0.5) == (actual_ret > 0)),
            in_range=int(row.q10 <= actual_ret <= row.q90),
        )
        count += 1
    return count


# ---------------------------------------------------------------- backtest
def _backtest(model_factory, X: pd.DataFrame, y: pd.Series, horizon: int) -> dict:
    n = len(X)
    split = int(n * (1 - BACKTEST_FRACTION))
    train_end = split - horizon  # jeda agar target latih tidak tumpang tindih dengan data uji
    if train_end < MIN_TRAIN_ROWS // 2:
        return {}
    m = model_factory().fit(X.iloc[:train_end], y.iloc[:train_end])
    p = m.predict(X.iloc[split:])
    yt = y.iloc[split:]
    return {
        "n": len(yt),
        "hit_rate": float(((p["p_up"] > 0.5) == (yt > 0)).mean()),
        "mae_ret": float((yt - p["q50"]).abs().mean()),
        "coverage": float(((yt >= p["q10"]) & (yt <= p["q90"])).mean()),
    }


# ---------------------------------------------------------------- prediksi
def predict_asset(store: Store, asset: str, horizon: int, model_names: list[str] | None = None,
                  extra_features: pd.DataFrame | None = None) -> bool:
    close = store.prices(asset)
    if len(close) < MIN_TRAIN_ROWS:
        return False
    X, y, X_latest = build_dataset(close, horizon, extra_features, store.ohlcv(asset))
    if len(X) < MIN_TRAIN_ROWS or X_latest.empty:
        return False
    base_date = X_latest.index[-1]
    base_price = float(close.loc[base_date])
    base_str = base_date.strftime("%Y-%m-%d")

    models = make_models(model_names)
    member_preds: dict[str, pd.Series] = {}
    backtest_mae: dict[str, float] = {}
    for m in models:
        bt = _backtest(type(m), X, y, horizon)
        if bt:
            store.save_backtest(asset, horizon, m.name, **bt)
            backtest_mae[m.name] = bt["mae_ret"]
        else:
            backtest_mae[m.name] = float(y.abs().mean())
        m.fit(X, y)
        pred = m.predict(X_latest).iloc[0]
        member_preds[m.name] = pred
        store.save_prediction(
            asset=asset, base_date=base_str, horizon=horizon, model=m.name, base_price=base_price,
            p_up=float(pred.p_up), q10=float(pred.q10), q50=float(pred.q50), q90=float(pred.q90),
            raw_q50=float(pred.q50), model_version=f"{m.name}-{m.version}/{FEATURE_VERSION}",
        )

    live = {}
    for name in member_preds:
        ev = store.evaluated(asset, horizon, name)
        live[name] = (ev["actual_ret"] - ev["q50"]).abs() if not ev.empty else pd.Series(dtype=float)
    weights = adaptive.member_weights(live, backtest_mae)
    combined = adaptive.combine(member_preds, weights)

    correction = adaptive.error_correction(store.evaluated(asset, horizon, "ensemble"))
    final = correction.apply(combined)

    store.save_prediction(
        asset=asset, base_date=base_str, horizon=horizon, model="ensemble", base_price=base_price,
        p_up=float(np.clip(final.p_up, 0, 1)), q10=float(final.q10), q50=float(final.q50),
        q90=float(final.q90), raw_q50=float(combined.q50),
        model_version=(f"ensemble-{ENGINE_VERSION} w=" + ",".join(f"{k}:{v:.2f}" for k, v in weights.items())
                       + f" bias={correction.bias:+.4f} scale={correction.width_scale:.2f}"),
    )
    return True


def predict_path_anchor(store: Store, asset: str, horizon: int) -> bool:
    """Prediksi titik jalur (model LightGBM ringan, tanpa backtest) + koreksi galat dari riwayatnya sendiri."""
    close = store.prices(asset)
    if len(close) < MIN_TRAIN_ROWS + horizon:
        return False
    X, y, X_latest = build_dataset(close, horizon, None, store.ohlcv(asset))
    if len(X) < MIN_TRAIN_ROWS or X_latest.empty:
        return False
    base_date = X_latest.index[-1]
    m = LightGBMForecaster(n_estimators=150).fit(X, y)
    raw = m.predict(X_latest).iloc[0]
    final = adaptive.error_correction(store.evaluated(asset, horizon, "path")).apply(raw)
    store.save_prediction(
        asset=asset, base_date=base_date.strftime("%Y-%m-%d"), horizon=horizon, model="path",
        base_price=float(close.loc[base_date]), p_up=float(raw.p_up), q10=float(final.q10),
        q50=float(final.q50), q90=float(final.q90), raw_q50=float(raw.q50),
        model_version=f"path-{ENGINE_VERSION}/{FEATURE_VERSION}",
    )
    return True


def run_cycle(store: Store, source: PriceSource | None = None, fetch: bool = True,
              assets: list[str] | None = None, model_names: list[str] | None = None,
              progress: Callable[[str], None] = lambda m: None, with_path: bool = True) -> CycleReport:
    report = CycleReport()
    if fetch:
        report.updated = update_all(store, source, progress)
        try:
            update_intraday(store, source, progress)
        except Exception:
            log.exception("Gagal memperbarui data per jam")
    progress("Mengevaluasi prediksi sebelumnya…")
    report.evaluated = evaluate_matured(store)
    for asset in assets or PREDICT_ASSETS:
        for label, h in HORIZONS.items():
            progress(f"Melatih model {ASSET_BY_KEY[asset].name} ({label})…")
            try:
                ok = predict_asset(store, asset, h, model_names)
            except Exception as exc:
                log.exception("Gagal memprediksi %s h=%s", asset, h)
                progress(f"Gagal memprediksi {asset} ({label}): {exc}")
                ok = False
            if ok:
                report.predicted += 1
            else:
                report.skipped.append(f"{asset}/{label}")
        if with_path:
            progress(f"Menyusun jalur prediksi {ASSET_BY_KEY[asset].name}…")
            for h in PATH_ANCHORS:
                try:
                    predict_path_anchor(store, asset, h)
                except Exception:
                    log.exception("Gagal jalur %s h=%s", asset, h)
    progress("Selesai.")
    return report
