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
from .data.fundamentals import fundamental_features, update_fundamentals
from .data.news import update_news
from .data.prices import PriceSource, intraday_local, update_all, update_intraday
from .db import Store
from .features import FEATURE_VERSION, build_dataset
from .models import make_models
from .models.gbm import LightGBMForecaster
from .reconcile import reconcile_asset

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 300
BACKTEST_FRACTION = 0.2
ENGINE_VERSION = "0.3.0"
# Fundamental tidak mengubah angka harga; ia hanya menggeser peluang naik, paling banyak sebesar ini.
MAX_FUND_SHIFT = 0.10

# Jalur prediksi: (jarak antar titik dalam hari bursa, jumlah titik) per tampilan.
PATH_POINTS: dict[str, tuple[int, int]] = {"harian": (1, 14), "mingguan": (5, 8), "bulanan": (21, 6),
                                           "1jam": (1, 8), "4jam": (1, 10)}
# Horizon yang benar-benar dilatih; titik di antaranya diinterpolasi.
PATH_ANCHORS: list[int] = [3, 10, 14, 20, 30, 40, 42, 63, 126]

# Prediksi per jam: aturan resample, horizon yang dilatih (dalam jumlah batang), panjang batang (jam).
INTRADAY: dict[str, tuple[str | None, list[int], int]] = {
    "1jam": (None, [1, 2, 4, 8], 1),
    "4jam": ("4h", [1, 2, 5, 10], 4),
}
MIN_INTRADAY_ROWS = 400


def intraday_model(key: str) -> str:
    return f"intra_{key}"


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
        if row.model.startswith("intra_"):
            key = row.model.removeprefix("intra_")
            ck = f"{row.asset}|{key}"
            if ck not in price_cache:
                df = intraday_local(store, row.asset, INTRADAY[key][0]) if key in INTRADAY else pd.DataFrame()
                price_cache[ck] = df["close"] if not df.empty else pd.Series(dtype=float)
            prices = price_cache[ck]
        else:
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
            target_date=target_date.strftime("%Y-%m-%d %H:%M" if row.model.startswith("intra_") else "%Y-%m-%d"),
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
    pu = p["p_up"].clip(0.01, 0.99).to_numpy()
    up = (yt > 0).to_numpy()
    return {
        "logloss": float(-np.mean(np.where(up, np.log(pu), np.log(1 - pu)))),
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

    # Angka harga (rentang dan median) selalu dari model tanpa fundamental. Fundamental hanya boleh
    # menggeser peluang naik, dan hanya kalau di uji data lama terbukti membuat peluang lebih tepat
    # (log loss lebih kecil).
    base_cols = [c for c in X.columns if not c.startswith("x_")]
    X_full, X_full_latest = X, X_latest
    X, X_latest = X[base_cols], X_latest[base_cols]
    use_fund = False
    if len(base_cols) < len(X_full.columns):
        with_x = _backtest(LightGBMForecaster, X_full, y, horizon)
        without = _backtest(LightGBMForecaster, X, y, horizon)
        if with_x and without:
            store.save_backtest(asset, horizon, "lightgbm_fund", **with_x)
            store.save_backtest(asset, horizon, "lightgbm_base", **without)
            use_fund = with_x["logloss"] < without["logloss"]

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

    fund_shift = 0.0
    if use_fund:
        p_fund = float(LightGBMForecaster().fit(X_full, y).predict(X_full_latest).iloc[0].p_up)
        p_base = float(member_preds["lightgbm"].p_up) if "lightgbm" in member_preds else p_fund
        fund_shift = float(np.clip(p_fund - p_base, -MAX_FUND_SHIFT, MAX_FUND_SHIFT))

    store.save_prediction(
        asset=asset, base_date=base_str, horizon=horizon, model="ensemble", base_price=base_price,
        p_up=float(np.clip(final.p_up + fund_shift, 0.01, 0.99)), q10=float(final.q10), q50=float(final.q50),
        q90=float(final.q90), raw_q50=float(combined.q50),
        model_version=(f"ensemble-{ENGINE_VERSION} w=" + ",".join(f"{k}:{v:.2f}" for k, v in weights.items())
                       + f" bias={correction.bias:+.4f} scale={correction.width_scale:.2f}"
                       + f" fund_shift={fund_shift:+.3f}"),
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


def predict_intraday(store: Store, asset: str, key: str) -> bool:
    """Prediksi per jam / 4 jam: model LightGBM per horizon (dalam batang) + koreksi galat sendiri.

    Backtest horizon 1 batang dibandingkan dengan tebakan naif untuk menilai keandalan.
    """
    rule, anchors, _ = INTRADAY[key]
    df = intraday_local(store, asset, rule)
    if len(df) < MIN_INTRADAY_ROWS:
        return False
    close = df["close"]
    name = intraday_model(key)
    from .models import NaiveForecaster
    for h in anchors:
        X, y, X_latest = build_dataset(close, h, None, df)
        if len(X) < MIN_INTRADAY_ROWS // 2 or X_latest.empty:
            return False
        factory = lambda: LightGBMForecaster(n_estimators=150, recent_weight_halflife=2000)
        if h == 1:
            for label, fac in (("lgbm", factory), ("naive", NaiveForecaster)):
                bt = _backtest(fac, X, y, h)
                if bt:
                    store.save_backtest(asset, h, f"{name}_{label}", **bt)
        raw = factory().fit(X, y).predict(X_latest).iloc[0]
        final = adaptive.error_correction(store.evaluated(asset, h, name)).apply(raw)
        base_ts = X_latest.index[-1]
        store.save_prediction(
            asset=asset, base_date=base_ts.strftime("%Y-%m-%d %H:%M"), horizon=h, model=name,
            base_price=float(close.loc[base_ts]), p_up=float(raw.p_up), q10=float(final.q10),
            q50=float(final.q50), q90=float(final.q90), raw_q50=float(raw.q50),
            model_version=f"{name}-{ENGINE_VERSION}/{FEATURE_VERSION}",
        )
    return True


def run_cycle(store: Store, source: PriceSource | None = None, fetch: bool = True,
              assets: list[str] | None = None, model_names: list[str] | None = None,
              progress: Callable[[str], None] = lambda m: None, with_path: bool = True,
              with_intraday: bool = True, with_fundamentals: bool = True) -> CycleReport:
    report = CycleReport()
    if fetch:
        report.updated = update_all(store, source, progress)
        try:
            update_intraday(store, source, progress)
        except Exception:
            log.exception("Gagal memperbarui data per jam")
        # Sumber harga palsu (tes) tanpa method fundamental/berita dilewati; None = sumber internet.
        if with_fundamentals and (source is None or hasattr(source, "statements")):
            try:
                update_fundamentals(store, source, progress)
            except Exception:
                log.exception("Gagal memperbarui data fundamental")
        if source is None or hasattr(source, "fetch_news"):
            try:
                update_news(store, source, assets or None, progress)
            except Exception:
                log.exception("Gagal memperbarui berita")
    progress("Mengevaluasi prediksi sebelumnya…")
    report.evaluated = evaluate_matured(store)
    for asset in assets or PREDICT_ASSETS:
        for label, h in HORIZONS.items():
            progress(f"Melatih model {ASSET_BY_KEY[asset].name} ({label})…")
            try:
                extra = fundamental_features(store, asset) if with_fundamentals else None
                ok = predict_asset(store, asset, h, model_names, extra)
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
            progress(f"Menyelaraskan prediksi antar-rentang {ASSET_BY_KEY[asset].name}…")
            try:
                reconcile_asset(store, asset, sorted(set(HORIZONS.values()) | set(PATH_ANCHORS)), ENGINE_VERSION)
            except Exception:
                log.exception("Gagal rekonsiliasi %s", asset)
        if with_intraday:
            for key in INTRADAY:
                progress(f"Melatih model per jam {ASSET_BY_KEY[asset].name} ({key})…")
                try:
                    predict_intraday(store, asset, key)
                except Exception:
                    log.exception("Gagal prediksi per jam %s %s", asset, key)
    progress("Selesai.")
    return report
