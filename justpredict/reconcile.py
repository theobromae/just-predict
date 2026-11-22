"""Gabungan prediksi multi-rentang (rekonsiliasi).

Setiap rentang (1, 3, 5, … 126 hari bursa) punya model sendiri, sehingga prediksinya bisa saling tidak
nyambung: misalnya 5 hari ke depan naik 2% tapi 10 hari ke depan turun 1%. Di sini semua rentang
diselaraskan ke satu "arah dasar" bersama:

- median return:  q(h) ≈ μ·h   (μ = laju harian bersama, regresi berbobot dari semua rentang)
- peluang naik:   logit p(h) ≈ s·√h   (kekuatan sinyal bersama; skala √h seperti random walk)

Bobot tiap rentang = 1 / galat² di uji data lama, jadi rentang yang rekam jejaknya buruk kurang
berpengaruh. Prediksi akhir tiap rentang = λ·prediksi langsung + (1−λ)·arah dasar.

λ dipilih di paruh pertama data uji, lalu hasilnya dinilai di paruh kedua (data yang tidak dipakai
memilih λ). Rekonsiliasi hanya dipakai untuk rentang yang di paruh kedua galatnya lebih kecil dan
ketepatan arahnya tidak lebih buruk daripada prediksi langsung.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .db import Store
from .features import build_dataset
from .models.gbm import LightGBMForecaster

log = logging.getLogger(__name__)

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
HOLDOUT_FRACTION = 0.3
MIN_TRAIN = 150
MIN_COMMON = 120


@dataclass
class Params:
    horizon: int
    lam_q: float
    lam_p: float
    used: bool
    mae_direct: float
    mae_recon: float
    hit_direct: float
    hit_recon: float
    n: int
    weight: float


def _logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def holdout_predictions(close: pd.Series, ohlcv: pd.DataFrame | None, horizons: list[int]) -> dict[int, pd.DataFrame]:
    """Prediksi out-of-sample (p_up, q50) dan return aktual per rentang, untuk menilai rekonsiliasi."""
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        X, y, _ = build_dataset(close, h, None, ohlcv)
        n = len(X)
        split = int(n * (1 - HOLDOUT_FRACTION))
        train_end = split - h  # jeda agar target latih tidak tumpang tindih dengan data uji
        if train_end < MIN_TRAIN or n - split < MIN_COMMON:
            continue
        m = LightGBMForecaster(n_estimators=150).fit(X.iloc[:train_end], y.iloc[:train_end])
        p = m.predict(X.iloc[split:])
        out[h] = pd.DataFrame({"p_up": p["p_up"], "q50": p["q50"], "y": y.iloc[split:]})
    return out


def common_signal(Q: np.ndarray, L: np.ndarray, hs: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arah dasar bersama per baris: μ (laju return harian) dan s (kekuatan sinyal per √hari)."""
    mu = (Q * hs * w).sum(axis=1) / (hs ** 2 * w).sum()
    s = (L / np.sqrt(hs) * w).sum(axis=1) / w.sum()
    return mu, s


def fit(hold: dict[int, pd.DataFrame]) -> dict[int, Params]:
    if len(hold) < 3:
        return {}
    common = None
    for df in hold.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < MIN_COMMON:
        return {}
    hs = np.array(sorted(hold), dtype=float)
    Q = np.column_stack([hold[int(h)].loc[common, "q50"].to_numpy() for h in hs])
    P = np.column_stack([hold[int(h)].loc[common, "p_up"].to_numpy() for h in hs])
    Y = np.column_stack([hold[int(h)].loc[common, "y"].to_numpy() for h in hs])
    L = _logit(P)
    half = len(common) // 2
    cal, ev = slice(0, half), slice(half, None)
    mae_cal = np.abs(Y[cal] - Q[cal]).mean(axis=0)
    w = 1 / np.maximum(mae_cal, 1e-6) ** 2
    mu, s = common_signal(Q, L, hs, w)
    out: dict[int, Params] = {}
    for j, h in enumerate(hs):
        iq, il = mu * h, s * math.sqrt(h)

        def blend_q(lam, sl):
            return lam * Q[sl, j] + (1 - lam) * iq[sl]

        def blend_l(lam, sl):
            return lam * L[sl, j] + (1 - lam) * il[sl]

        def logloss(lam, sl):
            pr = np.clip(_sigmoid(blend_l(lam, sl)), 0.01, 0.99)
            up = Y[sl, j] > 0
            return -np.mean(np.where(up, np.log(pr), np.log(1 - pr)))

        lam_q = min(LAMBDAS, key=lambda lam: np.abs(Y[cal, j] - blend_q(lam, cal)).mean())
        lam_p = min(LAMBDAS, key=lambda lam: logloss(lam, cal))
        up_ev = Y[ev, j] > 0
        mae_d = float(np.abs(Y[ev, j] - Q[ev, j]).mean())
        mae_r = float(np.abs(Y[ev, j] - blend_q(lam_q, ev)).mean())
        hit_d = float(((P[ev, j] > 0.5) == up_ev).mean())
        hit_r = float(((blend_l(lam_p, ev) > 0) == up_ev).mean())
        used = (lam_q < 1 or lam_p < 1) and mae_r < mae_d and hit_r >= hit_d - 0.005
        out[int(h)] = Params(int(h), lam_q, lam_p, bool(used), mae_d, mae_r, hit_d, hit_r,
                             int(len(common) - half), float(w[j]))
    return out


def latest_rows(store: Store, asset: str) -> pd.DataFrame:
    """Prediksi harian terbaru (ensemble + titik jalur) pada tanggal dasar yang sama."""
    parts = [store.latest_predictions(m) for m in ("ensemble", "path")]
    parts = [df[df.asset == asset] for df in parts if not df.empty]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    num = ["horizon", "base_price", "p_up", "q10", "q50", "q90"]
    rows[num] = rows[num].apply(pd.to_numeric, errors="coerce")
    rows = rows.dropna(subset=num)
    if rows.empty:
        return rows
    rows = rows[rows.base_date == rows.base_date.max()]
    return rows.sort_values("horizon").drop_duplicates("horizon", keep="first")  # ensemble didahulukan


def apply_live(store: Store, asset: str, params: dict[int, Params], version: str = "") -> int:
    """Simpan prediksi hasil rekonsiliasi (model 'recon') untuk semua rentang harian."""
    rows = latest_rows(store, asset)
    rows = rows[rows.horizon.astype(int).isin(list(params))]
    if len(rows) < 3:
        return 0
    hs = rows.horizon.to_numpy(dtype=float)
    w = np.array([params[int(h)].weight for h in hs])
    Q = rows.q50.to_numpy(dtype=float)[None, :]
    L = _logit(rows.p_up.to_numpy(dtype=float))[None, :]
    mu, s = common_signal(Q, L, hs, w)
    n = 0
    for r in rows.itertuples():
        h = int(r.horizon)
        pr = params[h]
        q50, p_up = float(r.q50), float(r.p_up)
        if pr.used:
            q50 = pr.lam_q * r.q50 + (1 - pr.lam_q) * float(mu[0]) * h
            p_up = float(_sigmoid(pr.lam_p * float(_logit(r.p_up)) + (1 - pr.lam_p) * float(s[0]) * math.sqrt(h)))
        shift = q50 - float(r.q50)
        store.save_prediction(
            asset=asset, base_date=r.base_date, horizon=h, model="recon", base_price=float(r.base_price),
            p_up=float(np.clip(p_up, 0.01, 0.99)), q10=float(r.q10) + shift, q50=q50, q90=float(r.q90) + shift,
            raw_q50=float(r.q50),
            model_version=f"recon-{version} from={r.model} used={int(pr.used)} lq={pr.lam_q:.2f} lp={pr.lam_p:.2f}",
        )
        n += 1
    return n


def reconcile_asset(store: Store, asset: str, horizons: list[int], version: str = "") -> dict[int, Params]:
    close = store.prices(asset)
    if close.empty:
        return {}
    params = fit(holdout_predictions(close, store.ohlcv(asset), horizons))
    for p in params.values():
        store.save_reconcile(asset, p)
    if params:
        apply_live(store, asset, params, version)
    return params
