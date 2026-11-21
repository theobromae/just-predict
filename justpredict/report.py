"""Ringkasan prediksi dan akurasi untuk tampilan (GUI dan CLI)."""
from __future__ import annotations

import math

import pandas as pd

from .config import ASSET_BY_KEY, HORIZONS
from .db import Store

HORIZON_LABEL = {v: k for k, v in HORIZONS.items()}

# Ambang sinyal. Sengaja konservatif: akurasi arah 52–56% sudah tergolong bagus.
BUY_P = 0.55
SELL_P = 0.45


def signal(p_up: float, q50: float) -> str:
    if p_up >= BUY_P and q50 > 0:
        return "BELI"
    if p_up <= SELL_P and q50 < 0:
        return "JUAL"
    return "TAHAN"


MIN_HIT_RATE = 0.52


def is_proven(bt: pd.DataFrame, asset: str, horizon: int) -> bool:
    """Model dianggap teruji kalau di backtest arahnya benar >= 52% dan mengalahkan baseline naif."""
    if bt.empty:
        return False
    sel = bt[(bt.asset == asset) & (bt.horizon == horizon)].set_index("model")["hit_rate"]
    if "lightgbm" not in sel:
        return False
    return sel["lightgbm"] >= MIN_HIT_RATE and sel["lightgbm"] > sel.get("naive", 0.5)


def latest_table(store: Store) -> pd.DataFrame:
    df = store.latest_predictions()
    if df.empty:
        return df
    bt = store.backtests()
    rows = []
    for r in df.itertuples():
        a = ASSET_BY_KEY[r.asset]
        proven = is_proven(bt, r.asset, r.horizon)
        rows.append({
            "aset": a.name,
            "asset_key": r.asset,
            "horizon": HORIZON_LABEL.get(r.horizon, f"{r.horizon} hari"),
            "horizon_days": r.horizon,
            "per_tanggal": r.base_date,
            "harga_terakhir": r.base_price,
            "prediksi_median": r.base_price * math.exp(r.q50),
            "rentang_bawah": r.base_price * math.exp(r.q10),
            "rentang_atas": r.base_price * math.exp(r.q90),
            "perubahan_pct": (math.exp(r.q50) - 1) * 100,
            "peluang_naik": r.p_up,
            # Sinyal hanya diberikan kalau model terbukti lebih baik dari tebakan naif.
            "sinyal": signal(r.p_up, r.q50) if proven else "TAHAN",
            "teruji": proven,
            "mata_uang": a.currency,
        })
    return pd.DataFrame(rows)


def accuracy_table(store: Store) -> pd.DataFrame:
    """Akurasi live (prediksi yang sudah jatuh tempo) dan backtest, per aset × horizon."""
    bt = store.backtests()
    ev = store.evaluated()
    rows = []
    keys = set(zip(bt["asset"], bt["horizon"])) if not bt.empty else set()
    if not ev.empty:
        keys |= set(zip(ev["asset"], ev["horizon"]))
    for asset, h in sorted(keys):
        row = {"aset": ASSET_BY_KEY[asset].name, "horizon": HORIZON_LABEL.get(h, str(h))}
        for model in ("lightgbm", "naive"):
            b = bt[(bt.asset == asset) & (bt.horizon == h) & (bt.model == model)] if not bt.empty else bt
            row[f"backtest_arah_{model}"] = float(b.hit_rate.iloc[0]) if len(b) else None
        e = ev[(ev.asset == asset) & (ev.horizon == h) & (ev.model == "ensemble")] if not ev.empty else ev
        row["live_n"] = len(e)
        row["live_arah"] = float(e.direction_hit.mean()) if len(e) else None
        row["live_galat_pct"] = float(e.abs_pct_error.mean() * 100) if len(e) else None
        row["live_dalam_rentang"] = float(e.in_range.mean()) if len(e) else None
        rows.append(row)
    return pd.DataFrame(rows)
