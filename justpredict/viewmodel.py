"""Data dan teks untuk layar grafik, dalam bahasa yang mudah dipahami orang awam.

Modul ini tidak bergantung pada Qt supaya mudah dites.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ASSET_BY_KEY, HORIZONS
from .db import Store
from .engine import PATH_POINTS
from .features import _rsi
from .report import is_proven, signal
from .technicals import TAResult, detect


@dataclass(frozen=True)
class Timeframe:
    label: str
    source: str          # "daily" atau "intraday"
    rule: str | None     # aturan resample pandas
    horizon: int         # horizon prediksi utama (hari bursa) yang ditampilkan di panel
    n_bars: int          # jumlah batang yang tampil saat dibuka
    path_key: str | None # jalur prediksi yang digambar (None = tidak digambar)


TIMEFRAMES: dict[str, Timeframe] = {
    "1jam": Timeframe("1 Jam", "intraday", None, 1, 120, None),
    "4jam": Timeframe("4 Jam", "intraday", "4h", 1, 120, None),
    "harian": Timeframe("Harian", "daily", None, 1, 120, "harian"),
    "mingguan": Timeframe("Mingguan", "daily", "W-FRI", 5, 104, "mingguan"),
    "bulanan": Timeframe("Bulanan", "daily", "ME", 21, 60, "bulanan"),
}
HORIZON_TEXT = {1: "besok (1 hari bursa)", 5: "1 minggu lagi", 21: "1 bulan lagi"}
PATH_TEXT = {"harian": "14 hari bursa ke depan (±3 minggu)", "mingguan": "8 minggu ke depan (±2 bulan)",
             "bulanan": "6 bulan ke depan"}


def local_tz():
    from datetime import datetime
    return datetime.now().astimezone().tzinfo


# ------------------------------------------------------------------ format
def money(v: float | None, currency: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if currency == "IDR":
        return "Rp" + f"{v:,.0f}".replace(",", ".")
    digits = 2 if abs(v) < 10_000 else 0
    return "$" + f"{v:,.{digits}f}"


def dec(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def pct(v: float | None, digits: int = 0) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:.{digits}f}%".replace(".", ",")


# ------------------------------------------------------------------ data grafik
def resample_ohlcv(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if rule is None or df.empty:
        return df
    out = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                 "volume": lambda s: s.sum(min_count=1)})
    out = out.dropna(subset=["close"])
    # Label batang = tanggal perdagangan terakhir di periode itu (bukan akhir kalender).
    last_dates = df["close"].resample(rule).apply(lambda s: s.index.max() if len(s) else pd.NaT)
    out.index = pd.DatetimeIndex(last_dates.reindex(out.index).values)
    return out


def _intraday_local(store: Store, asset: str, rule: str | None) -> pd.DataFrame:
    df = store.intraday(asset)
    if df.empty:
        return df
    df.index = df.index.tz_convert(local_tz()).tz_localize(None)
    if rule:
        df = df.resample(rule, origin="start_day").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "volume": lambda s: s.sum(min_count=1)}).dropna(subset=["close"])
    return df


def chart_data(store: Store, asset: str, timeframe: str) -> pd.DataFrame:
    """OHLCV sesuai tampilan, plus indikator RSI 14 dan rata-rata bergerak 20/50."""
    tf = TIMEFRAMES[timeframe]
    if tf.source == "intraday":
        df = _intraday_local(store, asset, tf.rule)
    else:
        df = resample_ohlcv(store.ohlcv(asset), tf.rule)
    if df.empty:
        return df
    df["rsi"] = _rsi(df["close"], 14)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    return df


# ------------------------------------------------------------------ ringkasan
@dataclass
class Summary:
    asset: str
    name: str
    currency: str
    horizon: int
    has_prediction: bool = False
    as_of: str = ""
    last_price: float = float("nan")
    median: float = float("nan")
    low: float = float("nan")
    high: float = float("nan")
    change_pct: float = float("nan")
    p_up: float = float("nan")
    signal: str = "TAHAN"
    proven: bool = False
    target_date: pd.Timestamp | None = None
    lines: dict[str, str] = field(default_factory=dict)
    reliability: str = "Belum ada data"
    reliability_level: int = 0   # 0 = belum ada, 1 rendah, 2 sedang, 3 baik
    past: pd.DataFrame = field(default_factory=pd.DataFrame)  # prediksi lalu yang sudah dievaluasi
    path: pd.DataFrame = field(default_factory=pd.DataFrame)  # jalur prediksi (titik-titik ke depan)
    path_key: str = "harian"
    ta: TAResult | None = None
    trend: str = "Sideways"
    trend_level: int = 0         # +2 bullish, +1 cenderung bullish, 0, -1, -2 bearish


def _lean(p_up: float) -> str:
    if p_up >= 0.60:
        return "lebih mungkin naik"
    if p_up >= 0.55:
        return "sedikit lebih mungkin naik"
    if p_up > 0.45:
        return "seimbang, sulit ditebak"
    if p_up > 0.40:
        return "sedikit lebih mungkin turun"
    return "lebih mungkin turun"


def _signal_text(sig: str, proven: bool) -> str:
    if not proven:
        return ("Belum ada saran beli/jual: di data masa lalu, model untuk aset dan jangka waktu ini "
                "belum terbukti lebih baik daripada tebakan sederhana.")
    return {
        "BELI": "Model memperkirakan harga naik, dan model ini terbukti lebih baik dari tebakan sederhana.",
        "JUAL": "Model memperkirakan harga turun, dan model ini terbukti lebih baik dari tebakan sederhana.",
        "TAHAN": "Arah harga belum cukup jelas untuk saran beli atau jual.",
    }[sig]


def _technical_text(cd: pd.DataFrame) -> list[str]:
    if cd.empty:
        return []
    last = cd.iloc[-1]
    out = []
    rsi = last.get("rsi")
    if rsi is not None and not pd.isna(rsi):
        if rsi >= 70:
            meaning = "sudah naik banyak dalam waktu singkat (jenuh beli), rawan koreksi turun"
        elif rsi <= 30:
            meaning = "sudah turun banyak dalam waktu singkat (jenuh jual), bisa memantul naik"
        else:
            meaning = "normal, tidak terlalu panas atau dingin"
        out.append(f"RSI {rsi:.0f}: {meaning}.")
    ma50 = last.get("ma50")
    if ma50 is not None and not pd.isna(ma50):
        rel = "di atas" if last["close"] > ma50 else "di bawah"
        trend = "tren naik" if rel == "di atas" else "tren turun"
        out.append(f"Harga {rel} rata-rata 50 periode: cenderung {trend}.")
    vol = cd["volume"].dropna()
    if len(vol) > 20:
        ratio = vol.iloc[-1] / vol.iloc[-21:-1].mean()
        if ratio > 1.5:
            out.append(f"Volume transaksi {dec(ratio)}× rata-rata: sedang ramai.")
        elif ratio < 0.6:
            out.append("Volume transaksi di bawah rata-rata: sedang sepi.")
    return out


def forecast_path(store: Store, asset: str, path_key: str) -> pd.DataFrame:
    """Titik-titik prediksi ke depan. Horizon yang tidak dilatih langsung diinterpolasi."""
    step, n = PATH_POINTS[path_key]
    ens, path = store.latest_predictions("ensemble"), store.latest_predictions("path")
    rows = pd.concat([ens[ens.asset == asset], path[path.asset == asset]]) if not ens.empty else ens
    if rows.empty:
        return pd.DataFrame()
    base_date = rows.base_date.max()
    rows = rows[rows.base_date == base_date].sort_values("horizon").drop_duplicates("horizon", keep="first")
    base = float(rows.base_price.iloc[0])
    hs = np.r_[0, rows.horizon.values]
    out = []
    for k in range(1, n + 1):
        h = step * k
        if h > hs.max():
            break
        q = {c: float(np.interp(h, hs, np.r_[0.0, rows[c].values])) for c in ("q10", "q50", "q90")}
        p_up = float(np.interp(h, hs, np.r_[0.5, rows.p_up.values]))
        lo, mid, hi = sorted([q["q10"], q["q50"], q["q90"]])
        out.append({"h": h, "date": pd.Timestamp(base_date) + pd.offsets.BDay(h),
                    "mid": base * math.exp(mid), "lo": base * math.exp(lo),
                    "hi": base * math.exp(hi), "p_up": p_up})
    df = pd.DataFrame(out)
    df.attrs.update(base_date=pd.Timestamp(base_date), base_price=base)
    return df


def _trend(path: pd.DataFrame, ta: TAResult | None) -> tuple[str, int, str]:
    model_dir = 0
    if not path.empty:
        end = path.iloc[-1]
        chg = end["mid"] / path.attrs["base_price"] - 1
        if end.p_up >= 0.55 and chg > 0:
            model_dir = 1
        elif end.p_up <= 0.45 and chg < 0:
            model_dir = -1
    score = ta.score if ta else 0
    ta_dir = 1 if score >= 2 else -1 if score <= -2 else 0
    total = model_dir + ta_dir
    label = {2: "Bullish", 1: "Cenderung bullish", 0: "Sideways", -1: "Cenderung bearish", -2: "Bearish"}[total]
    if model_dir and ta_dir and model_dir != ta_dir:
        label = "Belum jelas"
    word = {1: "naik", -1: "turun", 0: "datar"}
    why = f"Model: {word[model_dir]}. Analisis teknikal: {word[ta_dir]} (skor {score:+d})."
    return label, total if label != "Belum jelas" else 0, why


def summarize(store: Store, asset: str, timeframe: str, cd: pd.DataFrame | None = None) -> Summary:
    a = ASSET_BY_KEY[asset]
    tf = TIMEFRAMES[timeframe]
    horizon = tf.horizon
    s = Summary(asset=asset, name=a.name, currency=a.currency, horizon=horizon,
                path_key=tf.path_key or "harian")
    cd = chart_data(store, asset, timeframe) if cd is None else cd

    preds = store.latest_predictions()
    row = preds[(preds.asset == asset) & (preds.horizon == horizon)] if not preds.empty else preds
    bt = store.backtests()
    ev = store.evaluated(asset, horizon, "ensemble")
    s.past = ev

    if len(row):
        r = row.iloc[0]
        s.has_prediction = True
        s.as_of = r.base_date
        s.last_price = r.base_price
        s.median = r.base_price * math.exp(r.q50)
        s.low = r.base_price * math.exp(r.q10)
        s.high = r.base_price * math.exp(r.q90)
        s.change_pct = math.exp(r.q50) - 1
        s.p_up = float(r.p_up)
        s.proven = is_proven(bt, asset, horizon)
        s.signal = signal(s.p_up, r.q50) if s.proven else "TAHAN"
        s.target_date = pd.Timestamp(r.base_date) + pd.offsets.BDay(horizon)
        arrow = "▲ naik" if s.change_pct > 0 else "▼ turun" if s.change_pct < 0 else "tetap"
        s.lines["headline"] = f"Perkiraan harga {HORIZON_TEXT.get(horizon, f'{horizon} hari lagi')}"
        s.lines["change"] = (f"{arrow} {dec(abs(s.change_pct) * 100)}% dari harga terakhir "
                             f"{money(s.last_price, a.currency)}")
        s.lines["range"] = (f"Kemungkinan besar (8 dari 10 kali) harganya ada di antara "
                            f"{money(s.low, a.currency)} dan {money(s.high, a.currency)}.")
        s.lines["p_up"] = f"Peluang naik {pct(s.p_up)}: {_lean(s.p_up)}."
        s.lines["signal"] = _signal_text(s.signal, s.proven)
        s.lines["as_of"] = f"Berdasarkan harga penutupan {pd.Timestamp(r.base_date):%d %b %Y}."
    else:
        s.lines["headline"] = "Belum ada prediksi"
        s.lines["signal"] = "Data belum cukup untuk membuat prediksi aset ini."

    # ---- keandalan
    sel = bt[(bt.asset == asset) & (bt.horizon == horizon)].set_index("model")["hit_rate"] if not bt.empty else pd.Series(dtype=float)
    acc = []
    if "lightgbm" in sel:
        acc.append(f"Diuji pada data masa lalu: arah naik/turun benar {pct(sel['lightgbm'])}, "
                   f"tebakan sederhana {pct(sel.get('naive', float('nan')))}.")
    if len(ev):
        hits = int(ev.direction_hit.sum())
        acc.append(f"Prediksi yang sudah terbukti: {len(ev)} kali, arahnya benar {hits} kali "
                   f"({pct(hits / len(ev))}). Rata-rata meleset {dec(ev.abs_pct_error.mean() * 100)}%.")
    else:
        acc.append("Belum ada prediksi yang jatuh tempo. Akurasi nyata akan muncul setelah aplikasi "
                   "dipakai beberapa hari.")
    acc.append("Sebagai patokan, arah benar 52–56% sudah tergolong bagus untuk harga pasar.")
    s.lines["accuracy"] = "\n".join(acc)

    if "lightgbm" in sel:
        edge = sel["lightgbm"] - sel.get("naive", 0.5)
        live = ev.direction_hit.mean() if len(ev) >= 20 else None
        score = sel["lightgbm"] if live is None else (sel["lightgbm"] + live) / 2
        if live is not None and live < 0.5:
            s.reliability, s.reliability_level = "Rendah", 1  # prediksi nyata belakangan sering salah arah
        elif s.proven and score >= 0.55 and edge > 0.02:
            s.reliability, s.reliability_level = "Baik", 3
        elif s.proven:
            s.reliability, s.reliability_level = "Sedang", 2
        else:
            s.reliability, s.reliability_level = "Rendah", 1

    s.ta = detect(cd, money=lambda v: money(v, a.currency)) if cd is not None and not cd.empty else None
    s.path = forecast_path(store, asset, s.path_key)
    s.trend, s.trend_level, s.lines["trend_why"] = _trend(s.path, s.ta)
    if not s.path.empty:
        end = s.path.iloc[-1]
        chg = end["mid"] / s.path.attrs["base_price"] - 1
        arrow = "▲" if chg > 0 else "▼"
        s.lines["path"] = (f"Perkiraan {PATH_TEXT[s.path_key]}: {money(end['mid'], a.currency)} "
                           f"({arrow} {dec(abs(chg) * 100)}%).")
    s.lines["technical"] = "\n".join(_technical_text(cd))
    return s


def asset_list(store: Store, timeframe: str) -> list[dict]:
    """Untuk daftar aset di sisi kiri: harga terakhir, perubahan prediksi, sinyal."""
    from .config import PREDICT_ASSETS

    horizon = TIMEFRAMES[timeframe].horizon
    preds = store.latest_predictions()
    bt = store.backtests()
    out = []
    for k in PREDICT_ASSETS:
        a = ASSET_BY_KEY[k]
        row = preds[(preds.asset == k) & (preds.horizon == horizon)] if not preds.empty else preds
        item = {"asset": k, "name": a.name, "price": None, "change": None, "signal": None, "currency": a.currency}
        if len(row):
            r = row.iloc[0]
            proven = is_proven(bt, k, horizon)
            item.update(price=r.base_price, change=math.exp(r.q50) - 1,
                        signal=signal(r.p_up, r.q50) if proven else "TAHAN")
        else:
            last = store.prices(k)
            if len(last):
                item["price"] = float(last.iloc[-1])
        out.append(item)
    return out
