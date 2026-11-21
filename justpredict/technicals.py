"""Analisis teknikal: pola candle, support/resistance, tren RSI, persilangan rata-rata, tembus level.

Dipakai dua cara:
- `ta_features(df)`: fitur numerik untuk model ML (hanya memakai data sampai baris itu, tanpa bocoran).
- `detect(df)`: daftar sinyal yang bisa dibaca orang awam + level support/resistance untuk digambar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up / down
    out = 100 - 100 / (1 + rs)
    return out.where(~((up == 0) & (down == 0)), 50.0).where(delta.notna() | out.notna())


# ---------------------------------------------------------------- pola candle
def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom boolean per pola. Butuh open/high/low/close."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    prev_o, prev_c = o.shift(1), c.shift(1)
    # Konteks tren pendek: hammer bermakna setelah turun, shooting star setelah naik.
    down_before = c.shift(1) < c.shift(5)
    up_before = c.shift(1) > c.shift(5)
    out = pd.DataFrame(index=df.index)
    out["doji"] = (body / rng) < 0.1
    out["hammer"] = (lower >= 2 * body) & (upper <= body) & (body / rng > 0.1) & down_before
    out["shooting_star"] = (upper >= 2 * body) & (lower <= body) & (body / rng > 0.1) & up_before
    out["bull_engulfing"] = (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
    out["bear_engulfing"] = (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)
    return out.fillna(False).astype(bool)


# ---------------------------------------------------------------- fitur model
def ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur analisis teknikal untuk model. `df` = OHLCV harian (volume boleh kosong)."""
    c = df["close"].astype(float)
    h, l = df["high"].astype(float), df["low"].astype(float)
    f = pd.DataFrame(index=df.index)

    # Support/resistance kasar: jarak ke level tertinggi/terendah 20 dan 60 bar sebelumnya.
    for n in (20, 60):
        res = h.shift(1).rolling(n).max()
        sup = l.shift(1).rolling(n).min()
        f[f"dist_res_{n}"] = c / res - 1
        f[f"dist_sup_{n}"] = c / sup - 1
        f[f"breakout_{n}"] = (c > res).astype(float)
        f[f"breakdown_{n}"] = (c < sup).astype(float)

    r = rsi(c)
    f["rsi_slope_5"] = r.diff(5) / 100
    # Divergensi: harga di puncak 20 bar tapi RSI tidak (bearish), dan sebaliknya.
    price_hi = c >= c.rolling(20).max()
    price_lo = c <= c.rolling(20).min()
    f["bear_div"] = (price_hi & (r < r.rolling(20).max() - 5)).astype(float)
    f["bull_div"] = (price_lo & (r > r.rolling(20).min() + 5)).astype(float)

    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    above = (ma20 > ma50).astype(float)
    f["ma20_above_ma50"] = above
    cross = above.diff().fillna(0)
    f["golden_cross_10"] = (cross.rolling(10).max() > 0).astype(float)
    f["death_cross_10"] = (cross.rolling(10).min() < 0).astype(float)

    std20 = c.rolling(20).std()
    f["bb_pctb"] = (c - (ma20 - 2 * std20)) / (4 * std20)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c

    # Kemiringan tren 20 bar (regresi log harga), dinormalisasi volatilitas.
    logc = np.log(c)
    x = np.arange(20) - 9.5
    slope = logc.rolling(20).apply(lambda w: np.dot(x, w - w.mean()) / np.dot(x, x), raw=True)
    f["trend_slope_20"] = slope / logc.diff().rolling(20).std()

    vol = df.get("volume")
    if vol is not None and vol.notna().sum() > 30:
        v = vol.astype(float)
        f["vol_ratio_20"] = (v / v.rolling(20).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        f["vol_ratio_20"] = 1.0

    pats = candle_patterns(df).astype(float)
    for col in pats:
        # Pola dalam 3 bar terakhir (pengaruhnya biasanya beberapa hari).
        f[f"pat_{col}"] = pats[col].rolling(3, min_periods=1).max()
    return f


# ---------------------------------------------------------------- deteksi untuk tampilan
@dataclass
class TASignal:
    text: str
    direction: int  # +1 bullish, -1 bearish, 0 netral


@dataclass
class TAResult:
    supports: list[float]
    resistances: list[float]
    signals: list[TASignal]
    pattern_marks: list[tuple[pd.Timestamp, str, int]]  # (tanggal, nama pola, arah) untuk ditandai di grafik

    @property
    def score(self) -> int:
        return sum(s.direction for s in self.signals)


PATTERN_INFO = {
    "hammer": ("Hammer", 1, "pola pembalikan naik setelah turun"),
    "bull_engulfing": ("Bullish engulfing", 1, "pembeli mengambil alih"),
    "shooting_star": ("Shooting star", -1, "pola pembalikan turun setelah naik"),
    "bear_engulfing": ("Bearish engulfing", -1, "penjual mengambil alih"),
    "doji": ("Doji", 0, "pasar ragu-ragu"),
}


def _pivots(df: pd.DataFrame, k: int = 5) -> tuple[pd.Series, pd.Series]:
    h, l = df["high"], df["low"]
    win = 2 * k + 1
    ph = h[(h == h.rolling(win, center=True).max())]
    pl = l[(l == l.rolling(win, center=True).min())]
    return ph.dropna(), pl.dropna()


def _cluster(levels: list[float], tol: float) -> list[tuple[float, int]]:
    """Gabungkan level yang berdekatan (selisih < tol). Hasil: (level rata-rata, jumlah sentuhan)."""
    out: list[list[float]] = []
    for v in sorted(levels):
        if out and abs(v / np.mean(out[-1]) - 1) < tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [(float(np.mean(g)), len(g)) for g in out]


def support_resistance(df: pd.DataFrame, lookback: int = 150, n_each: int = 2) -> tuple[list[float], list[float]]:
    recent = df.iloc[-lookback:]
    if len(recent) < 15:
        return [], []
    ph, pl = _pivots(recent)
    price = float(recent["close"].iloc[-1])
    vol = float(recent["close"].pct_change().std() or 0.01)
    levels = _cluster(list(ph.values) + list(pl.values), tol=max(0.005, 1.5 * vol))
    # Level kuat = sering disentuh; pilih yang terdekat dengan harga.
    sup = sorted([lv for lv, n in levels if lv < price], key=lambda v: price - v)[:n_each]
    res = sorted([lv for lv, n in levels if lv > price], key=lambda v: v - price)[:n_each]
    return sup, res


def detect(df: pd.DataFrame, money=lambda v: f"{v:,.2f}") -> TAResult:
    """Sinyal teknikal pada bar terakhir `df` (harian/mingguan/bulanan/per jam)."""
    if len(df) < 30:
        return TAResult([], [], [], [])
    c = df["close"]
    sup, res = support_resistance(df)
    signals: list[TASignal] = []
    last = float(c.iloc[-1])

    # Tembus level
    prev_hi = df["high"].shift(1).rolling(20).max().iloc[-1]
    prev_lo = df["low"].shift(1).rolling(20).min().iloc[-1]
    if last > prev_hi:
        signals.append(TASignal(f"Harga menembus resistance {money(prev_hi)} (breakout): tanda kuat naik.", 1))
    elif last < prev_lo:
        signals.append(TASignal(f"Harga jebol support {money(prev_lo)} (breakdown): tanda kuat turun.", -1))
    else:
        if sup and (last / sup[0] - 1) < 0.02:
            signals.append(TASignal(f"Harga dekat support {money(sup[0])}: sering jadi titik pantul naik.", 1))
        if res and (res[0] / last - 1) < 0.02:
            signals.append(TASignal(f"Harga dekat resistance {money(res[0])}: sering jadi titik tahanan.", -1))

    # RSI
    r = rsi(c)
    rv, rslope = r.iloc[-1], r.iloc[-1] - r.iloc[-6]
    if rv >= 70:
        signals.append(TASignal(f"RSI {rv:.0f}: jenuh beli, rawan turun.", -1))
    elif rv <= 30:
        signals.append(TASignal(f"RSI {rv:.0f}: jenuh jual, berpeluang memantul.", 1))
    if abs(rslope) >= 8:
        signals.append(TASignal(f"RSI sedang {'naik' if rslope > 0 else 'turun'} tajam: momentum "
                                f"{'menguat' if rslope > 0 else 'melemah'}.", 1 if rslope > 0 else -1))
    window = slice(-20, None)
    if c.iloc[-1] >= c.iloc[window].max() and r.iloc[-1] < r.iloc[window].max() - 5:
        signals.append(TASignal("Divergensi bearish: harga cetak puncak baru tapi RSI tidak ikut.", -1))
    if c.iloc[-1] <= c.iloc[window].min() and r.iloc[-1] > r.iloc[window].min() + 5:
        signals.append(TASignal("Divergensi bullish: harga cetak dasar baru tapi RSI tidak ikut.", 1))

    # Rata-rata bergerak
    if len(c) >= 55:
        ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
        above = ma20 > ma50
        flips = above.astype(int).diff().iloc[-5:]
        if (flips > 0).any():
            signals.append(TASignal("Golden cross: rata-rata 20 baru memotong ke atas rata-rata 50.", 1))
        elif (flips < 0).any():
            signals.append(TASignal("Death cross: rata-rata 20 baru memotong ke bawah rata-rata 50.", -1))
        elif last > ma50.iloc[-1] and above.iloc[-1]:
            signals.append(TASignal("Harga di atas rata-rata 20 dan 50: tren naik.", 1))
        elif last < ma50.iloc[-1] and not above.iloc[-1]:
            signals.append(TASignal("Harga di bawah rata-rata 20 dan 50: tren turun.", -1))

    # Pola candle 3 bar terakhir
    pats = candle_patterns(df)
    marks: list[tuple[pd.Timestamp, str, int]] = []
    for col, (name, direction, meaning) in PATTERN_INFO.items():
        hits = pats[col].iloc[-30:]
        for d in hits[hits].index:
            marks.append((d, name, direction))
        if pats[col].iloc[-3:].any():
            signals.append(TASignal(f"Pola candle {name}: {meaning}.", direction))
    return TAResult(sup, res, signals, marks)
