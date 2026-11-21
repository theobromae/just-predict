"""Data fundamental (laporan keuangan saham) dan makro, fitur untuk model, dan skor untuk Arah Tren.

Sumber gratis tanpa API key:
- Laporan keuangan saham: Yahoo Finance (yfinance).
- Indeks dolar AS (DXY) dan imbal hasil obligasi AS 10 tahun: Yahoo Finance.
- Suku bunga The Fed, suku bunga Indonesia, stok minyak mentah AS (data EIA): FRED (unduhan CSV publik).
- Bitcoin: indeks Fear & Greed (alternative.me) dan hash rate (blockchain.com).

Semua angka disimpan dengan tanggal saat angka itu SUDAH TERSEDIA bagi publik. Laporan kuartal
dianggap terbit 45 hari setelah akhir kuartal, laporan tahunan 90 hari (batas waktu di BEI), dan data
makro bulanan 45 hari. Dengan begitu model tidak pernah "mengintip" angka sebelum terbit.
"""
from __future__ import annotations

import io
import json
import logging
import math
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..config import ASSET_BY_KEY, ASSETS
from ..db import Store

log = logging.getLogger(__name__)

QUARTER_LAG_DAYS = 45
ANNUAL_LAG_DAYS = 90
REFRESH_HOURS = 20  # data fundamental/makro paling sering diambil sekali sehari

MACRO_YAHOO = {"DXY": "DX-Y.NYB", "US10Y": "^TNX"}
# seri FRED: (id, jeda tersedia dalam hari)
MACRO_FRED = {"FEDFUNDS": ("DFF", 1), "BIRATE": ("INTDSRIDM193N", 45), "CRUDE_STOCKS": ("WCESTUS1", 5)}
# Umur maksimum data (hari) agar masih dipakai di skor; lebih tua dari ini dianggap usang.
STALE_DAYS = {"DXY": 10, "US10Y": 10, "FEDFUNDS": 10, "BIRATE": 120, "CRUDE_STOCKS": 21,
              "FNG": 5, "HASHRATE": 10, "net_income_ttm": 220}

STOCK_SERIES = ("net_income_ttm", "revenue_ttm", "equity", "debt", "shares")


def is_stock(asset: str) -> bool:
    a = ASSET_BY_KEY.get(asset)
    return bool(a and a.ticker and a.ticker.endswith(".JK"))


# ------------------------------------------------------------------ sumber data
class FundamentalSource(Protocol):
    def statements(self, ticker: str) -> dict: ...
    def daily(self, ticker: str) -> pd.Series: ...
    def fred(self, series_id: str) -> pd.Series: ...
    def fear_greed(self) -> pd.Series: ...
    def hash_rate(self) -> pd.Series: ...


def _http_get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (JustPredict desktop app)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class WebFundamentalSource:
    """Sumber gratis di internet. Setiap method boleh gagal; pemanggil menangani error."""

    def statements(self, ticker: str) -> dict:
        import yfinance as yf
        t = yf.Ticker(ticker)
        currency = None
        try:
            currency = (t.get_info() or {}).get("financialCurrency")
        except Exception:  # info kadang gagal, laporan tetap bisa diambil
            log.info("Info %s tidak tersedia", ticker)
        return {
            "currency": currency,
            "quarterly_income": t.quarterly_income_stmt,
            "annual_income": t.income_stmt,
            "quarterly_balance": t.quarterly_balance_sheet,
            "annual_balance": t.balance_sheet,
        }

    def daily(self, ticker: str) -> pd.Series:
        import yfinance as yf
        df = yf.download(ticker, start="2012-01-01", progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return close.dropna().astype(float)

    def fred(self, series_id: str) -> pd.Series:
        raw = _http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
        df = pd.read_csv(io.BytesIO(raw))
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                      index=pd.to_datetime(df.iloc[:, 0]))
        return s.dropna()

    def fear_greed(self) -> pd.Series:
        data = json.loads(_http_get("https://api.alternative.me/fng/?limit=0&format=json"))["data"]
        s = pd.Series({pd.Timestamp(int(d["timestamp"]), unit="s").normalize(): float(d["value"]) for d in data})
        return s.sort_index()

    def hash_rate(self) -> pd.Series:
        data = json.loads(_http_get(
            "https://api.blockchain.info/charts/hash-rate?timespan=6years&format=json&sampled=true"))["values"]
        s = pd.Series({pd.Timestamp(int(d["x"]), unit="s").normalize(): float(d["y"]) for d in data})
        return s.sort_index()


# ------------------------------------------------------------------ olah laporan keuangan
def _row(df: pd.DataFrame | None, names: tuple[str, ...]) -> pd.Series:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            idx = pd.DatetimeIndex(pd.to_datetime(s.index))
            s.index = idx.tz_localize(None) if idx.tz is not None else idx
            return s.dropna().sort_index()
    return pd.Series(dtype=float)


def _ttm(q: pd.Series) -> pd.Series:
    """Jumlah 4 kuartal berturut-turut (trailing twelve months)."""
    out = {}
    for i in range(3, len(q)):
        span = (q.index[i] - q.index[i - 3]).days
        if 250 <= span <= 300:
            out[q.index[i]] = float(q.iloc[i - 3:i + 1].sum())
    return pd.Series(out, dtype=float)


def _available(s: pd.Series) -> pd.Series:
    """Geser tanggal akhir periode ke tanggal laporan dianggap terbit."""
    if s.empty:
        return s
    lag = [ANNUAL_LAG_DAYS if d.month == 12 else QUARTER_LAG_DAYS for d in s.index]
    out = pd.Series(s.values, index=s.index + pd.to_timedelta(lag, unit="D"), dtype=float)
    return out[~out.index.duplicated(keep="last")].sort_index()


def statement_series(st: dict) -> dict[str, pd.Series]:
    """Laporan keuangan mentah -> seri per tanggal terbit: laba & pendapatan 12 bulan, ekuitas, utang, saham."""
    qi, ai = st.get("quarterly_income"), st.get("annual_income")
    qb, ab = st.get("quarterly_balance"), st.get("annual_balance")
    ni_names = ("Net Income Common Stockholders", "Net Income", "Net Income Including Noncontrolling Interests")
    rev_names = ("Total Revenue", "Operating Revenue")

    def combine(annual: pd.Series, quarterly: pd.Series) -> pd.Series:
        s = pd.concat([annual, quarterly])  # kuartal menimpa tahunan di tanggal yang sama
        return s[~s.index.duplicated(keep="last")].sort_index()

    out = {
        "net_income_ttm": combine(_row(ai, ni_names), _ttm(_row(qi, ni_names))),
        "revenue_ttm": combine(_row(ai, rev_names), _ttm(_row(qi, rev_names))),
        "equity": combine(_row(ab, ("Stockholders Equity", "Common Stock Equity")),
                          _row(qb, ("Stockholders Equity", "Common Stock Equity"))),
        "debt": combine(_row(ab, ("Total Debt",)), _row(qb, ("Total Debt",))),
        "shares": combine(_row(ab, ("Ordinary Shares Number", "Share Issued")),
                          _row(qb, ("Ordinary Shares Number", "Share Issued"))),
    }
    return {k: _available(v) for k, v in out.items() if not v.empty}


# ------------------------------------------------------------------ pembaruan
def _due(store: Store, key: str, force: bool) -> bool:
    last = store.last_fetch(key)
    return force or last is None or pd.Timestamp.now(tz="UTC").tz_localize(None) - last > pd.Timedelta(hours=REFRESH_HOURS)


def _shift_days(s: pd.Series, days: int) -> pd.Series:
    return pd.Series(s.values, index=s.index + pd.Timedelta(days=days)) if days else s


def update_fundamentals(store: Store, source: FundamentalSource | None = None,
                        progress: Callable[[str], None] = lambda m: None, force: bool = False) -> dict[str, int]:
    """Ambil data fundamental & makro (paling sering sekali sehari). Sumber yang gagal dilewati."""
    source = source or WebFundamentalSource()
    written: dict[str, int] = {}
    if not _due(store, "fundamentals", force):
        return written
    progress("Mengambil data fundamental dan makro…")

    def attempt(key: str, fn):
        try:
            written[key] = fn()
        except Exception as exc:
            log.warning("Gagal mengambil %s: %s", key, exc)

    for name, ticker in MACRO_YAHOO.items():
        attempt(name, lambda n=name, t=ticker: store.upsert_fundamental("MACRO", n, source.daily(t)))
    for name, (sid, lag) in MACRO_FRED.items():
        attempt(name, lambda n=name, i=sid, l=lag: store.upsert_fundamental("MACRO", n, _shift_days(source.fred(i), l)))
    attempt("FNG", lambda: store.upsert_fundamental("BTC", "FNG", source.fear_greed()))
    attempt("HASHRATE", lambda: store.upsert_fundamental("BTC", "HASHRATE", _shift_days(source.hash_rate(), 1)))

    fx = store.prices("USDIDR")
    for a in ASSETS:
        if not is_stock(a.key):
            continue
        progress(f"Mengambil laporan keuangan {a.name}…")

        def _stock(a=a):
            st = source.statements(a.ticker)
            series = statement_series(st)
            n = 0
            for name, s in series.items():
                # Laporan berdenominasi USD (mis. ADRO) dikonversi ke Rupiah dengan kurs saat terbit.
                if st.get("currency") == "USD" and a.currency == "IDR" and name != "shares" and len(fx):
                    rate = fx.reindex(fx.index.union(s.index)).sort_index().ffill().reindex(s.index)
                    s = s * rate
                n += store.upsert_fundamental(a.key, name, s)
            return n
        attempt(a.key, _stock)
    store.mark_fetch("fundamentals")
    return written


# ------------------------------------------------------------------ fitur model
def _ff(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    if s.empty:
        return pd.Series(np.nan, index=idx)
    return s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)


def _yoy(s: pd.Series) -> pd.Series:
    """Perubahan dibanding angka ±1 tahun sebelumnya (dibagi nilai absolut agar aman untuk rugi)."""
    out = {}
    for d, v in s.items():
        prev = s[s.index <= d - pd.Timedelta(days=330)]
        if len(prev) and prev.iloc[-1] != 0:
            out[d] = (v - prev.iloc[-1]) / abs(prev.iloc[-1])
    return pd.Series(out, dtype=float)


def _ratio_hist(s: pd.Series, idx: pd.DatetimeIndex, a: int, b: int) -> pd.Series:
    if s.empty:
        return pd.Series(np.nan, index=idx)
    r = s.rolling(a, min_periods=max(2, a // 2)).mean() / s.rolling(b, min_periods=max(2, b // 2)).mean() - 1
    return _ff(r, idx)


def fundamental_features(store: Store, asset: str, close: pd.Series | None = None) -> pd.DataFrame:
    """Fitur harian (index = tanggal harga) dari data fundamental & makro yang sudah tersedia saat itu."""
    close = store.prices(asset) if close is None else close
    if close.empty:
        return pd.DataFrame()
    idx = close.index
    f = pd.DataFrame(index=idx)
    dxy, us10, fed = (store.fundamental("MACRO", k) for k in ("DXY", "US10Y", "FEDFUNDS"))
    if len(dxy):
        f["dxy_ret_20"] = np.log(_ff(dxy, idx)).diff(20)
    if len(us10):
        f["us10y_chg_20"] = _ff(us10, idx).diff(20)
    if len(fed):
        f["fed_chg_63"] = _ff(fed, idx).diff(63)
    bi = store.fundamental("MACRO", "BIRATE")
    if len(bi) and len(fed) and (asset in ("USDIDR", "IHSG", "GOLD_IDR_G") or is_stock(asset)):
        f["bi_fed_spread"] = _ff(bi, idx) - _ff(fed, idx)
    if asset == "WTI":
        crude = store.fundamental("MACRO", "CRUDE_STOCKS")
        if len(crude):
            f["crude_stocks_chg_4w"] = _ff(crude.pct_change(4), idx)
    if asset == "BTC":
        fng = store.fundamental("BTC", "FNG")
        if len(fng):
            f["fng"] = _ff(fng, idx) / 100
        f["hash_ratio"] = _ratio_hist(store.fundamental("BTC", "HASHRATE"), idx, 30, 90)
    if is_stock(asset):
        ni, eq, debt, sh, rev = (store.fundamental(asset, k) for k in
                                 ("net_income_ttm", "equity", "debt", "shares", "revenue_ttm"))
        if len(ni) and len(sh):
            eps = _ff(ni, idx) / _ff(sh, idx)
            pe = (close / eps).where(eps > 0)
            lpe = np.log(pe)
            f["log_pe_rel"] = lpe - lpe.rolling(1260, min_periods=120).median()
        if len(eq) and len(sh):
            pbv = (close * _ff(sh, idx) / _ff(eq, idx)).where(_ff(eq, idx) > 0)
            lpb = np.log(pbv)
            f["log_pbv_rel"] = lpb - lpb.rolling(1260, min_periods=120).median()
        if len(ni) and len(eq):
            f["roe"] = _ff(ni, idx) / _ff(eq, idx)
        if len(debt) and len(eq):
            f["der"] = _ff(debt, idx) / _ff(eq, idx)
        if len(ni):
            f["ni_growth_yoy"] = _ff(_yoy(ni), idx)
        if len(rev):
            f["rev_growth_yoy"] = _ff(_yoy(rev), idx)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f.loc[:, f.notna().any()]


def report_dates(store: Store, asset: str) -> list[pd.Timestamp]:
    """Perkiraan tanggal terbit laporan keuangan (untuk penanda di grafik)."""
    if not is_stock(asset):
        return []
    return list(store.fundamental(asset, "net_income_ttm").index)


# ------------------------------------------------------------------ skor untuk Arah Tren
@dataclass
class Assessment:
    items: list[tuple[str, int]] = field(default_factory=list)  # (penjelasan, poin -1/0/+1)
    label: str | None = None    # untuk saham: Murah / Wajar / Mahal
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(p for _, p in self.items)

    @property
    def direction(self) -> int:
        return 1 if self.score >= 2 else -1 if self.score <= -2 else 0

    @property
    def available(self) -> bool:
        return bool(self.items)


def _fresh(s: pd.Series, key: str, today: pd.Timestamp) -> bool:
    return len(s) > 0 and (today - s.index[-1]).days <= STALE_DAYS.get(key, 30)


def _p(v: float, digits: int = 1) -> str:
    return f"{v * 100:+.{digits}f}%".replace(".", ",")


def _num(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def assess(store: Store, asset: str, today: pd.Timestamp | None = None) -> Assessment:
    """Skor fundamental/makro: tiap faktor +1 (mendukung naik), −1 (mendukung turun), atau 0."""
    res = Assessment()
    close = store.prices(asset)
    if close.empty:
        return res
    # Umur data diukur dari harga terakhir aset ini, agar konsisten dengan grafik yang ditampilkan.
    today = close.index[-1] if today is None else pd.Timestamp(today)
    stock = is_stock(asset)
    # Arah pengaruh tiap faktor makro terhadap harga aset ini (+1 searah, -1 berlawanan, 0 tidak dipakai).
    dollar_effect = {"WTI": -1, "GOLD_IDR_G": -1, "GOLD_USD": -1, "BTC": -1, "USDIDR": 1, "IHSG": -1}.get(asset, -1 if stock else 0)
    yield_effect = {"GOLD_IDR_G": -1, "GOLD_USD": -1, "BTC": -1, "USDIDR": 1, "IHSG": -1}.get(asset, -1 if stock else 0)
    fed_effect = yield_effect

    dxy = store.fundamental("MACRO", "DXY")
    if dollar_effect and _fresh(dxy, "DXY", today) and len(dxy) > 21:
        chg = dxy.iloc[-1] / dxy.iloc[-21] - 1
        pts = 0 if abs(chg) < 0.01 else int(np.sign(chg)) * dollar_effect
        trend = "menguat" if chg > 0 else "melemah"
        why = {1: "biasanya mendukung kenaikan", -1: "biasanya menekan harga", 0: "pengaruhnya kecil"}[pts]
        res.items.append((f"Dolar AS (DXY) {trend} {_p(abs(chg))[1:]} sebulan: {why}.", pts))
    us10 = store.fundamental("MACRO", "US10Y")
    if yield_effect and _fresh(us10, "US10Y", today) and len(us10) > 21:
        d = us10.iloc[-1] - us10.iloc[-21]
        pts = 0 if abs(d) < 0.25 else int(np.sign(d)) * yield_effect
        res.items.append((f"Imbal hasil obligasi AS 10 th {_num(us10.iloc[-1])}% "
                          f"({'+' if d >= 0 else '−'}{_num(abs(d))} poin sebulan).", pts))
    fed = store.fundamental("MACRO", "FEDFUNDS")
    if fed_effect and _fresh(fed, "FEDFUNDS", today):
        past = fed[fed.index <= today - pd.Timedelta(days=90)]
        if len(past):
            d = fed.iloc[-1] - past.iloc[-1]
            pts = 0 if abs(d) < 0.2 else int(np.sign(d)) * fed_effect
            act = "naik" if d > 0 else "turun" if d < 0 else "tetap"
            res.items.append((f"Suku bunga The Fed {_num(fed.iloc[-1])}%, {act} dalam 3 bulan terakhir.", pts))
    bi = store.fundamental("MACRO", "BIRATE")
    if asset == "USDIDR" and _fresh(bi, "BIRATE", today) and len(fed):
        spread = bi.iloc[-1] - fed.iloc[-1]
        pts = 1 if spread < 1.5 else -1 if spread > 3 else 0
        res.items.append((f"Selisih bunga Indonesia−AS {_num(spread)} poin: "
                          f"{'kurang menarik bagi dana asing (Rupiah rawan melemah)' if pts > 0 else 'menarik bagi dana asing' if pts < 0 else 'normal'}.", pts))
    if asset in ("GOLD_IDR_G", "IHSG") or stock:
        fx = store.prices("USDIDR")
        if len(fx) > 21:
            chg = fx.iloc[-1] / fx.iloc[-21] - 1
            if asset == "GOLD_IDR_G":
                pts = 0 if abs(chg) < 0.01 else int(np.sign(chg))
                res.items.append((f"Rupiah {'melemah' if chg > 0 else 'menguat'} {_p(abs(chg))[1:]} sebulan: "
                                  f"{'mengangkat' if chg > 0 else 'menekan'} harga emas dalam Rupiah.", pts))
            else:
                pts = 0 if abs(chg) < 0.02 else -int(np.sign(chg))
                res.items.append((f"Rupiah {'melemah' if chg > 0 else 'menguat'} {_p(abs(chg))[1:]} sebulan.", pts))
    if asset == "WTI":
        crude = store.fundamental("MACRO", "CRUDE_STOCKS")
        if _fresh(crude, "CRUDE_STOCKS", today) and len(crude) > 4:
            chg = crude.iloc[-1] / crude.iloc[-5] - 1
            pts = 0 if abs(chg) < 0.02 else -int(np.sign(chg))
            res.items.append((f"Stok minyak mentah AS {'naik' if chg > 0 else 'turun'} {_p(abs(chg))[1:]} "
                              f"dalam 4 minggu (data EIA): {'pasokan menumpuk' if chg > 0 else 'pasokan menipis'}.", pts))
    if asset == "BTC":
        fng = store.fundamental("BTC", "FNG")
        if _fresh(fng, "FNG", today):
            v = fng.iloc[-1]
            pts = 1 if v <= 25 else -1 if v >= 75 else 0
            mood = ("ketakutan ekstrem, secara historis sering jadi area beli" if pts > 0 else
                    "keserakahan ekstrem, rawan koreksi" if pts < 0 else "netral")
            res.items.append((f"Indeks Fear & Greed {v:.0f}: {mood}.", pts))
        hr = store.fundamental("BTC", "HASHRATE")
        if _fresh(hr, "HASHRATE", today) and len(hr) > 30:
            r = _ratio_hist(hr, hr.index[-1:], 30, 90).iloc[-1]
            if not pd.isna(r):
                pts = 1 if r > 0.03 else -1 if r < -0.03 else 0
                res.items.append((f"Hash rate jaringan {'naik' if r > 0 else 'turun'} {_p(abs(r))[1:]} "
                                  f"dibanding rata-rata 3 bulan: penambang {'yakin' if r > 0 else 'mengurangi'}.", pts))
    if stock:
        f = fundamental_features(store, asset, close)
        ni = store.fundamental(asset, "net_income_ttm")
        if not f.empty and _fresh(ni, "net_income_ttm", today):
            last = f.iloc[-1]
            eps = ni.iloc[-1] / store.fundamental(asset, "shares").iloc[-1] if len(store.fundamental(asset, "shares")) else np.nan
            pe = close.iloc[-1] / eps if eps and eps > 0 else np.nan
            rel = last.get("log_pe_rel", np.nan)
            basis = "PER"
            if pd.isna(rel):
                rel, basis = last.get("log_pbv_rel", np.nan), "PBV"
            if not pd.isna(rel):
                ratio = math.exp(rel)
                pts = 1 if ratio < 0.85 else -1 if ratio > 1.15 else 0
                res.label = {1: "Murah", -1: "Mahal", 0: "Wajar"}[pts]
                val = f"PER {_num(pe, 1)}×" if basis == "PER" and not pd.isna(pe) else basis
                res.items.append((f"Valuasi {res.label.lower()}: {val}, {abs(ratio - 1) * 100:.0f}% "
                                  f"{'di bawah' if ratio < 1 else 'di atas'} rata-rata 5 tahunnya.", pts))
            g = last.get("ni_growth_yoy", np.nan)
            if not pd.isna(g):
                pts = 1 if g > 0.10 else -1 if g < -0.10 else 0
                res.items.append((f"Laba bersih 12 bulan {'naik' if g >= 0 else 'turun'} {abs(g) * 100:.0f}% "
                                  f"dari tahun lalu.", pts))
            roe = last.get("roe", np.nan)
            if not pd.isna(roe):
                pts = 1 if roe >= 0.15 else -1 if roe < 0.05 else 0
                res.items.append((f"ROE {roe * 100:.0f}%: {'tinggi' if pts > 0 else 'rendah' if pts < 0 else 'cukup'}.", pts))
            der = last.get("der", np.nan)
            if not pd.isna(der):
                pts = -1 if der > 2 else 0
                res.items.append((f"Utang/ekuitas (DER) {_num(der)}×: {'tinggi' if pts else 'aman'}.", pts))
            res.notes.append(f"Laporan terakhir (perkiraan terbit): {ni.index[-1]:%d %b %Y}.")
        elif len(ni):
            res.notes.append("Laporan keuangan terakhir sudah lama; valuasi tidak dinilai.")
    return res
