"""Data dan teks untuk layar grafik, dalam bahasa yang mudah dipahami orang awam.

Modul ini tidak bergantung pada Qt supaya mudah dites.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ASSET_BY_KEY, HORIZONS
from .db import Store
from .data.fundamentals import Assessment, assess, report_dates
from .data.news import Sentiment, assess_sentiment
from .data.prices import intraday_local, local_tz
from .engine import INTRADAY, PATH_POINTS, intraday_model
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
    model: str = "ensemble"  # nama model prediksi utama di database
    bar_hours: int = 0       # panjang satu batang (jam) untuk tampilan per jam; 0 = harian

    @property
    def intraday(self) -> bool:
        return self.source == "intraday"

    @property
    def backtest_models(self) -> tuple[str, str]:
        """(model, pembanding naif) di tabel backtest."""
        return (f"{self.model}_lgbm", f"{self.model}_naive") if self.intraday else ("lightgbm", "naive")


TIMEFRAMES: dict[str, Timeframe] = {
    "1jam": Timeframe("1 Jam", "intraday", INTRADAY["1jam"][0], 1, 120, "1jam", intraday_model("1jam"), 1),
    "4jam": Timeframe("4 Jam", "intraday", INTRADAY["4jam"][0], 1, 120, "4jam", intraday_model("4jam"), 4),
    "harian": Timeframe("Harian", "daily", None, 1, 120, "harian"),
    "mingguan": Timeframe("Mingguan", "daily", "W-FRI", 5, 104, "mingguan"),
    "bulanan": Timeframe("Bulanan", "daily", "ME", 21, 60, "bulanan"),
}
HORIZON_TEXT = {1: "besok (1 hari bursa)", 5: "1 minggu lagi", 21: "1 bulan lagi"}
INTRADAY_HORIZON_TEXT = {"1jam": "1 jam lagi", "4jam": "4 jam lagi"}
PATH_TEXT = {"harian": "14 hari bursa ke depan (±3 minggu)", "mingguan": "8 minggu ke depan (±2 bulan)",
             "bulanan": "6 bulan ke depan", "1jam": "8 jam bursa ke depan",
             "4jam": "10 batang 4 jam ke depan (±40 jam bursa)"}


# ------------------------------------------------------------------ format
HARI = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def tgl(ts, day: bool = False, time: bool = False) -> str:
    """Tanggal berbahasa Indonesia, mis. 'Sen, 24 Nov 2026 14:00'."""
    t = pd.Timestamp(ts)
    out = f"{t.day:02d} {BULAN[t.month - 1]} {t.year}"
    if day:
        out = f"{HARI[t.dayofweek]}, {out}"
    if time:
        out += f" {t:%H:%M}"
    return out


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


def chart_data(store: Store, asset: str, timeframe: str) -> pd.DataFrame:
    """OHLCV sesuai tampilan, plus indikator RSI 14 dan rata-rata bergerak 20/50."""
    tf = TIMEFRAMES[timeframe]
    if tf.source == "intraday":
        df = intraday_local(store, asset, tf.rule)
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
    # Rincian skor Arah Tren: (jenis, teks, poin). jenis: "head", "item", "link", "note", "total".
    trend_details: list[tuple[str, str, str]] = field(default_factory=list)
    reconciled: bool = False     # angka sudah diselaraskan dengan rentang lain
    conflict: str = ""           # penjelasan kalau arah antar-rentang bertentangan
    fundamental: Assessment | None = None
    sentiment: Sentiment | None = None
    fund_used: str = ""          # penjelasan apakah fundamental ikut dipakai model
    fund_shift: float = 0.0      # geseran peluang naik dari fundamental (sudah termasuk di p_up)
    reports: list[pd.Timestamp] = field(default_factory=list)  # perkiraan tanggal terbit laporan keuangan


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
        base = vol.iloc[-21:-1].mean()
        ratio = vol.iloc[-1] / base if base > 0 else 1.0
        if ratio > 1.5:
            out.append(f"Volume transaksi {dec(ratio)}× rata-rata: sedang ramai.")
        elif ratio < 0.6:
            out.append("Volume transaksi di bawah rata-rata: sedang sepi.")
    return out


def forecast_path(store: Store, asset: str, path_key: str) -> pd.DataFrame:
    """Titik-titik prediksi ke depan. Horizon yang tidak dilatih langsung diinterpolasi.

    Untuk tampilan per jam, `h` dihitung dalam batang (1 jam atau 4 jam) dan `date` hanya perkiraan
    (jam bursa tutup tidak diperhitungkan); grafik per jam memakai `h` sebagai posisi.
    """
    step, n = PATH_POINTS[path_key]
    intraday = path_key in INTRADAY
    models = [intraday_model(path_key)] if intraday else ["ensemble", "path"]
    parts = [store.latest_predictions(m) for m in models]
    parts = [df[df.asset == asset] for df in parts if not df.empty]
    parts = [p for p in parts if not p.empty]
    if not intraday:  # pakai hasil rekonsiliasi antar-rentang kalau sudah ada untuk tanggal terbaru
        rec = store.latest_predictions("recon")
        rec = rec[rec.asset == asset] if not rec.empty else rec
        if len(rec) and (not parts or rec.base_date.max() >= max(p.base_date.max() for p in parts)):
            parts = [rec]
    if not parts:
        return pd.DataFrame()
    rows = pd.concat(parts, ignore_index=True)
    num = ["horizon", "base_price", "p_up", "q10", "q50", "q90"]
    rows[num] = rows[num].apply(pd.to_numeric, errors="coerce")  # pastikan float, bukan object
    rows = rows.dropna(subset=num)
    if rows.empty:
        return pd.DataFrame()
    base_date = rows.base_date.max()
    rows = rows[rows.base_date == base_date].sort_values("horizon").drop_duplicates("horizon", keep="first")
    base = float(rows.base_price.iloc[0])
    hs = np.r_[0.0, rows.horizon.to_numpy(dtype=float)]
    bar_hours = INTRADAY[path_key][2] if intraday else 0
    out = []
    for k in range(1, n + 1):
        h = step * k
        if h > hs.max():
            break
        q = {c: float(np.interp(h, hs, np.r_[0.0, rows[c].to_numpy(dtype=float)])) for c in ("q10", "q50", "q90")}
        p_up = float(np.interp(h, hs, np.r_[0.5, rows.p_up.to_numpy(dtype=float)]))
        lo, mid, hi = sorted([q["q10"], q["q50"], q["q90"]])
        date = (pd.Timestamp(base_date) + pd.Timedelta(hours=bar_hours * h)) if intraday \
            else pd.Timestamp(base_date) + pd.offsets.BDay(h)
        out.append({"h": h, "date": date,
                    "mid": base * math.exp(mid), "lo": base * math.exp(lo),
                    "hi": base * math.exp(hi), "p_up": p_up})
    df = pd.DataFrame(out)
    df.attrs.update(base_date=pd.Timestamp(base_date), base_price=base, intraday=intraday)
    return df


WORD = {1: "naik", -1: "turun", 0: "datar"}


def _pts(v: int) -> str:
    return f"{v:+d}" if v else "0"


def _trend(path: pd.DataFrame, ta: TAResult | None, fund: Assessment | None = None,
           senti: Sentiment | None = None, fund_used: str = "", fund_shift: float = 0.0) -> tuple[str, int, str, list[tuple[str, str, str]]]:
    """Arah tren = model ML + analisis teknikal + fundamental/makro + sentimen berita.

    Setiap kelompok menyumbang +1 (naik), 0, atau −1 (turun). Total −4..+4 menentukan label.
    """
    d: list[tuple[str, str, str]] = []
    # 1. Model ML
    model_dir = 0
    d.append(("head", "Model ML", ""))
    if not path.empty:
        end = path.iloc[-1]
        chg = end["mid"] / path.attrs["base_price"] - 1
        if end.p_up >= 0.55 and chg > 0:
            model_dir = 1
        elif end.p_up <= 0.45 and chg < 0:
            model_dir = -1
        d.append(("item", f"Titik terakhir {'▲' if chg > 0 else '▼'} {dec(abs(chg) * 100)}%, "
                          f"peluang naik {pct(end.p_up)}", _pts(model_dir)))
    else:
        d.append(("item", "Belum ada prediksi", "0"))
    # 2. Teknikal
    score = ta.score if ta else 0
    ta_dir = 1 if score >= 2 else -1 if score <= -2 else 0
    d.append(("head", f"Teknikal (skor {score:+d})", _pts(ta_dir)))
    for sig in (ta.signals if ta else []):
        d.append(("item", sig.text, _pts(sig.direction)))
    if not ta or not ta.signals:
        d.append(("item", "Tidak ada sinyal menonjol", "0"))
    # 3. Fundamental / makro
    f_dir = fund.direction if fund else 0
    title = "Fundamental" + (f": {fund.label}" if fund and fund.label else "")
    d.append(("head", f"{title} (skor {fund.score:+d})" if fund and fund.available else title, _pts(f_dir)))
    if fund and fund.available:
        for text, p in fund.items:
            d.append(("item", text, _pts(p)))
    else:
        d.append(("item", "Data fundamental belum tersedia (diambil saat pembaruan)", "0"))
    for n in (fund.notes if fund else []):
        d.append(("note", n, ""))
    if fund_shift:
        d.append(("note", f"Peluang naik di kartu Perkiraan sudah termasuk geseran "
                          f"{fund_shift * 100:+.0f} poin dari fundamental.", ""))
    if fund_used:
        d.append(("note", fund_used, ""))
    # 4. Sentimen berita
    s_dir = senti.direction if senti else 0
    if senti and senti.n:
        d.append(("head", f"Sentimen berita: {senti.mood} ({senti.n} berita, 7 hari)", _pts(s_dir)))
        for h in senti.headlines:
            d.append(("link", h["title"] + (f" · {h['source']}" if h.get("source") else ""),
                      f"{'+' if h['score'] > 0 else '−'}|{h.get('link') or ''}"))
    else:
        d.append(("head", "Sentimen berita", "0"))
        d.append(("item", "Belum ada berita (diambil saat pembaruan)", "0"))

    parts = [model_dir, ta_dir, f_dir, s_dir]
    total = sum(parts)
    if total >= 3:
        label, level = "Bullish", 2
    elif total >= 1:
        label, level = "Cenderung bullish", 1
    elif total <= -3:
        label, level = "Bearish", -2
    elif total <= -1:
        label, level = "Cenderung bearish", -1
    else:
        label, level = ("Belum jelas" if 1 in parts and -1 in parts else "Sideways"), 0
    d.append(("total", f"Total {total:+d} dari 4 kelompok = {label}", _pts(total)))
    why = (f"Model: {WORD[model_dir]} · Teknikal: {WORD[ta_dir]} · Fundamental: {WORD[f_dir]} · "
           f"Sentimen: {WORD[s_dir]}.")
    return label, level, why, d


TREND_RULES = ("Cara hitung: ada 4 kelompok, masing-masing bernilai +1 (mendukung naik), 0, atau −1 "
               "(mendukung turun). Model ML +1 kalau peluang naik ≥ 55% dan perkiraannya naik. Teknikal dan "
               "fundamental: tiap faktor +1/−1, total ≥ +2 berarti +1, ≤ −2 berarti −1. Sentimen: rata-rata "
               "skor judul berita 7 hari terakhir ≥ +0,15 berarti +1 (minimal 5 berita). Total ≥ +3 = Bullish, "
               "+1..+2 = Cenderung bullish, 0 = Sideways (atau Belum jelas kalau ada yang berlawanan), "
               "−1..−2 = Cenderung bearish, ≤ −3 = Bearish.")


def _horizon_text(tf: Timeframe) -> str:
    if tf.intraday:
        return INTRADAY_HORIZON_TEXT.get(tf.path_key or "", f"{tf.bar_hours} jam lagi")
    return HORIZON_TEXT.get(tf.horizon, f"{tf.horizon} hari lagi")


def _target(tf: Timeframe, base_date) -> pd.Timestamp:
    b = pd.Timestamp(base_date)
    return b + pd.Timedelta(hours=tf.bar_hours * tf.horizon) if tf.intraday else b + pd.offsets.BDay(tf.horizon)


def _fund_used(bt: pd.DataFrame, asset: str, horizon: int) -> str:
    if bt.empty:
        return ""
    sel = bt[(bt.asset == asset) & (bt.horizon == horizon)].set_index("model")
    if "lightgbm_fund" not in sel.index or "lightgbm_base" not in sel.index or "logloss" not in sel:
        return ""
    lf, lb = sel.logloss["lightgbm_fund"], sel.logloss["lightgbm_base"]
    if pd.isna(lf) or pd.isna(lb):
        return ""
    if lf < lb:  # aturan sama dengan engine
        return ("Fundamental boleh menggeser peluang naik (maks. 10 poin): di uji data lama, peluang jadi "
                "lebih tepat. Angka harga tetap dari model teknikal.")
    return ("Fundamental tidak menggeser peluang naik: di uji data lama tidak membuat peluang lebih tepat. "
            "Fundamental tetap dihitung di Arah Tren.")


def main_rows(store: Store, asset: str) -> dict[int, object]:
    """Prediksi harian utama (1, 5, 21 hari) terbaru: hasil rekonsiliasi kalau ada, kalau tidak ensemble."""
    out = {}
    ens = store.latest_predictions("ensemble")
    rec = store.latest_predictions("recon")
    for h in HORIZONS.values():
        e = ens[(ens.asset == asset) & (ens.horizon == h)] if len(ens) else ens
        r = rec[(rec.asset == asset) & (rec.horizon == h)] if len(rec) else rec
        if len(r) and (not len(e) or r.base_date.iloc[0] >= e.base_date.iloc[0]):
            out[h] = r.iloc[0]
        elif len(e):
            out[h] = e.iloc[0]
    return out


def _direction(row) -> int:
    if row.p_up >= 0.55 and row.q50 > 0:
        return 1
    if row.p_up <= 0.45 and row.q50 < 0:
        return -1
    return 0


def _conflict(main: dict, horizon: int) -> str:
    """Teks penjelasan kalau rentang lain jelas berlawanan arah dengan rentang ini."""
    if horizon not in main:
        return ""
    mine = _direction(main[horizon])
    if not mine:
        return ""
    name = {v: k for k, v in HORIZONS.items()}
    opp = [name[h] for h, r in main.items() if h != horizon and _direction(r) == -mine]
    if not opp:
        return ""
    return f"{name.get(horizon, horizon)} {'naik' if mine > 0 else 'turun'}, {' dan '.join(opp)} {'turun' if mine > 0 else 'naik'}"


def summarize(store: Store, asset: str, timeframe: str, cd: pd.DataFrame | None = None) -> Summary:
    a = ASSET_BY_KEY[asset]
    tf = TIMEFRAMES[timeframe]
    horizon = tf.horizon
    bt_model, bt_naive = tf.backtest_models
    s = Summary(asset=asset, name=a.name, currency=a.currency, horizon=horizon,
                path_key=tf.path_key or "harian")
    cd = chart_data(store, asset, timeframe) if cd is None else cd

    preds = store.latest_predictions(tf.model)
    row = preds[(preds.asset == asset) & (preds.horizon == horizon)] if not preds.empty else preds
    bt = store.backtests()
    ev = store.evaluated(asset, horizon, tf.model)
    s.past = ev
    main = {} if tf.intraday else main_rows(store, asset)

    if len(row):
        r = row.iloc[0]
        fund_version = str(r.get("model_version") or "")
        if horizon in main and main[horizon].model == "recon":
            r = main[horizon]
            s.reconciled = "used=1" in str(r.model_version)
        s.has_prediction = True
        s.as_of = r.base_date
        s.last_price = r.base_price
        s.median = r.base_price * math.exp(r.q50)
        s.low = r.base_price * math.exp(r.q10)
        s.high = r.base_price * math.exp(r.q90)
        s.change_pct = math.exp(r.q50) - 1
        s.p_up = float(r.p_up)
        m = re.search(r"fund_shift=([+-][0-9.]+)", fund_version)
        s.fund_shift = float(m.group(1)) if m else 0.0
        s.proven = is_proven(bt, asset, horizon, bt_model, bt_naive)
        s.signal = signal(s.p_up, r.q50) if s.proven else "TAHAN"
        s.conflict = _conflict(main, horizon)
        if s.conflict and s.signal != "TAHAN":
            s.signal = "TAHAN"  # arah antar-rentang bertentangan: jangan beri saran beli/jual
        s.target_date = _target(tf, r.base_date)
        arrow = "▲ naik" if s.change_pct > 0 else "▼ turun" if s.change_pct < 0 else "tetap"
        s.lines["headline"] = f"Perkiraan harga {_horizon_text(tf)}"
        s.lines["change"] = (f"{arrow} {dec(abs(s.change_pct) * 100)}% dari harga terakhir "
                             f"{money(s.last_price, a.currency)}")
        s.lines["range"] = (f"Kemungkinan besar (8 dari 10 kali) harganya ada di antara "
                            f"{money(s.low, a.currency)} dan {money(s.high, a.currency)}.")
        s.lines["p_up"] = f"Peluang naik {pct(s.p_up)}: {_lean(s.p_up)}."
        s.lines["signal"] = _signal_text(s.signal, s.proven)
        if s.conflict:
            s.lines["signal"] = f"Arah antar-rentang bertentangan ({s.conflict}), jadi belum ada saran beli/jual."
        when = pd.Timestamp(r.base_date).strftime("%d %b %Y %H:%M" if tf.intraday else "%d %b %Y")
        s.lines["as_of"] = (f"Berdasarkan harga batang {when}." if tf.intraday
                            else f"Berdasarkan harga penutupan {when}.")
    else:
        s.lines["headline"] = "Belum ada prediksi"
        s.lines["signal"] = ("Data per jam belum cukup untuk membuat prediksi. Prediksi dibuat saat "
                             "pembaruan berikutnya." if tf.intraday
                             else "Data belum cukup untuk membuat prediksi aset ini.")

    # ---- keandalan
    sel = bt[(bt.asset == asset) & (bt.horizon == horizon)].set_index("model")["hit_rate"] if not bt.empty else pd.Series(dtype=float)
    acc = []
    if bt_model in sel:
        acc.append(f"Diuji pada data masa lalu: arah naik/turun benar {pct(sel[bt_model])}, "
                   f"tebakan sederhana {pct(sel.get(bt_naive, float('nan')))}.")
    if len(ev):
        hits = int(ev.direction_hit.sum())
        acc.append(f"Prediksi yang sudah terbukti: {len(ev)} kali, arahnya benar {hits} kali "
                   f"({pct(hits / len(ev))}). Rata-rata meleset {dec(ev.abs_pct_error.mean() * 100)}%.")
    else:
        acc.append("Belum ada prediksi yang jatuh tempo. Akurasi nyata akan muncul setelah aplikasi "
                   "dipakai beberapa hari.")
    acc.append("Sebagai patokan, arah benar 52–56% sudah tergolong bagus untuk harga pasar.")
    s.lines["accuracy"] = "\n".join(acc)

    if bt_model in sel:
        edge = sel[bt_model] - sel.get(bt_naive, 0.5)
        live = ev.direction_hit.mean() if len(ev) >= 20 else None
        score = sel[bt_model] if live is None else (sel[bt_model] + live) / 2
        if live is not None and live < 0.5:
            s.reliability, s.reliability_level = "Rendah", 1  # prediksi nyata belakangan sering salah arah
        elif s.proven and score >= 0.55 and edge > 0.02:
            s.reliability, s.reliability_level = "Baik", 3
        elif s.proven:
            s.reliability, s.reliability_level = "Sedang", 2
        else:
            s.reliability, s.reliability_level = "Rendah", 1

    if s.conflict and s.reliability_level > 1:
        s.reliability_level -= 1
        s.reliability = {1: "Rendah", 2: "Sedang"}[s.reliability_level]
    rec = store.reconcile_stats(asset) if not tf.intraday else pd.DataFrame()
    rec_h = rec[rec.horizon == horizon] if len(rec) else rec
    if len(rec_h) and bool(rec_h.used.iloc[0]):
        rr = rec_h.iloc[0]
        s.lines["recon"] = (f"Diselaraskan dengan rentang lain: di uji data lama galat {dec(rr.mae_recon * 100)}% vs "
                            f"{dec(rr.mae_direct * 100)}% tanpa penyelarasan, arah benar {pct(rr.hit_recon)} vs {pct(rr.hit_direct)}.")
    elif len(rec_h):
        s.lines["recon"] = "Penyelarasan antar-rentang tidak dipakai di sini karena tidak lebih akurat di uji data lama."

    s.ta = detect(cd, money=lambda v: money(v, a.currency)) if cd is not None and not cd.empty else None
    s.path = forecast_path(store, asset, s.path_key)
    try:
        s.fundamental = assess(store, asset)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Gagal menilai fundamental %s", asset)
    s.sentiment = assess_sentiment(store, asset)
    s.reports = report_dates(store, asset)
    s.fund_used = _fund_used(bt, asset, horizon) if not tf.intraday else ""
    s.trend, s.trend_level, s.lines["trend_why"], s.trend_details = _trend(
        s.path, s.ta, s.fundamental, s.sentiment, s.fund_used, s.fund_shift)
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

    tf = TIMEFRAMES[timeframe]
    preds = store.latest_predictions(tf.model)
    bt = store.backtests()
    out = []
    for k in PREDICT_ASSETS:
        a = ASSET_BY_KEY[k]
        row = preds[(preds.asset == k) & (preds.horizon == tf.horizon)] if not preds.empty else preds
        item = {"asset": k, "name": a.name, "price": None, "change": None, "signal": None, "currency": a.currency}
        if len(row):
            r = row.iloc[0]
            proven = is_proven(bt, k, tf.horizon, *tf.backtest_models)
            item.update(price=r.base_price, change=math.exp(r.q50) - 1,
                        signal=signal(r.p_up, r.q50) if proven else "TAHAN")
        else:
            last = store.prices(k)
            if len(last):
                item["price"] = float(last.iloc[-1])
        out.append(item)
    return out


def confidence(p_up: float, hit: float | None, conflict: bool = False,
               change: float | None = None) -> tuple[str, int]:
    """Tingkat keyakinan satu titik prediksi: (label, level 1–3).

    Menggabungkan seberapa yakin model soal arah (peluang arah yang diperkirakan) dengan rekam jejak
    model di rentang itu pada uji data lama.
    """
    dir_prob = max(p_up, 1 - p_up)
    if hit is None or pd.isna(hit):
        return "Belum teruji", 0
    skill = hit - 0.5
    disagree = change is not None and change != 0 and (change > 0) != (p_up > 0.5)
    if dir_prob < 0.55 or skill <= 0.01 or disagree:  # median dan peluang tidak sepakat arah = ragu
        level = 1
    elif dir_prob >= 0.60 and skill >= 0.04:
        level = 3
    else:
        level = 2
    if conflict:
        level = min(level, 2)
    return {1: "Rendah", 2: "Sedang", 3: "Tinggi"}[level], level


def forecast_table(store: Store, asset: str, timeframe: str) -> pd.DataFrame:
    """Tabel prediksi untuk tampilan tabel: satu baris per titik ke depan."""
    tf = TIMEFRAMES[timeframe]
    path = forecast_path(store, asset, tf.path_key or "harian")
    if path.empty:
        return path
    base = path.attrs["base_price"]
    bt = store.backtests()
    hits: dict[float, float] = {}
    if tf.intraday:
        sel = bt[(bt.asset == asset) & (bt.model == f"{tf.model}_lgbm")] if len(bt) else bt
        if len(sel):
            hits = {0.0: float(sel.hit_rate.iloc[0])}
    else:
        rec = store.reconcile_stats(asset)
        for r in rec.itertuples():
            hits[float(r.horizon)] = r.hit_recon if r.used else r.hit_direct
        if not hits and len(bt):  # belum ada rekonsiliasi: pakai backtest model utama
            sel = bt[(bt.asset == asset) & (bt.model == "lightgbm")]
            hits = {float(r.horizon): r.hit_rate for r in sel.itertuples()}
    main = {} if tf.intraday else main_rows(store, asset)
    conflict = any(_conflict(main, h) for h in main)
    rows = []
    for r in path.itertuples():
        hit = hits[min(hits, key=lambda k: abs(k - r.h))] if hits else None
        label, level = confidence(r.p_up, hit, conflict, r.mid / base - 1)
        up = r.p_up >= 0.5
        rows.append({
            "date": r.date, "h": r.h, "mid": r.mid, "lo": r.lo, "hi": r.hi, "change": r.mid / base - 1,
            "p_up": r.p_up, "confidence": label, "level": level,
            "why": (f"Peluang arah {'naik' if up else 'turun'} {pct(max(r.p_up, 1 - r.p_up))}"
                    + (f"; di uji data lama arah benar {pct(hit)}" if hit is not None else "")
                    + ("; arah antar-rentang bertentangan" if conflict else "")
                    + ("; median dan peluang tidak sepakat arah" if (r.mid > base) != up else "") + "."),
        })
    df = pd.DataFrame(rows)
    df.attrs.update(path.attrs)
    return df
