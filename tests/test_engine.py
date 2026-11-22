import math

import numpy as np
import pandas as pd
import pytest

from justpredict import adaptive
from justpredict.config import ASSETS, TROY_OUNCE_GRAMS
from justpredict.data.prices import gold_idr_per_gram, update_all
from justpredict.db import Store
from justpredict.engine import evaluate_matured, run_cycle
from justpredict.features import build_dataset, technical_features
from justpredict.report import latest_table, signal


class FakeSource:
    """Harga sintetis (random walk) yang bisa 'maju' satu hari per siklus."""

    def __init__(self, days=800, seed=0):
        rng = np.random.default_rng(seed)
        self.full_index = pd.bdate_range("2021-01-01", periods=days + 60)
        self.series = {}
        for i, a in enumerate(ASSETS):
            if a.ticker is None:
                continue
            start = {"IDR=X": 15000.0, "GC=F": 2000.0}.get(a.ticker, 100.0 + i)
            r = rng.normal(0.0003, 0.01, len(self.full_index))
            self.series[a.ticker] = pd.Series(start * np.exp(np.cumsum(r)), index=self.full_index)
        self.cut = days

    def advance(self, n=1):
        self.cut += n

    def fetch(self, ticker, start):
        c = self.series[ticker].iloc[: self.cut]
        c = c[c.index >= pd.Timestamp(start)]
        rng = np.random.default_rng(len(c))
        o = c.shift(1).fillna(c)
        spread = c * 0.005 * (1 + rng.random(len(c)))
        return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                             "close": c, "volume": rng.integers(1e5, 1e6, len(c)).astype(float)})

    intraday_bars_per_day = 7

    def fetch_intraday(self, ticker, start):
        daily = self.fetch(ticker, "2000-01-01").iloc[-120:]
        rows = []
        rng = np.random.default_rng(1)
        for d, r in daily.iterrows():
            px = r.open
            for k in range(self.intraday_bars_per_day):
                nxt = px * np.exp(rng.normal(0, 0.003))
                ts = pd.Timestamp(d) + pd.Timedelta(hours=2 + k)  # UTC 02:00.. = 09:00 WIB
                rows.append((ts, px, max(px, nxt) * 1.001, min(px, nxt) * 0.999, nxt, float(rng.integers(1e4, 1e5))))
                px = nxt
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
        df.index = df.index.tz_localize("UTC")
        return df[df.index >= start]


@pytest.fixture(autouse=True)
def fewer_path_anchors(monkeypatch):
    # Jalur penuh lambat; di tes cukup dua titik.
    import justpredict.engine as eng
    monkeypatch.setattr(eng, "PATH_ANCHORS", [3, 10])


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    yield s
    s.close()


def test_gold_per_gram():
    idx = pd.bdate_range("2024-01-01", periods=3)
    g = pd.Series([2000.0, 2010.0, 2020.0], index=idx)
    fx = pd.Series([16000.0, np.nan, 16100.0], index=idx)
    out = gold_idr_per_gram(g, fx.dropna())
    assert out.iloc[0] == pytest.approx(2000 * 16000 / TROY_OUNCE_GRAMS)
    assert out.iloc[1] == pytest.approx(2010 * 16000 / TROY_OUNCE_GRAMS)  # kurs diisi maju


def test_features_have_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 400))) * 100, index=idx)
    full = technical_features(close)
    partial = technical_features(close.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], partial)


def test_build_dataset_alignment():
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(np.linspace(100, 200, 400), index=idx)
    X, y, X_latest = build_dataset(close, 5)
    assert X_latest.index[-1] == idx[-1]
    assert y.index.max() == idx[-6]
    assert y.iloc[-1] == pytest.approx(math.log(close.iloc[-1] / close.iloc[-6]))


def test_incremental_update(store):
    src = FakeSource(days=500)
    first = update_all(store, src)
    assert first["WTI"] == 500 and first["GOLD_IDR_G"] > 0
    src.advance(3)
    second = update_all(store, src)
    assert second["WTI"] < 20  # hanya beberapa hari terakhir yang diambil ulang
    assert len(store.prices("WTI")) == 503


def test_full_cycle_evaluates_and_learns(store):
    src = FakeSource(days=600)
    rep = run_cycle(store, src, assets=["WTI", "GOLD_IDR_G"])
    assert rep.predicted == 6 and rep.evaluated == 0
    tbl = latest_table(store)
    assert set(tbl["sinyal"]) <= {"BELI", "JUAL", "TAHAN"}
    assert (tbl["rentang_bawah"] <= tbl["prediksi_median"]).all()
    assert (tbl["prediksi_median"] <= tbl["rentang_atas"]).all()

    # Hari berikutnya: prediksi harian kemarin harus dievaluasi.
    src.advance(1)
    rep2 = run_cycle(store, src, assets=["WTI", "GOLD_IDR_G"])
    assert rep2.evaluated >= 2  # ensemble + anggota, untuk horizon harian
    ev = store.evaluated("WTI", 1, "ensemble")
    assert len(ev) == 1 and ev["target_date"].iloc[0] == src.full_index[600].strftime("%Y-%m-%d")


def test_evaluate_waits_until_due(store):
    src = FakeSource(days=600)
    run_cycle(store, src, assets=["WTI"])
    src.advance(3)
    update_all(store, src)
    evaluate_matured(store)
    ev = store.evaluated("WTI")
    ev = ev[ev.model != "path"]
    assert set(ev["horizon"]) == {1}  # horizon 5 dan 21 belum jatuh tempo


def test_error_correction_learns_bias():
    # Prediksi selalu terlalu rendah 20% (mis. prediksi 20rb, aktual 25rb).
    n = 40
    err = math.log(25 / 20)
    df = pd.DataFrame({
        "base_date": pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "raw_error": np.full(n, err) + np.random.default_rng(0).normal(0, 0.05, n),
        "error": np.full(n, err),
        "in_range": np.zeros(n, dtype=int),
    })
    c = adaptive.error_correction(df)
    assert c.bias > 0.08
    assert c.width_scale > 1.3  # rentang terlalu sempit -> diperlebar
    pred = pd.Series({"p_up": 0.5, "q10": -0.05, "q50": 0.0, "q90": 0.05})
    out = c.apply(pred)
    assert out["q50"] == pytest.approx(c.bias)
    assert out["q10"] < out["q50"] < out["q90"]


def test_member_weights_favor_accurate_model():
    live = {"good": pd.Series([0.01] * 50), "bad": pd.Series([0.05] * 50)}
    w = adaptive.member_weights(live, {"good": 0.03, "bad": 0.03})
    assert w["good"] > 0.8 and sum(w.values()) == pytest.approx(1)


def test_signal_thresholds():
    assert signal(0.6, 0.01) == "BELI"
    assert signal(0.4, -0.01) == "JUAL"
    assert signal(0.6, -0.01) == "TAHAN"


def test_signal_withheld_when_model_unproven(store):
    from justpredict.report import is_proven
    store.save_backtest("WTI", 1, "lightgbm", 100, 0.51, 0.01, 0.8)
    store.save_backtest("WTI", 1, "naive", 100, 0.50, 0.01, 0.8)
    store.save_backtest("WTI", 5, "lightgbm", 100, 0.56, 0.01, 0.8)
    store.save_backtest("WTI", 5, "naive", 100, 0.53, 0.01, 0.8)
    bt = store.backtests()
    assert not is_proven(bt, "WTI", 1)
    assert is_proven(bt, "WTI", 5)


def test_ohlcv_stored_and_old_db_migrated(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE prices (asset TEXT NOT NULL, date TEXT NOT NULL, close REAL NOT NULL, "
                "PRIMARY KEY (asset, date))")
    con.executemany("INSERT INTO prices VALUES ('WTI', ?, 1.0)",
                    [(d.strftime("%Y-%m-%d"),) for d in pd.bdate_range("2021-01-01", periods=30)])
    con.commit(); con.close()
    s = Store(path)
    assert s.missing_ohlc("WTI")
    src = FakeSource(days=500)
    update_all(s, src)
    o = s.ohlcv("WTI")
    assert not s.missing_ohlc("WTI") and len(o) == 500
    assert (o["high"] >= o[["open", "close"]].max(axis=1)).all()
    g = s.ohlcv("GOLD_IDR_G")
    assert (g["high"] >= g["low"]).all() and g["volume"].isna().all()
    s.close()


def test_viewmodel_summary_and_resample(store):
    from justpredict.viewmodel import chart_data, resample_ohlcv, summarize
    src = FakeSource(days=700)
    run_cycle(store, src, assets=["TLKM"])
    daily = store.ohlcv("TLKM")
    weekly = resample_ohlcv(daily, "W-FRI")
    assert len(weekly) < len(daily) / 4
    assert weekly["high"].iloc[-1] == daily["high"].loc[daily.index > weekly.index[-2]].max()
    assert weekly.index[-1] == daily.index[-1]  # label = hari perdagangan terakhir
    cd = chart_data(store, "TLKM", "bulanan")
    assert {"rsi", "ma20", "ma50"} <= set(cd.columns)
    s = summarize(store, "TLKM", "mingguan")
    assert s.has_prediction and s.horizon == 5
    assert s.low <= s.median <= s.high
    assert "8 dari 10" in s.lines["range"] and s.signal in {"BELI", "JUAL", "TAHAN"}
    empty = summarize(store, "BTC", "harian")
    assert not empty.has_prediction


def test_intraday_and_path(store):
    from justpredict.viewmodel import chart_data, forecast_path, summarize
    src = FakeSource(days=700)
    run_cycle(store, src, assets=["WTI", "GOLD_IDR_G"])
    h1 = chart_data(store, "WTI", "1jam")
    h4 = chart_data(store, "WTI", "4jam")
    assert len(h1) > 500 and len(h4) < len(h1) / 2
    assert h1.index.min().year >= 2021  # waktu tersimpan dengan satuan yang benar
    assert len(chart_data(store, "GOLD_IDR_G", "1jam")) > 0
    path = forecast_path(store, "WTI", "harian")
    assert len(path) == 14  # anchor 1,3,5,10,21 -> 14 titik harian
    assert (path.lo <= path.mid).all() and (path.mid <= path.hi).all()
    assert path.date.is_monotonic_increasing
    s = summarize(store, "WTI", "harian")
    assert s.trend in {"Bullish", "Cenderung bullish", "Sideways", "Cenderung bearish", "Bearish", "Belum jelas"}
    assert s.ta is not None
    assert s.trend_details and s.trend_details[0][1] == "Model ML"
    # Prediksi per jam: 8 titik untuk 1 jam, horizon dalam batang.
    p1 = forecast_path(store, "WTI", "1jam")
    assert list(p1.h) == list(range(1, 9)) and p1.attrs["intraday"]
    s1 = summarize(store, "WTI", "1jam")
    assert s1.has_prediction and "1 jam lagi" in s1.lines["headline"]
    bt = store.backtests()
    assert {"intra_1jam_lgbm", "intra_1jam_naive"} <= set(bt.model)
    # Satu siklus lagi dengan data per jam baru: prediksi per jam yang lama dievaluasi.
    src.advance(2)
    run_cycle(store, src, assets=["WTI"], with_path=False)
    assert len(store.evaluated("WTI", 1, "intra_1jam")) > 0


def test_drawings_persist_and_index_mapping(tmp_path):
    from justpredict.gui.drawings import index_mapping
    db = tmp_path / "d.sqlite"
    s = Store(db)
    did = s.save_drawing("WTI", "trend", [(1.7e9, 70.0), (1.7e9 + 86400, 72.5)])
    s.save_drawing("WTI", "text", [(1.7e9, 71.0)], "beli di sini")
    s.save_drawing("BTC", "hline", [(1.7e9, 90000.0)])
    s.save_drawing("WTI", "trend", [(1.7e9, 60.0), (1.7e9 + 86400, 61.0)], drawing_id=did)
    s.close()
    s = Store(db)  # dibuka ulang: gambar tetap ada
    d = s.drawings("WTI")
    assert [x["kind"] for x in d] == ["trend", "text"] and d[0]["points"][0][1] == 60.0
    assert d[1]["text"] == "beli di sini"
    s.delete_drawing(d[1]["id"])
    s.clear_drawings("BTC")
    assert len(s.drawings("WTI")) == 1 and s.drawings("BTC") == []
    to_x, from_x = index_mapping(np.array([0.0, 3600, 7200, 90000]))
    assert to_x(3600) == 1 and to_x(5400) == 1.5 and to_x(90000 + 7200) == 5
    for x in (-2.0, 0.5, 2.7, 6.0):
        assert abs(to_x(from_x(x)) - x) < 1e-9


def test_technicals_detect_patterns():
    from justpredict.technicals import candle_patterns, detect, ta_features
    idx = pd.bdate_range("2024-01-01", periods=60)
    c = pd.Series(np.linspace(100, 80, 60), index=idx)
    df = pd.DataFrame({"open": c + 0.5, "high": c + 1, "low": c - 1, "close": c, "volume": 1000.0})
    # Hammer di akhir tren turun: badan kecil, sumbu bawah panjang.
    df.iloc[-1] = [79.5, 80.5, 76.0, 80.4, 1000.0]
    pats = candle_patterns(df)
    assert pats["hammer"].iloc[-1]
    res = detect(df)
    assert any("Hammer" in s.text for s in res.signals)
    f = ta_features(df)
    pd.testing.assert_frame_equal(f.iloc[:50], ta_features(df.iloc[:50]))  # tanpa bocoran masa depan


def test_close_only_prices_still_predict(store):
    """Data tanpa OHLC/volume (mis. database lama) tetap bisa dilatih."""
    from justpredict.engine import predict_asset
    idx = pd.bdate_range("2020-01-01", periods=700)
    store.upsert_prices("WTI", pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 700))), index=idx))
    assert store.ohlcv("WTI").dtypes.eq(float).all()
    assert predict_asset(store, "WTI", 1)


def test_path_with_only_old_ensemble_predictions(store):
    """Database dari versi sebelumnya: ada prediksi ensemble, belum ada jalur. Tidak boleh error."""
    from justpredict.viewmodel import forecast_path, summarize
    src = FakeSource(days=700)
    run_cycle(store, src, assets=["WTI"], with_path=False)
    path = forecast_path(store, "WTI", "harian")
    assert len(path) == 14  # anchor 1, 5, 21 dari ensemble
    assert path.mid.dtype == float
    summarize(store, "WTI", "harian")
    summarize(store, "WTI", "1jam")


class FakeFullSource(FakeSource):
    """FakeSource plus data fundamental, makro, dan berita (meniru format yfinance/FRED/RSS)."""

    def statements(self, ticker):
        end = self.full_index[self.cut - 1]
        q_dates = pd.date_range(end=end - pd.Timedelta(days=60), periods=6, freq="QE")[::-1]
        a_dates = pd.date_range(end=end - pd.Timedelta(days=120), periods=4, freq="YE")[::-1]
        grow = lambda n, base: [base * (1.05 ** -i) for i in range(n)]
        qi = pd.DataFrame([grow(6, 2.5e12), grow(6, 2e13)], index=["Net Income", "Total Revenue"], columns=q_dates)
        ai = pd.DataFrame([grow(4, 9e12), grow(4, 8e13)], index=["Net Income", "Total Revenue"], columns=a_dates)
        bs = lambda cols: pd.DataFrame([[6e13] * len(cols), [3e13] * len(cols), [1e11] * len(cols)],
                                       index=["Stockholders Equity", "Total Debt", "Ordinary Shares Number"], columns=cols)
        return {"currency": "IDR", "quarterly_income": qi, "annual_income": ai,
                "quarterly_balance": bs(q_dates), "annual_balance": bs(a_dates)}

    def _daily(self, start_value, drift):
        idx = self.full_index[: self.cut]
        return pd.Series(start_value * np.exp(np.linspace(0, drift, len(idx))), index=idx)

    def daily(self, ticker):
        return self._daily(100.0, -0.2) if ticker == "DX-Y.NYB" else self._daily(4.0, -0.3)

    def fred(self, series_id):
        idx = pd.date_range(self.full_index[0], self.full_index[self.cut - 1], freq="W-FRI")
        if series_id == "DFF":
            return pd.Series(np.where(np.arange(len(idx)) > len(idx) - 8, 4.0, 4.5), index=idx)
        if series_id == "WCESTUS1":
            return pd.Series(np.linspace(450e3, 400e3, len(idx)), index=idx)
        return pd.Series(6.0, index=pd.date_range(idx[0], idx[-1], freq="MS"))

    def fear_greed(self):
        return pd.Series(20.0, index=self.full_index[: self.cut])

    def hash_rate(self):
        return self._daily(500.0, 1.0)

    def fetch_news(self, query, lang):
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        titles = ["Harga naik menguat tajam", "Saham melonjak ke rekor tertinggi", "Laba tumbuh positif",
                  "Analis optimis, harga rebound", "Investor borong, harga melesat", "Pasar tidak melemah"]
        return [{"title": f"{t} ({query})", "link": f"https://example.com/{i}", "source": "Contoh",
                 "published": now - pd.Timedelta(hours=i)} for i, t in enumerate(titles)]


def test_sentiment_lexicon():
    from justpredict.data.news import score_title
    assert score_title("IHSG menguat, saham bank melonjak") == 1.0
    assert score_title("Harga minyak anjlok karena kekhawatiran resesi") == -1.0
    assert score_title("Saham tidak melemah") == 1.0  # negasi
    assert score_title("Rupiah menguat terhadap dolar", "USDIDR") == -1.0  # Rupiah kuat = USD/IDR turun
    assert score_title("Rapat umum pemegang saham") == 0.0


def test_statement_series_lagged_and_ttm():
    from justpredict.data.fundamentals import statement_series
    st = FakeFullSource().statements("TLKM.JK")
    s = statement_series(st)
    ni = s["net_income_ttm"]
    q_end = st["quarterly_income"].columns.max()
    lag = 90 if q_end.month == 12 else 45
    assert ni.index.max() == q_end + pd.Timedelta(days=lag)  # tanggal terbit, bukan akhir kuartal
    assert ni.iloc[-1] == pytest.approx(sum(2.5e12 * 1.05 ** -i for i in range(4)))
    assert set(s) == {"net_income_ttm", "revenue_ttm", "equity", "debt", "shares"}


def test_fundamentals_news_and_trend_details(store):
    from justpredict.data.fundamentals import assess, fundamental_features
    from justpredict.viewmodel import summarize
    src = FakeFullSource(days=700)
    run_cycle(store, src, assets=["TLKM", "BTC"], with_path=False, with_intraday=False)
    f = fundamental_features(store, "TLKM")
    assert {"log_pe_rel", "roe", "der", "ni_growth_yoy", "dxy_ret_20"} <= set(f.columns)
    today = store.prices("TLKM").index[-1]
    a = assess(store, "TLKM", today=today)
    texts = " ".join(t for t, _ in a.items)
    assert "Laba bersih" in texts and "ROE" in texts and "Dolar AS" in texts
    b = assess(store, "BTC", today=today)
    assert any("Fear & Greed 20" in t and p == 1 for t, p in b.items)
    bt = store.backtests()
    assert {"lightgbm_fund", "lightgbm_base"} <= set(bt[bt.asset == "TLKM"].model)
    s = summarize(store, "TLKM", "harian")
    kinds = [k for k, _, _ in s.trend_details]
    heads = [t for k, t, _ in s.trend_details if k == "head"]
    assert any(h.startswith("Fundamental") for h in heads) and any(h.startswith("Sentimen berita: positif") for h in heads)
    assert "link" in kinds and kinds[-1] == "total"
    assert s.sentiment.direction == 1 and s.reports
    # Fundamental tidak mengubah angka harga: median/rentang sama dengan model tanpa fundamental.
    import re
    ens = store.latest_predictions("ensemble")
    row = ens[(ens.asset == "TLKM") & (ens.horizon == 1)].iloc[0]
    shift = float(re.search(r"fund_shift=([+-][0-9.]+)", row.model_version).group(1))
    assert abs(shift) <= 0.10 + 1e-9 and s.fund_shift == pytest.approx(shift, abs=1e-3)
    bt = store.backtests()
    assert bt[bt.model == "lightgbm_fund"].logloss.notna().all()


def test_reconcile_fit_and_apply(store):
    from justpredict import reconcile
    # Deret dengan tren halus: semua rentang semestinya sepakat arah, rekonsiliasi bisa membantu.
    idx = pd.bdate_range("2019-01-01", periods=1500)
    rng = np.random.default_rng(3)
    drift = 0.0008 * np.sin(np.arange(1500) / 120)
    close = pd.Series(100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, 1500))), index=idx)
    hold = reconcile.holdout_predictions(close, None, [1, 3, 5, 10, 21])
    params = reconcile.fit(hold)
    assert set(params) == {1, 3, 5, 10, 21}
    for p in params.values():
        assert p.lam_q in reconcile.LAMBDAS and p.n > 50 and p.weight > 0
        if p.used:  # hanya dipakai kalau di paruh kedua lebih akurat
            assert p.mae_recon < p.mae_direct and p.hit_recon >= p.hit_direct - 0.005
    # Prediksi langsung yang tidak konsisten: rentang 5 hari berlawanan dengan rentang lain.
    for h, q, pu in ((1, 0.002, 0.6), (3, 0.006, 0.6), (5, -0.02, 0.35), (10, 0.02, 0.62), (21, 0.04, 0.65)):
        store.save_prediction(asset="WTI", base_date="2024-01-02", horizon=h, model="ensemble" if h in (1, 5, 21) else "path",
                              base_price=100.0, p_up=pu, q10=q - 0.03, q50=q, q90=q + 0.03, raw_q50=q, model_version="t")
    forced = {h: reconcile.Params(h, 0.5, 0.5, True, 0.02, 0.019, 0.52, 0.53, 100, 1 / (0.01 * h) ** 2)
              for h in (1, 3, 5, 10, 21)}
    assert reconcile.apply_live(store, "WTI", forced) == 5
    rec = store.latest_predictions("recon").set_index("horizon")
    assert rec.loc[5, "q50"] > -0.02 and rec.loc[5, "p_up"] > 0.35  # ditarik ke arah bersama
    assert rec.loc[5, "q90"] - rec.loc[5, "q10"] == pytest.approx(0.06)  # lebar rentang tetap


def test_conflict_withholds_signal_and_table(store):
    from justpredict.viewmodel import _conflict, confidence, forecast_table, main_rows
    for h, q, pu in ((1, 0.01, 0.7), (5, -0.03, 0.3), (21, 0.05, 0.7)):
        store.save_prediction(asset="WTI", base_date="2024-01-02", horizon=h, model="ensemble", base_price=100.0,
                              p_up=pu, q10=q - 0.03, q50=q, q90=q + 0.03, raw_q50=q, model_version="t")
    main = main_rows(store, "WTI")
    assert "mingguan" in _conflict(main, 1) and "harian" in _conflict(main, 5)
    assert confidence(0.65, 0.56) == ("Tinggi", 3)
    assert confidence(0.65, 0.56, conflict=True) == ("Sedang", 2)
    assert confidence(0.52, 0.60)[1] == 1 and confidence(0.7, 0.50)[1] == 1
    assert confidence(0.7, None) == ("Belum teruji", 0)
    assert confidence(0.35, 0.60, change=0.02)[1] == 1  # median naik tapi peluang bilang turun
    t = forecast_table(store, "WTI", "harian")
    assert len(t) == 14 and {"date", "mid", "lo", "hi", "confidence"} <= set(t.columns)
    assert (t.level <= 2).all()  # ada pertentangan arah: keyakinan maksimal Sedang
