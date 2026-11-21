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
        s = self.series[ticker].iloc[: self.cut]
        return s[s.index >= pd.Timestamp(start)]


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
