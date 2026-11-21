"""Penyimpanan lokal SQLite: harga, prediksi, dan hasil evaluasi."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    asset TEXT NOT NULL,
    date  TEXT NOT NULL,          -- YYYY-MM-DD
    close REAL NOT NULL,
    PRIMARY KEY (asset, date)
);

-- Satu baris per (aset, tanggal dasar, horizon, model).
-- model = nama anggota ansambel, atau 'ensemble' untuk hasil gabungan final.
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    asset         TEXT NOT NULL,
    base_date     TEXT NOT NULL,  -- tanggal harga terakhir yang dipakai
    horizon       INTEGER NOT NULL, -- hari bursa ke depan
    model         TEXT NOT NULL,
    base_price    REAL NOT NULL,
    p_up          REAL NOT NULL,  -- peluang harga naik
    q10           REAL NOT NULL,  -- return log kuantil 10%
    q50           REAL NOT NULL,  -- return log median (setelah koreksi)
    q90           REAL NOT NULL,
    raw_q50       REAL,           -- median sebelum koreksi galat
    model_version TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (asset, base_date, horizon, model)
);

CREATE TABLE IF NOT EXISTS evaluations (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
    target_date   TEXT NOT NULL,
    actual_price  REAL NOT NULL,
    actual_ret    REAL NOT NULL,  -- return log aktual
    error         REAL NOT NULL,  -- actual_ret - q50
    raw_error     REAL,           -- actual_ret - raw_q50
    abs_pct_error REAL NOT NULL,  -- |harga prediksi - aktual| / aktual
    direction_hit INTEGER NOT NULL,
    in_range      INTEGER NOT NULL,
    evaluated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Metrik validasi saat pelatihan (backtest out-of-sample).
CREATE TABLE IF NOT EXISTS backtests (
    asset      TEXT NOT NULL,
    horizon    INTEGER NOT NULL,
    model      TEXT NOT NULL,
    n          INTEGER NOT NULL,
    hit_rate   REAL NOT NULL,
    mae_ret    REAL NOT NULL,
    coverage   REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (asset, horizon, model)
);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---------- harga ----------
    def upsert_prices(self, asset: str, series: pd.Series) -> int:
        series = series.dropna()
        rows = [(asset, pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for d, v in series.items()]
        self.conn.executemany(
            "INSERT INTO prices(asset, date, close) VALUES (?,?,?) "
            "ON CONFLICT(asset, date) DO UPDATE SET close=excluded.close",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def last_price_date(self, asset: str) -> str | None:
        row = self.conn.execute("SELECT MAX(date) FROM prices WHERE asset=?", (asset,)).fetchone()
        return row[0]

    def prices(self, asset: str) -> pd.Series:
        df = pd.read_sql_query(
            "SELECT date, close FROM prices WHERE asset=? ORDER BY date", self.conn, params=(asset,)
        )
        if df.empty:
            return pd.Series(dtype=float, name=asset)
        s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]), name=asset)
        return s

    # ---------- prediksi ----------
    def save_prediction(self, **row) -> None:
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        updates = ",".join(f"{c}=excluded.{c}" for c in row if c not in ("asset", "base_date", "horizon", "model"))
        self.conn.execute(
            f"INSERT INTO predictions({cols}) VALUES ({qs}) "
            f"ON CONFLICT(asset, base_date, horizon, model) DO UPDATE SET {updates}",
            tuple(row.values()),
        )
        self.conn.commit()

    def pending_predictions(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT p.* FROM predictions p LEFT JOIN evaluations e ON e.prediction_id = p.id "
            "WHERE e.prediction_id IS NULL",
            self.conn,
        )

    def save_evaluation(self, **row) -> None:
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        self.conn.execute(f"INSERT OR REPLACE INTO evaluations({cols}) VALUES ({qs})", tuple(row.values()))
        self.conn.commit()

    def evaluated(self, asset: str | None = None, horizon: int | None = None,
                  model: str | None = None) -> pd.DataFrame:
        q = ("SELECT p.asset, p.base_date, p.horizon, p.model, p.base_price, p.p_up, p.q10, p.q50, p.q90, "
             "p.raw_q50, e.* FROM predictions p JOIN evaluations e ON e.prediction_id = p.id WHERE 1=1")
        params: list = []
        for col, val in (("p.asset", asset), ("p.horizon", horizon), ("p.model", model)):
            if val is not None:
                q += f" AND {col}=?"
                params.append(val)
        q += " ORDER BY p.base_date"
        return pd.read_sql_query(q, self.conn, params=params)

    def latest_predictions(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT p.* FROM predictions p JOIN ("
            "  SELECT asset, horizon, MAX(base_date) AS bd FROM predictions WHERE model='ensemble' "
            "  GROUP BY asset, horizon) m "
            "ON p.asset=m.asset AND p.horizon=m.horizon AND p.base_date=m.bd "
            "WHERE p.model='ensemble' ORDER BY p.asset, p.horizon",
            self.conn,
        )

    # ---------- backtest ----------
    def save_backtest(self, asset: str, horizon: int, model: str, n: int,
                      hit_rate: float, mae_ret: float, coverage: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO backtests(asset, horizon, model, n, hit_rate, mae_ret, coverage) "
            "VALUES (?,?,?,?,?,?,?)",
            (asset, horizon, model, n, hit_rate, mae_ret, coverage),
        )
        self.conn.commit()

    def backtests(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM backtests ORDER BY asset, horizon, model", self.conn)
