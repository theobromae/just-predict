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
    open REAL, high REAL, low REAL, volume REAL,
    PRIMARY KEY (asset, date)
);

-- Harga per jam. ts = detik epoch UTC (awal batang).
CREATE TABLE IF NOT EXISTS prices_intraday (
    asset TEXT NOT NULL,
    ts    INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
    PRIMARY KEY (asset, ts)
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
    logloss    REAL,              -- ketepatan peluang naik (makin kecil makin baik)
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (asset, horizon, model)
);

-- Data fundamental dan makro, format panjang. date = tanggal data dianggap SUDAH TERSEDIA
-- (mis. akhir kuartal + jeda terbit laporan), agar model tidak memakai angka sebelum terbit.
-- asset = kunci aset, atau 'MACRO' untuk data makro bersama.
CREATE TABLE IF NOT EXISTS fundamentals (
    asset  TEXT NOT NULL,
    series TEXT NOT NULL,
    date   TEXT NOT NULL,
    value  REAL NOT NULL,
    PRIMARY KEY (asset, series, date)
);

-- Judul berita dari RSS beserta skor sentimennya (-1 negatif .. +1 positif).
CREATE TABLE IF NOT EXISTS news (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    asset     TEXT NOT NULL,
    published TEXT NOT NULL,      -- YYYY-MM-DD HH:MM (UTC)
    title     TEXT NOT NULL,
    link      TEXT,
    source    TEXT,
    score     REAL NOT NULL,
    UNIQUE (asset, title)
);

-- Kapan sebuah sumber terakhir diambil (agar data yang jarang berubah tidak diambil tiap buka).
CREATE TABLE IF NOT EXISTS fetch_log (
    key        TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL
);

-- Hasil uji rekonsiliasi antar-rentang per aset × rentang (dinilai di paruh kedua data uji).
CREATE TABLE IF NOT EXISTS reconcile (
    asset      TEXT NOT NULL,
    horizon    INTEGER NOT NULL,
    lam_q      REAL NOT NULL,
    lam_p      REAL NOT NULL,
    used       INTEGER NOT NULL,
    mae_direct REAL NOT NULL,
    mae_recon  REAL NOT NULL,
    hit_direct REAL NOT NULL,
    hit_recon  REAL NOT NULL,
    n          INTEGER NOT NULL,
    weight     REAL NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (asset, horizon)
);

-- Gambar pengguna di grafik (garis tren, kotak, Fibonacci, dll.), per aset.
-- points = JSON [[detik epoch, harga], ...] agar tetap di tempatnya di semua tampilan waktu.
CREATE TABLE IF NOT EXISTS drawings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    asset      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    points     TEXT NOT NULL,
    text       TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        # Database v0.1 awal hanya punya kolom close.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(prices)")}
        for c in ("open", "high", "low", "volume"):
            if c not in cols:
                self.conn.execute(f"ALTER TABLE prices ADD COLUMN {c} REAL")
        if "logloss" not in {r[1] for r in self.conn.execute("PRAGMA table_info(backtests)")}:
            self.conn.execute("ALTER TABLE backtests ADD COLUMN logloss REAL")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- harga ----------
    def upsert_prices(self, asset: str, data: pd.Series | pd.DataFrame) -> int:
        """Simpan harga. `data` = deret close, atau DataFrame berkolom open/high/low/close/volume."""
        df = data.to_frame("close") if isinstance(data, pd.Series) else data.copy()
        df = df.dropna(subset=["close"])
        for c in ("open", "high", "low", "volume"):
            if c not in df:
                df[c] = None
        f = lambda v: None if v is None or pd.isna(v) else float(v)
        rows = [(asset, pd.Timestamp(d).strftime("%Y-%m-%d"), float(r.close), f(r.open), f(r.high),
                 f(r.low), f(r.volume)) for d, r in zip(df.index, df.itertuples())]
        self.conn.executemany(
            "INSERT INTO prices(asset, date, close, open, high, low, volume) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(asset, date) DO UPDATE SET close=excluded.close, "
            "open=COALESCE(excluded.open, open), high=COALESCE(excluded.high, high), "
            "low=COALESCE(excluded.low, low), volume=COALESCE(excluded.volume, volume)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    # ---------- harga per jam ----------
    def upsert_intraday(self, asset: str, df: pd.DataFrame) -> int:
        """`df` ber-index waktu (tz-aware, atau naive = UTC) dengan kolom open/high/low/close/volume."""
        if df is None or df.empty:
            return 0
        df = df.dropna(subset=["close"])
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        ts = (idx.as_unit("ns").asi8 // 10**9).astype("int64")
        f = lambda v: None if v is None or pd.isna(v) else float(v)
        rows = [(asset, int(t), f(r.open), f(r.high), f(r.low), float(r.close), f(getattr(r, "volume", None)))
                for t, r in zip(ts, df.itertuples())]
        self.conn.executemany(
            "INSERT OR REPLACE INTO prices_intraday(asset, ts, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def last_intraday_ts(self, asset: str) -> int | None:
        return self.conn.execute("SELECT MAX(ts) FROM prices_intraday WHERE asset=?", (asset,)).fetchone()[0]

    def intraday(self, asset: str) -> pd.DataFrame:
        """OHLCV per jam, index waktu UTC (tz-aware)."""
        df = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM prices_intraday WHERE asset=? ORDER BY ts",
            self.conn, params=(asset,))
        df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
        df = df.astype(float)
        for c in ("open", "high", "low"):
            df[c] = df[c].fillna(df["close"])
        return df

    def missing_ohlc(self, asset: str) -> bool:
        """True kalau banyak baris belum punya OHLC (database dari versi lama)."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM prices WHERE asset=? AND open IS NULL", (asset,)).fetchone()
        return row[0] > 10

    def ohlcv(self, asset: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices WHERE asset=? ORDER BY date",
            self.conn, params=(asset,))
        df.index = pd.to_datetime(df.pop("date"))
        df = df.astype(float)  # kolom yang seluruhnya NULL terbaca sebagai object/None
        # Hari tanpa OHLC (mis. data lama): pakai close supaya candle tetap tergambar.
        for c in ("open", "high", "low"):
            df[c] = df[c].fillna(df["close"])
        return df

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

    def latest_predictions(self, model: str = "ensemble") -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT p.* FROM predictions p JOIN ("
            "  SELECT asset, horizon, MAX(base_date) AS bd FROM predictions WHERE model=? "
            "  GROUP BY asset, horizon) m "
            "ON p.asset=m.asset AND p.horizon=m.horizon AND p.base_date=m.bd "
            "WHERE p.model=? ORDER BY p.asset, p.horizon",
            self.conn, params=(model, model),
        )

    # ---------- backtest ----------
    def save_backtest(self, asset: str, horizon: int, model: str, n: int,
                      hit_rate: float, mae_ret: float, coverage: float, logloss: float | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO backtests(asset, horizon, model, n, hit_rate, mae_ret, coverage, logloss) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (asset, horizon, model, n, hit_rate, mae_ret, coverage, logloss),
        )
        self.conn.commit()

    def backtests(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM backtests ORDER BY asset, horizon, model", self.conn)

    # ---------- gambar pengguna ----------
    def drawings(self, asset: str) -> list[dict]:
        import json
        rows = self.conn.execute("SELECT id, kind, points, text FROM drawings WHERE asset=? ORDER BY id",
                                 (asset,)).fetchall()
        return [{"id": i, "kind": k, "points": json.loads(p), "text": t} for i, k, p, t in rows]

    def save_drawing(self, asset: str, kind: str, points: list, text: str | None = None,
                     drawing_id: int | None = None) -> int:
        import json
        pts = json.dumps([[float(x), float(y)] for x, y in points])
        if drawing_id is None:
            cur = self.conn.execute("INSERT INTO drawings(asset, kind, points, text) VALUES (?,?,?,?)",
                                    (asset, kind, pts, text))
            drawing_id = int(cur.lastrowid)
        else:
            self.conn.execute("UPDATE drawings SET points=?, text=? WHERE id=?", (pts, text, drawing_id))
        self.conn.commit()
        return drawing_id

    def delete_drawing(self, drawing_id: int) -> None:
        self.conn.execute("DELETE FROM drawings WHERE id=?", (drawing_id,))
        self.conn.commit()

    def clear_drawings(self, asset: str) -> None:
        self.conn.execute("DELETE FROM drawings WHERE asset=?", (asset,))
        self.conn.commit()

    # ---------- fundamental & makro ----------
    def upsert_fundamental(self, asset: str, series: str, values: pd.Series) -> int:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return 0
        rows = [(asset, series, pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for d, v in values.items()]
        self.conn.executemany("INSERT OR REPLACE INTO fundamentals(asset, series, date, value) VALUES (?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def fundamental(self, asset: str, series: str) -> pd.Series:
        rows = self.conn.execute("SELECT date, value FROM fundamentals WHERE asset=? AND series=? ORDER BY date",
                                 (asset, series)).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        d, v = zip(*rows)
        return pd.Series(v, index=pd.DatetimeIndex(pd.to_datetime(d)), dtype=float, name=series)

    def fundamental_series(self, asset: str) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT series FROM fundamentals WHERE asset=?", (asset,))]

    # ---------- berita ----------
    def upsert_news(self, asset: str, items: list[dict]) -> int:
        rows = [(asset, it["published"], it["title"], it.get("link"), it.get("source"), float(it["score"]))
                for it in items]
        if not rows:
            return 0
        self.conn.executemany("INSERT OR IGNORE INTO news(asset, published, title, link, source, score) "
                              "VALUES (?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def news(self, asset: str, since: str | None = None) -> pd.DataFrame:
        q, params = "SELECT published, title, link, source, score FROM news WHERE asset=?", [asset]
        if since:
            q += " AND published >= ?"
            params.append(since)
        return pd.read_sql_query(q + " ORDER BY published DESC", self.conn, params=params)

    # ---------- log pengambilan ----------
    def last_fetch(self, key: str) -> pd.Timestamp | None:
        r = self.conn.execute("SELECT fetched_at FROM fetch_log WHERE key=?", (key,)).fetchone()
        return pd.Timestamp(r[0]) if r else None

    def mark_fetch(self, key: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO fetch_log(key, fetched_at) VALUES (?, ?)",
                          (key, pd.Timestamp.now(tz="UTC").tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    # ---------- rekonsiliasi ----------
    def save_reconcile(self, asset: str, p) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO reconcile(asset, horizon, lam_q, lam_p, used, mae_direct, mae_recon, "
            "hit_direct, hit_recon, n, weight) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (asset, p.horizon, p.lam_q, p.lam_p, int(p.used), p.mae_direct, p.mae_recon, p.hit_direct,
             p.hit_recon, p.n, p.weight))
        self.conn.commit()

    def reconcile_stats(self, asset: str) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM reconcile WHERE asset=? ORDER BY horizon", self.conn, params=(asset,))
