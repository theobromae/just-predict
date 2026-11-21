"""Uji asap GUI untuk CI: data sintetis yang mirip data Yahoo, buka jendela, coba semua tampilan.

Dijalankan pada aplikasi hasil build: `JustPredict.app/Contents/MacOS/JustPredict gui-smoke`.
Keluar 0 kalau semua langkah berhasil.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

log = logging.getLogger("justpredict.smoke")


def _synthetic(store, days: int = 900) -> None:
    from ..config import ASSETS

    rng = np.random.default_rng(7)
    for i, a in enumerate(ASSETS):
        if a.ticker is None:
            continue
        # Bitcoin diperdagangkan 7 hari seminggu; lainnya hari kerja.
        idx = (pd.date_range(end="2026-11-20", periods=days, freq="D") if a.key == "BTC"
               else pd.bdate_range(end="2026-11-20", periods=days))
        c = (15000.0 if a.key == "USDIDR" else 2000.0 if a.key == "GOLD_USD" else 100.0 + i) * \
            np.exp(np.cumsum(rng.normal(0.0002, 0.012, days)))
        c = pd.Series(c, index=idx)
        o = c.shift(1).fillna(c)
        spread = c * 0.004
        # Kurs dan indeks sering bervolume 0 di Yahoo.
        vol = 0.0 if a.key in ("USDIDR", "IHSG") else rng.integers(1e5, 1e7, days).astype(float)
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                           "close": c, "volume": vol})
        store.upsert_prices(a.key, df)
        # 1 jam: 60 hari terakhir, 7 batang per hari.
        rows = []
        for d, r in df.iloc[-60:].iterrows():
            px = r.open
            for k in range(7):
                nxt = px * np.exp(rng.normal(0, 0.003))
                rows.append((pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=2 + k),
                             px, max(px, nxt), min(px, nxt), nxt, 1000.0))
                px = nxt
        h = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
        store.upsert_intraday(a.key, h)
    from ..data.prices import gold_idr_per_gram, gold_idr_per_gram_intraday
    store.upsert_prices("GOLD_IDR_G", gold_idr_per_gram(store.ohlcv("GOLD_USD"), store.prices("USDIDR")))
    store.upsert_intraday("GOLD_IDR_G", gold_idr_per_gram_intraday(store.intraday("GOLD_USD"),
                                                                    store.intraday("USDIDR")))


def run() -> int:
    tmp = tempfile.mkdtemp(prefix="justpredict-smoke-")
    os.environ["JUSTPREDICT_HOME"] = tmp
    try:
        from PySide6.QtCore import QEventLoop, QTimer

        from .. import engine
        from ..config import db_path
        from ..db import Store
        from .app import MainWindow, make_app

        engine.PATH_ANCHORS = [3, 10]  # cukup untuk menguji jalur, lebih cepat
        engine.PREDICT_ASSETS = ["WTI", "BTC"]  # siklus lewat thread cukup dua aset
        store = Store(db_path())
        _synthetic(store)
        # Tanpa jalur dulu: meniru database dari versi sebelumnya (pernah bikin crash saat dibuka).
        engine.run_cycle(store, fetch=False, assets=["WTI", "BTC", "USDIDR", "GOLD_IDR_G"], with_path=False)
        store.close()
        print("smoke: data siap", flush=True)

        app = make_app()
        win = MainWindow(autostart=False)
        win.show()
        app.processEvents()
        for row in range(win.asset_list.count()):
            win.asset_list.setCurrentRow(row)
            for tf in win.tf_group.buttons():
                tf.setChecked(True)
                for ty in win.type_group.buttons():
                    ty.setChecked(True)
                    win._refresh_chart()  # tanpa penangkap error: kegagalan harus terlihat di CI
                    app.processEvents()
        print("smoke: semua tampilan tergambar", flush=True)

        # Jalankan siklus lewat thread seperti saat aplikasi dibuka (tanpa internet).
        loop = QEventLoop()
        result = {}
        win.start_cycle(fetch=False)
        win._worker.finished.connect(lambda rep: (result.setdefault("ok", rep), loop.quit()))
        win._worker.failed.connect(lambda tb: (result.setdefault("err", tb), loop.quit()))
        QTimer.singleShot(600_000, loop.quit)
        loop.exec()
        app.processEvents()
        if "err" in result or "ok" not in result:
            print("smoke: siklus gagal\n" + result.get("err", "timeout"), flush=True)
            return 1
        print(f"smoke: siklus selesai ({result['ok'].predicted} prediksi)", flush=True)
        for tf in win.tf_group.buttons():  # gambar ulang setelah jalur prediksi tersedia
            tf.setChecked(True)
            win._refresh_chart()
            app.processEvents()
        win.close()
        app.processEvents()
        return 0
    except Exception:
        print("smoke: GAGAL\n" + traceback.format_exc(), flush=True)
        return 1
