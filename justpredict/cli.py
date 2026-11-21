"""Mode baris perintah: `python -m justpredict run` (tanpa GUI)."""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import db_path
from .db import Store
from .engine import run_cycle
from .report import accuracy_table, latest_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="justpredict")
    sub = parser.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="Update data, evaluasi, latih ulang, dan prediksi")
    run.add_argument("--no-fetch", action="store_true", help="Jangan ambil data baru")
    run.add_argument("--asset", action="append", help="Batasi ke aset tertentu (boleh berulang)")
    sub.add_parser("show", help="Tampilkan prediksi terakhir")
    sub.add_parser("gui", help="Buka aplikasi desktop")
    sub.add_parser("selftest", help="Uji cepat tanpa internet (dipakai CI setelah build)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd in (None, "gui"):
        from .gui.app import main as gui_main
        return gui_main()

    if args.cmd == "selftest":
        return selftest()

    store = Store(db_path())
    try:
        if args.cmd == "run":
            rep = run_cycle(store, fetch=not args.no_fetch, assets=args.asset, progress=print)
            print(f"\nDievaluasi: {rep.evaluated}, diprediksi: {rep.predicted}, dilewati: {rep.skipped}")
        with pd.option_context("display.width", 200, "display.max_columns", 30,
                               "display.float_format", "{:,.2f}".format):
            print("\n== Prediksi terbaru ==")
            print(latest_table(store).drop(columns=["asset_key", "horizon_days"], errors="ignore"))
            print("\n== Akurasi ==")
            print(accuracy_table(store))
    finally:
        store.close()
    return 0


def selftest() -> int:
    """Latih dan prediksi di data sintetis dalam folder sementara. Keluar 0 kalau berhasil."""
    import tempfile

    import numpy as np

    from .engine import predict_asset

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(f"{tmp}/selftest.sqlite")
        idx = pd.bdate_range("2020-01-01", periods=700)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 700))), index=idx)
        store.upsert_prices("WTI", prices)
        ok = predict_asset(store, "WTI", 1)
        n = len(latest_table(store))
        store.close()
    import yfinance  # noqa: F401  pastikan ikut terbungkus
    print(f"selftest ok={ok} rows={n}")
    return 0 if ok and n == 1 else 1
