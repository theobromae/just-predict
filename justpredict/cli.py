"""Mode baris perintah: `python -m justpredict run` (tanpa GUI)."""
from __future__ import annotations

import argparse
import logging
import sys

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
    sub.add_parser("gui-smoke", help="Buka jendela dengan data sintetis lalu tutup (dipakai CI)")
    argv = list(sys.argv[1:] if argv is None else argv)
    # macOS kadang menambahkan argumen seperti "-psn_0_12345" saat aplikasi dibuka dari Finder.
    # Abaikan semua argumen yang tidak dikenal agar aplikasi tidak langsung keluar.
    args, _unknown = parser.parse_known_args([a for a in argv if not a.startswith("-psn")])

    from .applog import setup_logging
    setup_logging()
    if args.cmd in (None, "gui"):
        import os
        from .gui.app import main as gui_main
        code = gui_main()
        # Data sudah tersimpan saat jendela ditutup; keluar langsung agar aplikasi tidak
        # menggantung menunggu thread OpenMP/Qt dibersihkan.
        logging.shutdown()
        os._exit(code)
    if args.cmd == "gui-smoke":
        import os
        from .gui.smoke import run as smoke
        code = smoke()
        print(f"smoke: selesai dengan kode {code}", flush=True)
        logging.shutdown()
        os._exit(code)  # jangan menunggu pembersihan Qt/OpenMP saat keluar


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
