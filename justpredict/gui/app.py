"""Aplikasi desktop (PySide6). Saat dibuka langsung menjalankan siklus update + prediksi."""
from __future__ import annotations

import logging
import math
import sys
import traceback

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPlainTextEdit,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from ..config import ASSET_BY_KEY, PREDICT_ASSETS, db_path
from ..db import Store
from ..engine import run_cycle
from ..report import accuracy_table, latest_table

SIGNAL_COLORS = {"BELI": "#1f9d55", "JUAL": "#d64545", "TAHAN": "#8a8a8a"}
DISCLAIMER = ("Prediksi bersifat probabilistik dan bisa salah. Bukan saran investasi; "
              "cek tab Akurasi sebelum mengambil keputusan.")


def fmt_num(v: float, cur: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if cur == "IDR":
        return f"Rp{v:,.0f}".replace(",", ".")
    return f"${v:,.2f}"


def fmt_pct(v, digits=1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:.{digits}f}%"


class CycleWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fetch: bool = True):
        super().__init__()
        self.fetch = fetch

    def run(self):
        store = Store(db_path())  # koneksi SQLite terpisah untuk thread ini
        try:
            rep = run_cycle(store, fetch=self.fetch, progress=self.progress.emit)
            self.finished.emit(rep)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            store.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"JustPredict {__version__}")
        self.resize(1200, 780)
        self.store = Store(db_path())
        self._thread: QThread | None = None

        root = QWidget()
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.status = QLabel("Memulai…")
        self.refresh_btn = QPushButton("Perbarui sekarang")
        self.refresh_btn.clicked.connect(lambda: self.start_cycle(fetch=True))
        top.addWidget(self.status, 1)
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.pred_table = self._make_table()
        self.acc_table = self._make_table()
        self.tabs.addTab(self.pred_table, "Prediksi")
        self.tabs.addTab(self._make_chart_tab(), "Grafik")
        self.tabs.addTab(self.acc_table, "Akurasi")
        splitter.addWidget(self.tabs)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        splitter.addWidget(self.log)
        splitter.setSizes([620, 120])
        layout.addWidget(splitter, 1)

        disclaimer = QLabel(DISCLAIMER)
        disclaimer.setStyleSheet("color: #888;")
        layout.addWidget(disclaimer)
        self.setCentralWidget(root)

        self.refresh_views()
        self.start_cycle(fetch=True)

    # ------------------------------------------------------------ widgets
    @staticmethod
    def _make_table() -> QTableWidget:
        t = QTableWidget()
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        t.setAlternatingRowColors(True)
        return t

    def _make_chart_tab(self) -> QWidget:
        import pyqtgraph as pg

        pg.setConfigOptions(antialias=True)
        w = QWidget()
        v = QVBoxLayout(w)
        self.asset_combo = QComboBox()
        for k in PREDICT_ASSETS:
            self.asset_combo.addItem(ASSET_BY_KEY[k].name, k)
        self.asset_combo.currentIndexChanged.connect(self.refresh_chart)
        v.addWidget(self.asset_combo)
        self.plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend()
        v.addWidget(self.plot, 1)
        return w

    # ------------------------------------------------------------ siklus
    def start_cycle(self, fetch: bool):
        if self._thread is not None:
            return
        self.refresh_btn.setEnabled(False)
        self.status.setText("Memperbarui data dan melatih model…")
        self._thread = QThread(self)
        self._worker = CycleWorker(fetch)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _cleanup_thread(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self.refresh_btn.setEnabled(True)

    def _on_progress(self, msg: str):
        self.status.setText(msg)
        self.log.appendPlainText(msg)

    def _on_finished(self, rep):
        self._cleanup_thread()
        msg = f"Selesai: {rep.evaluated} prediksi lama dievaluasi, {rep.predicted} prediksi baru."
        if rep.skipped:
            msg += f" Dilewati (data kurang): {', '.join(rep.skipped)}"
        self.status.setText(msg)
        self.log.appendPlainText(msg)
        self.refresh_views()

    def _on_failed(self, tb: str):
        self._cleanup_thread()
        self.status.setText("Terjadi kesalahan. Lihat log di bawah.")
        self.log.appendPlainText(tb)

    # ------------------------------------------------------------ tampilan
    def refresh_views(self):
        self.refresh_predictions()
        self.refresh_accuracy()
        self.refresh_chart()

    def refresh_predictions(self):
        df = latest_table(self.store)
        headers = ["Aset", "Horizon", "Per tanggal", "Harga terakhir", "Prediksi (median)",
                   "Perubahan", "Rentang 80%", "Peluang naik", "Sinyal", "Keandalan model"]
        t = self.pred_table
        t.clear()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(df))
        for i, r in enumerate(df.itertuples()):
            cur = r.mata_uang
            vals = [r.aset, r.horizon, r.per_tanggal, fmt_num(r.harga_terakhir, cur),
                    fmt_num(r.prediksi_median, cur), f"{r.perubahan_pct:+.2f}%",
                    f"{fmt_num(r.rentang_bawah, cur)} – {fmt_num(r.rentang_atas, cur)}",
                    fmt_pct(r.peluang_naik, 0), r.sinyal,
                    "Teruji" if r.teruji else "Belum terbukti"]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if j == len(vals) - 1 and not r.teruji:
                    item.setForeground(QColor("#b07d00"))
                    item.setToolTip("Di backtest, model belum mengalahkan tebakan naif. Sinyal ditahan.")
                if j == len(vals) - 2:
                    item.setForeground(QColor(SIGNAL_COLORS[r.sinyal]))
                    f = QFont()
                    f.setBold(True)
                    item.setFont(f)
                t.setItem(i, j, item)

    def refresh_accuracy(self):
        df = accuracy_table(self.store)
        headers = ["Aset", "Horizon", "Arah benar (backtest model)", "Arah benar (backtest naif)",
                   "Prediksi live dievaluasi", "Arah benar (live)", "Rata-rata galat harga",
                   "Aktual dalam rentang"]
        t = self.acc_table
        t.clear()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(df))
        for i, r in enumerate(df.itertuples()):
            gal = "–" if r.live_galat_pct is None or pd.isna(r.live_galat_pct) else f"{r.live_galat_pct:.2f}%"
            vals = [r.aset, r.horizon, fmt_pct(r.backtest_arah_lightgbm), fmt_pct(r.backtest_arah_naive),
                    str(r.live_n), fmt_pct(r.live_arah), gal, fmt_pct(r.live_dalam_rentang, 0)]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(v))

    def refresh_chart(self):
        import pyqtgraph as pg

        asset = self.asset_combo.currentData()
        self.plot.clear()
        if not asset:
            return
        prices = self.store.prices(asset).iloc[-250:]
        if prices.empty:
            return
        ts = lambda idx: [pd.Timestamp(d).timestamp() for d in idx]
        self.plot.plot(ts(prices.index), prices.values, pen=pg.mkPen("#4a90d9", width=2), name="Harga aktual")

        hist = self.store.evaluated(asset, 1, "ensemble")
        if not hist.empty:
            hist = hist[pd.to_datetime(hist.target_date) >= prices.index[0]]
            pred_price = hist.base_price * hist.q50.apply(math.exp)
            self.plot.plot(ts(pd.to_datetime(hist.target_date)), pred_price.values,
                           pen=None, symbol="o", symbolSize=5, symbolBrush="#e0a030",
                           name="Prediksi harian (lalu)")

        latest = latest_table(self.store)
        latest = latest[latest.asset_key == asset].sort_values("horizon_days") if not latest.empty else latest
        if not latest.empty:
            base_t = pd.Timestamp(latest.per_tanggal.iloc[0])
            base_p = float(latest.harga_terakhir.iloc[0])
            xs = [base_t.timestamp()] + [
                (base_t + pd.offsets.BDay(int(h))).timestamp() for h in latest.horizon_days]
            med = [base_p] + list(latest.prediksi_median)
            lo = [base_p] + list(latest.rentang_bawah)
            hi = [base_p] + list(latest.rentang_atas)
            c_lo = self.plot.plot(xs, lo, pen=pg.mkPen("#2ca58d", width=1, style=Qt.DashLine))
            c_hi = self.plot.plot(xs, hi, pen=pg.mkPen("#2ca58d", width=1, style=Qt.DashLine),
                                  name="Rentang 80%")
            self.plot.addItem(pg.FillBetweenItem(c_lo, c_hi, brush=(44, 165, 141, 50)))
            self.plot.plot(xs, med, pen=pg.mkPen("#2ca58d", width=2), symbol="t", name="Prediksi median")

    def closeEvent(self, event):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self.store.close()
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("JustPredict")
    win = MainWindow()
    win.show()
    return app.exec()
