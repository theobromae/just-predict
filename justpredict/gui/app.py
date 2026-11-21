"""Aplikasi desktop (PySide6): satu layar grafik dengan prediksi dan akurasi di sampingnya.

Saat dibuka, grafik langsung tampil dari data tersimpan, lalu siklus update + prediksi
berjalan di belakang dan grafik diperbarui otomatis setelah selesai.
"""
from __future__ import annotations

import html
import logging
import sys
import traceback

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPicture
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from .. import __version__
from ..config import ASSET_BY_KEY, db_path
from ..db import Store
from ..engine import run_cycle
from ..viewmodel import TIMEFRAMES, asset_list, chart_data, dec, money, summarize

UP = "#16a34a"
DOWN = "#dc2626"
NEUTRAL = "#6b7280"
FORECAST = "#7c3aed"
SIGNAL_COLORS = {"BELI": UP, "JUAL": DOWN, "TAHAN": NEUTRAL}
RELIABILITY_COLORS = {0: NEUTRAL, 1: "#d97706", 2: "#2563eb", 3: UP}
DISCLAIMER = "Prediksi bisa salah dan bukan saran investasi. Gunakan sebagai bahan pertimbangan."

pg.setConfigOptions(antialias=True, background="w", foreground="#374151")


# ====================================================================== grafik
class CandlestickItem(pg.GraphicsObject):
    """Candle: badan = buka→tutup, sumbu = tertinggi→terendah. Hijau naik, merah turun."""

    def __init__(self, x, o, h, l, c, width):
        super().__init__()
        self._picture = QPicture()
        p = QPainter(self._picture)
        for xi, oi, hi, li, ci in zip(x, o, h, l, c):
            color = QColor(UP if ci >= oi else DOWN)
            p.setPen(pg.mkPen(color, width=1))
            p.drawLine(QPointF(xi, li), QPointF(xi, hi))
            p.setBrush(pg.mkBrush(color))
            top, bottom = max(oi, ci), min(oi, ci)
            if top == bottom:
                p.drawLine(QPointF(xi - width / 2, top), QPointF(xi + width / 2, top))
            else:
                p.drawRect(QRectF(xi - width / 2, bottom, width, top - bottom))
        p.end()
        self._bounds = QRectF(min(x) - width, min(l), max(x) - min(x) + 2 * width, max(h) - min(l)) if len(x) else QRectF()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        return self._bounds


class VolumeAxis(pg.AxisItem):
    """Sumbu volume dengan satuan yang mudah dibaca: rb (ribu), jt (juta), M (miliar)."""

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            for div, unit in ((1e9, " M"), (1e6, " jt"), (1e3, " rb")):
                if abs(v) >= div:
                    out.append(f"{v / div:g}".replace(".", ",") + unit)
                    break
            else:
                out.append(f"{v:g}")
        return out


def _ts(index) -> np.ndarray:
    return np.array([pd.Timestamp(d).timestamp() for d in index], dtype=float)


# ====================================================================== worker
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
            self.finished.emit(run_cycle(store, fetch=self.fetch, progress=self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            store.close()


# ====================================================================== kartu panel kanan
def card(title: str) -> tuple[QFrame, QLabel]:
    f = QFrame()
    f.setObjectName("card")
    v = QVBoxLayout(f)
    v.setContentsMargins(14, 12, 14, 12)
    t = QLabel(title)
    t.setObjectName("cardTitle")
    body = QLabel()
    body.setWordWrap(True)
    body.setTextFormat(Qt.RichText)
    v.addWidget(t)
    v.addWidget(body)
    return f, body


STYLE = """
QMainWindow, QWidget#root { background: #f3f4f6; }
QFrame#card { background: white; border: 1px solid #e5e7eb; border-radius: 10px; }
QLabel#cardTitle { font-weight: 600; color: #6b7280; font-size: 12px; }
QListWidget { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 4px; }
QListWidget::item { padding: 8px 6px; border-radius: 6px; }
QListWidget::item:selected { background: #ede9fe; color: #111827; }
QPushButton[seg="true"] { padding: 5px 12px; border: 1px solid #d1d5db; background: white; }
QPushButton[seg="true"]:checked { background: #7c3aed; color: white; border-color: #7c3aed; }
QLabel#hover { color: #374151; font-size: 12px; }
QLabel#status { color: #6b7280; }
"""


# ====================================================================== jendela utama
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"JustPredict {__version__}")
        self.resize(1400, 860)
        self.setStyleSheet(STYLE)
        self.store = Store(db_path())
        self.settings = QSettings("JustPredict", "JustPredict")
        self._thread: QThread | None = None
        self._cd = pd.DataFrame()

        root = QWidget()
        root.setObjectName("root")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 8)
        outer.setSpacing(12)

        # ---- kiri: daftar aset
        self.asset_list = QListWidget()
        self.asset_list.setFixedWidth(240)
        self.asset_list.currentItemChanged.connect(lambda *_: self.refresh_chart())
        outer.addWidget(self.asset_list)

        # ---- tengah: toolbar + grafik
        center = QVBoxLayout()
        bar = QHBoxLayout()
        self.tf_group = self._segmented(bar, [("Harian", "harian"), ("Mingguan", "mingguan"), ("Bulanan", "bulanan")])
        bar.addSpacing(16)
        self.type_group = self._segmented(bar, [("Garis", "line"), ("Candle", "candle")])
        bar.addSpacing(16)
        self.cb_volume = QCheckBox("Volume")
        self.cb_rsi = QCheckBox("RSI")
        self.cb_ma = QCheckBox("Rata-rata 20/50")
        self.cb_past = QCheckBox("Prediksi lalu")
        for cb in (self.cb_volume, self.cb_rsi, self.cb_ma, self.cb_past):
            cb.setChecked(True)
            cb.toggled.connect(lambda *_: self.refresh_chart())
            bar.addWidget(cb)
        self.cb_rsi.setToolTip("RSI mengukur seberapa cepat harga naik/turun. Di atas 70 = jenuh beli, "
                               "di bawah 30 = jenuh jual.")
        self.cb_ma.setToolTip("Harga rata-rata 20 dan 50 periode terakhir. Harga di atas garis = tren naik.")
        self.cb_past.setToolTip("Titik = prediksi yang sudah jatuh tempo. Hijau arahnya benar, merah salah.")
        bar.addStretch(1)
        self.refresh_btn = QPushButton("Perbarui data")
        self.refresh_btn.clicked.connect(lambda: self.start_cycle(fetch=True))
        bar.addWidget(self.refresh_btn)
        center.addLayout(bar)

        self.hover = QLabel(" ")
        self.hover.setObjectName("hover")
        center.addWidget(self.hover)

        date_axis = lambda: pg.DateAxisItem(orientation="bottom")
        self.price_plot = pg.PlotWidget(axisItems={"bottom": date_axis()})
        self.vol_plot = pg.PlotWidget(axisItems={"bottom": date_axis(), "left": VolumeAxis("left")})
        self.rsi_plot = pg.PlotWidget(axisItems={"bottom": date_axis()})
        for p in (self.vol_plot, self.rsi_plot):
            p.setXLink(self.price_plot)
            p.setMaximumHeight(140)
            p.setMouseEnabled(y=False)
        for p in (self.price_plot, self.vol_plot, self.rsi_plot):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis("left").setWidth(70)
        self.price_plot.addLegend(offset=(10, 10))
        self.price_plot.setLabel("left", "Harga")
        self.vol_plot.setLabel("left", "Volume")
        self.rsi_plot.setLabel("left", "RSI")
        self.rsi_plot.setYRange(0, 100)
        center.addWidget(self.price_plot, 4)
        center.addWidget(self.vol_plot, 1)
        center.addWidget(self.rsi_plot, 1)
        self.price_plot.scene().sigMouseMoved.connect(self._on_mouse)

        status_row = QHBoxLayout()
        self.status = QLabel("Memulai…")
        self.status.setObjectName("status")
        self.log_btn = QPushButton("Lihat log")
        self.log_btn.setFlat(True)
        self.log_btn.clicked.connect(lambda: self.log.setVisible(not self.log.isVisible()))
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.log_btn)
        center.addLayout(status_row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(120)
        self.log.setVisible(False)
        center.addWidget(self.log)
        disclaimer = QLabel(DISCLAIMER)
        disclaimer.setObjectName("status")
        center.addWidget(disclaimer)
        outer.addLayout(center, 1)

        # ---- kanan: kartu prediksi, sinyal, akurasi, teknikal
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(10)
        f1, self.pred_card = card("PERKIRAAN")
        f2, self.signal_card = card("SARAN")
        f3, self.acc_card = card("SEBERAPA BISA DIPERCAYA?")
        f4, self.tech_card = card("KONDISI TEKNIKAL SAAT INI")
        for f in (f1, f2, f3, f4):
            sv.addWidget(f)
        sv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedWidth(340)
        scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        outer.addWidget(scroll)

        self.setCentralWidget(root)

        # ---- pulihkan pilihan terakhir
        self._check(self.tf_group, self.settings.value("timeframe", "harian"))
        self._check(self.type_group, self.settings.value("chart_type", "candle"))
        self.tf_group.buttonToggled.connect(self._on_tf)
        self.type_group.buttonToggled.connect(lambda b, on: on and self.refresh_chart())

        self.refresh_asset_list(select=self.settings.value("asset", None))
        self.start_cycle(fetch=True)

    # ------------------------------------------------------------ util widget
    def _segmented(self, layout, options) -> QButtonGroup:
        group = QButtonGroup(self)
        group.setExclusive(True)
        for label, key in options:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("seg", True)
            b.setProperty("key", key)
            group.addButton(b)
            layout.addWidget(b)
        group.buttons()[0].setChecked(True)
        return group

    @staticmethod
    def _check(group: QButtonGroup, key: str):
        for b in group.buttons():
            if b.property("key") == key:
                b.setChecked(True)

    @staticmethod
    def _selected(group: QButtonGroup) -> str:
        return group.checkedButton().property("key")

    @property
    def timeframe(self) -> str:
        return self._selected(self.tf_group)

    @property
    def chart_type(self) -> str:
        return self._selected(self.type_group)

    def current_asset(self) -> str | None:
        it = self.asset_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _on_tf(self, button, checked):
        if checked:
            self.refresh_asset_list(select=self.current_asset())

    # ------------------------------------------------------------ siklus
    def start_cycle(self, fetch: bool):
        if self._thread is not None:
            return
        self.refresh_btn.setEnabled(False)
        self.status.setText("Memperbarui data dan melatih model… grafik akan diperbarui otomatis.")
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
        msg = (f"Data terbaru sudah masuk. {rep.evaluated} prediksi lama dicek, "
               f"{rep.predicted} prediksi baru dibuat.")
        self.status.setText(msg)
        self.log.appendPlainText(msg + (f" Dilewati: {', '.join(rep.skipped)}" if rep.skipped else ""))
        self.refresh_asset_list(select=self.current_asset())

    def _on_failed(self, tb: str):
        self._cleanup_thread()
        self.status.setText("Terjadi kesalahan saat memperbarui. Klik 'Lihat log' untuk detail.")
        self.log.appendPlainText(tb)

    # ------------------------------------------------------------ daftar aset
    def refresh_asset_list(self, select: str | None = None):
        items = asset_list(self.store, self.timeframe)
        self.asset_list.blockSignals(True)
        self.asset_list.clear()
        row_to_select = 0
        for i, it in enumerate(items):
            price = money(it["price"], it["currency"]) if it["price"] is not None else "belum ada data"
            line2 = price
            if it["change"] is not None:
                arrow = "▲" if it["change"] > 0 else "▼"
                line2 += f"   {arrow} {dec(abs(it['change']) * 100)}%"
            li = QListWidgetItem(f"{it['name']}\n{line2}")
            li.setData(Qt.UserRole, it["asset"])
            if it["signal"]:
                li.setForeground(QColor("#111827"))
                li.setToolTip(f"Saran: {it['signal']}")
            self.asset_list.addItem(li)
            if it["asset"] == select:
                row_to_select = i
        self.asset_list.blockSignals(False)
        self.asset_list.setCurrentRow(row_to_select)
        self.refresh_chart()

    # ------------------------------------------------------------ grafik
    def refresh_chart(self):
        asset = self.current_asset()
        for p in (self.price_plot, self.vol_plot, self.rsi_plot):
            p.clear()
        if not asset:
            return
        tf = self.timeframe
        self.settings.setValue("asset", asset)
        self.settings.setValue("timeframe", tf)
        self.settings.setValue("chart_type", self.chart_type)

        cd = chart_data(self.store, asset, tf)
        self._cd = cd
        summary = summarize(self.store, asset, tf, cd)
        self.render_side(summary)
        if cd.empty:
            self.hover.setText("Belum ada data harga. Tunggu pembaruan selesai.")
            return

        a = ASSET_BY_KEY[asset]
        x = _ts(cd.index)
        step = float(np.median(np.diff(x))) if len(x) > 1 else 86400.0
        width = step * 0.7
        self.price_plot.setLabel("left", f"Harga ({'Rp' if a.currency == 'IDR' else 'USD'})")

        if self.chart_type == "candle":
            self.price_plot.addItem(CandlestickItem(x, cd.open.values, cd.high.values, cd.low.values,
                                                    cd.close.values, width))
            # Item tak terlihat agar legenda menampilkan "Harga".
            self.price_plot.plot([x[-1]], [cd.close.iloc[-1]], pen=None, symbol="s", symbolSize=8,
                                 symbolBrush=UP, symbolPen=None, name=f"Harga ({tf})")
        else:
            self.price_plot.plot(x, cd.close.values, pen=pg.mkPen("#2563eb", width=2), name=f"Harga ({tf})")

        if self.cb_ma.isChecked():
            self.price_plot.plot(x, cd.ma20.values, pen=pg.mkPen("#f59e0b", width=1.5), name="Rata-rata 20",
                                 connect="finite")
            self.price_plot.plot(x, cd.ma50.values, pen=pg.mkPen("#0ea5e9", width=1.5), name="Rata-rata 50",
                                 connect="finite")

        if self.cb_past.isChecked() and not summary.past.empty:
            past = summary.past
            px = _ts(pd.to_datetime(past.target_date))
            py = (past.base_price * np.exp(past.q50)).values
            brushes = [pg.mkBrush(UP if h else DOWN) for h in past.direction_hit]
            self.price_plot.addItem(pg.ScatterPlotItem(px, py, size=7, brush=brushes, pen=pg.mkPen("w", width=0.5)))
            self.price_plot.plot([], [], pen=None, symbol="o", symbolBrush=UP, name="Prediksi lalu (hijau = arah benar)")

        x_end = x[-1]
        if summary.has_prediction:
            bx = pd.Timestamp(summary.as_of).timestamp()
            tx = summary.target_date.timestamp()
            x_end = max(x_end, tx)
            lo = self.price_plot.plot([bx, tx], [summary.last_price, summary.low], pen=pg.mkPen(FORECAST, width=1, style=Qt.DotLine))
            hi = self.price_plot.plot([bx, tx], [summary.last_price, summary.high], pen=pg.mkPen(FORECAST, width=1, style=Qt.DotLine))
            self.price_plot.addItem(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(124, 58, 237, 40)))
            self.price_plot.plot([bx, tx], [summary.last_price, summary.median],
                                 pen=pg.mkPen(FORECAST, width=2.5, style=Qt.DashLine),
                                 symbol="o", symbolSize=[0, 9], symbolBrush=FORECAST, name="Perkiraan")
            label = pg.TextItem(html=f"<div style='background:#7c3aed;color:white;padding:2px 6px;"
                                     f"border-radius:4px'>{html.escape(money(summary.median, a.currency))}</div>",
                                anchor=(0, 0.5))
            label.setPos(tx, summary.median)
            self.price_plot.addItem(label)

        # ---- volume
        has_vol = bool(cd.volume.notna().any())
        self.vol_plot.setVisible(self.cb_volume.isChecked() and has_vol)
        if self.cb_volume.isChecked() and has_vol:
            up = (cd.close >= cd.open).values
            brushes = [pg.mkBrush(UP if u else DOWN) for u in up]
            self.vol_plot.addItem(pg.BarGraphItem(x=x, height=cd.volume.fillna(0).values, width=width,
                                                  brushes=brushes, pens=[pg.mkPen(None)] * len(x)))

        # ---- RSI
        self.rsi_plot.setVisible(self.cb_rsi.isChecked())
        if self.cb_rsi.isChecked():
            self.rsi_plot.addItem(pg.LinearRegionItem([30, 70], orientation="horizontal", movable=False,
                                                      brush=pg.mkBrush(124, 58, 237, 18), pen=pg.mkPen(None)))
            for lvl, txt, anchor in ((70, "jenuh beli", (1, 1)), (30, "jenuh jual", (1, 0))):
                self.rsi_plot.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen("#9ca3af", style=Qt.DashLine),
                                                      label=txt, labelOpts={"position": 0.98, "color": "#6b7280",
                                                                            "anchors": [anchor, anchor]}))
            self.rsi_plot.plot(x, cd.rsi.values, pen=pg.mkPen(FORECAST, width=1.5), connect="finite")
            self.rsi_plot.setYRange(0, 100)

        # ---- rentang awal: N batang terakhir + area perkiraan
        _, _, n_bars = TIMEFRAMES[tf]
        x0 = x[max(0, len(x) - n_bars)]
        # Ruang kosong di kanan untuk garis perkiraan dan labelnya.
        self.price_plot.setXRange(x0, x_end + (x_end - x0) * 0.12, padding=0)
        vis = cd.loc[cd.index >= pd.Timestamp(x0, unit="s")]
        ys = [vis.low.min(), vis.high.max()]
        if summary.has_prediction:
            ys += [summary.low, summary.high]
        self.price_plot.setYRange(min(ys), max(ys), padding=0.05)
        if has_vol:
            self.vol_plot.setYRange(0, float(vis.volume.max() or 1), padding=0.05)
        self._show_hover(len(cd) - 1)

    def _on_mouse(self, pos):
        if self._cd.empty or not self.price_plot.sceneBoundingRect().contains(pos):
            return
        xv = self.price_plot.plotItem.vb.mapSceneToView(pos).x()
        idx = int(np.argmin(np.abs(_ts(self._cd.index) - xv)))
        self._show_hover(idx)

    def _show_hover(self, i: int):
        if self._cd.empty:
            return
        r = self._cd.iloc[i]
        cur = ASSET_BY_KEY[self.current_asset()].currency
        txt = (f"<b>{self._cd.index[i]:%d %b %Y}</b> &nbsp; Buka {money(r.open, cur)} · "
               f"Tertinggi {money(r.high, cur)} · Terendah {money(r.low, cur)} · "
               f"<b>Tutup {money(r.close, cur)}</b>")
        if not pd.isna(r.volume):
            txt += f" · Volume {r.volume:,.0f}".replace(",", ".")
        if not pd.isna(r.rsi):
            txt += f" · RSI {r.rsi:.0f}"
        self.hover.setText(txt)

    # ------------------------------------------------------------ panel kanan
    def render_side(self, s):
        e = html.escape
        if s.has_prediction:
            color = UP if s.change_pct > 0 else DOWN if s.change_pct < 0 else NEUTRAL
            self.pred_card.setText(
                f"<div style='color:#374151'>{e(s.name)}</div>"
                f"<div style='font-size:13px;color:#6b7280'>{e(s.lines['headline'])}</div>"
                f"<div style='font-size:26px;font-weight:700;margin:4px 0'>{e(money(s.median, s.currency))}</div>"
                f"<div style='color:{color};font-weight:600'>{e(s.lines['change'])}</div>"
                f"<p>{e(s.lines['range'])}</p>"
                f"<p>{e(s.lines['p_up'])}</p>"
                f"<div style='color:#9ca3af;font-size:11px'>{e(s.lines['as_of'])}</div>")
        else:
            self.pred_card.setText(f"<b>{e(s.name)}</b><p>{e(s.lines['signal'])}</p>")

        sig_color = SIGNAL_COLORS[s.signal]
        self.signal_card.setText(
            f"<div style='font-size:22px;font-weight:700;color:{sig_color}'>{s.signal}</div>"
            f"<p>{e(s.lines['signal'])}</p>")

        rel_color = RELIABILITY_COLORS[s.reliability_level]
        dots = "●" * s.reliability_level + "○" * (3 - s.reliability_level)
        acc_html = "".join(f"<p>{e(line)}</p>" for line in s.lines.get("accuracy", "").split("\n") if line)
        self.acc_card.setText(
            f"<div style='font-size:16px;font-weight:700;color:{rel_color}'>{dots} &nbsp;{e(s.reliability)}</div>"
            f"{acc_html}")

        tech = s.lines.get("technical", "")
        self.tech_card.setText("".join(f"<p>{e(line)}</p>" for line in tech.split("\n") if line)
                               or "<p>Belum ada data.</p>")

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
