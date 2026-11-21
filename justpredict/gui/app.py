"""Aplikasi desktop (PySide6), tema gelap flat: satu layar grafik dengan panel penjelasan di kanan.

Saat dibuka, grafik langsung tampil dari data tersimpan, lalu siklus update + prediksi
berjalan di belakang dan grafik diperbarui otomatis setelah selesai.
"""
from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QPainter, QPalette, QPicture
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from .. import __version__
from ..config import ASSET_BY_KEY, db_path
from ..db import Store
from ..engine import run_cycle
from ..viewmodel import TIMEFRAMES, TREND_RULES, asset_list, chart_data, dec, local_tz, money, pct, summarize
from .drawings import TOOLS, DrawingLayer, index_mapping

# ---------------------------------------------------------------- palet (flat, gelap)
BG = "#0f1115"
SURFACE = "#171a21"
SURFACE_2 = "#1f232c"
BORDER = "#262b35"
TEXT = "#e5e7eb"
MUTED = "#9ca3af"
FAINT = "#6b7280"
ACCENT = "#8b5cf6"
UP = "#22c55e"
DOWN = "#ef4444"
NEUTRAL = "#9ca3af"
AMBER = "#f59e0b"
SKY = "#38bdf8"
SIGNAL_COLORS = {"BELI": UP, "JUAL": DOWN, "TAHAN": NEUTRAL}
RELIABILITY_COLORS = {0: FAINT, 1: AMBER, 2: SKY, 3: UP}
TREND_COLORS = {2: UP, 1: UP, 0: NEUTRAL, -1: DOWN, -2: DOWN}
DISCLAIMER = "Prediksi bisa salah dan bukan saran investasi. Gunakan sebagai bahan pertimbangan."

pg.setConfigOptions(antialias=True, background=BG, foreground=MUTED)

STYLE = f"""
* {{ color: {TEXT}; font-size: 13px; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#side {{ background: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QFrame#card {{ background: {SURFACE}; border: none; border-radius: 12px; }}
QLabel#cardTitle {{ color: {FAINT}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#faint {{ color: {FAINT}; font-size: 11px; }}
QLabel#big {{ font-size: 28px; font-weight: 700; }}
QLabel#key {{ color: {MUTED}; }}
QLabel#value {{ font-weight: 600; }}
QListWidget {{ background: {SURFACE}; border: none; border-radius: 12px; padding: 6px; outline: 0; }}
QListWidget::item {{ padding: 9px 8px; border-radius: 8px; color: {TEXT}; }}
QListWidget::item:selected {{ background: {SURFACE_2}; color: {TEXT}; }}
QListWidget::item:hover {{ background: {SURFACE_2}; }}
QPushButton {{ background: {SURFACE}; border: none; border-radius: 8px; padding: 6px 12px; color: {TEXT}; }}
QPushButton:hover {{ background: {SURFACE_2}; }}
QPushButton:disabled {{ color: {FAINT}; }}
QPushButton[seg="true"] {{ border-radius: 0px; padding: 6px 12px; }}
QPushButton[seg="true"]:checked {{ background: {ACCENT}; color: white; }}
QPushButton[segFirst="true"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
QPushButton[segLast="true"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
QPushButton#flat {{ background: transparent; color: {MUTED}; }}
QToolButton {{ background: {SURFACE}; border: none; border-radius: 8px; padding: 6px 10px; color: {TEXT}; }}
QToolButton:hover {{ background: {SURFACE_2}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#tool {{ min-width: 34px; max-width: 34px; min-height: 30px; padding: 0; font-size: 14px; }}
QToolButton#tool:checked {{ background: {ACCENT}; color: white; }}
QToolButton#expander {{ background: transparent; color: {MUTED}; padding: 2px 0; text-align: left; }}
QToolButton#expander:hover {{ color: {TEXT}; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER}; padding: 6px; border-radius: 8px; }}
QMenu::item {{ padding: 6px 18px 6px 10px; border-radius: 6px; }}
QMenu::item:selected {{ background: {SURFACE}; }}
QMenu::indicator {{ width: 14px; height: 14px; }}
QSplitter::handle {{ background: transparent; }}
QCheckBox {{ color: {MUTED}; spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px; background: {SURFACE_2}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; }}
QPlainTextEdit {{ background: {SURFACE}; border: none; border-radius: 8px; color: {MUTED}; }}
QProgressBar {{ background: {SURFACE_2}; border: none; border-radius: 3px; height: 6px; }}
QProgressBar::chunk {{ background: {UP}; border-radius: 3px; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: none; padding: 6px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {SURFACE_2}; border-radius: 4px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""


def dark_palette() -> QPalette:
    p = QPalette()
    for role, color in (
        (QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, SURFACE),
        (QPalette.AlternateBase, SURFACE_2), (QPalette.Text, TEXT), (QPalette.Button, SURFACE),
        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.ToolTipBase, SURFACE_2), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, FAINT),
    ):
        p.setColor(role, QColor(color))
    return p


# ---------------------------------------------------------------- komponen grafik
def _ts(index) -> np.ndarray:
    """Tanggal/waktu naive (waktu lokal) -> detik epoch."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize(local_tz())
    return (idx.as_unit("ns").asi8 / 1e9).astype(float)


class DateAxis(pg.DateAxisItem):
    """Label sumbu X berupa tanggal (mis. '12 Nov'); jam ikut ditampilkan saat diperbesar.

    Mode indeks (untuk grafik per jam): x = nomor batang, label diambil dari `self.labels`,
    sehingga jam malam dan akhir pekan tidak meninggalkan celah kosong.
    """

    labels: list[str] | None = None

    def tickValues(self, minVal, maxVal, size):
        if self.labels is None:
            return super().tickValues(minVal, maxVal, size)
        n = max(1, int(size / 110))
        step = max(1, int(np.ceil((maxVal - minVal) / n)))
        start = int(np.ceil(minVal / step) * step)
        return [(step, list(range(start, int(maxVal) + 1, step)))]

    def tickStrings(self, values, scale, spacing):
        if self.labels is not None:
            out = []
            for v in values:
                i = int(round(v))
                out.append(self.labels[i] if 0 <= i < len(self.labels) else "")
            return out
        if spacing < 86400:
            fmt = "%d %b %H:%M"
        elif spacing < 86400 * 200:
            fmt = "%d %b"
        else:
            fmt = "%b %Y"
        out = []
        for v in values:
            try:
                out.append(datetime.fromtimestamp(v).strftime(fmt))
            except (OverflowError, OSError, ValueError):
                out.append("")
        return out


class VolumeAxis(pg.AxisItem):
    """Sumbu volume: rb (ribu), jt (juta), M (miliar)."""

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
        self._bounds = (QRectF(min(x) - width, min(l), max(x) - min(x) + 2 * width, max(h) - min(l))
                        if len(x) else QRectF())

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        return self._bounds


# ---------------------------------------------------------------- worker
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


# ---------------------------------------------------------------- kartu panel kanan
class Card(QFrame):
    """Kartu flat: judul kecil + isi yang dibangun ulang setiap pembaruan."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        # Jangan dipadatkan saat panel penuh; biarkan panel kanan bergulir.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(16, 14, 16, 14)
        self.v.setSpacing(6)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        self.v.addWidget(t)
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        self.v.addLayout(self.body)

    def clear(self):
        def _clear(layout):
            while layout.count():
                it = layout.takeAt(0)
                if it.widget():
                    w = it.widget()
                    w.setParent(None)  # lepas sekarang juga agar tidak tertinggal di layar
                    w.deleteLater()
                elif it.layout():
                    _clear(it.layout())
        _clear(self.body)

    def label(self, text: str, name: str | None = None, color: str | None = None, wrap: bool = True,
              size: int | None = None, bold: bool = False) -> QLabel:
        lb = QLabel(text)
        lb.setWordWrap(wrap)
        if name:
            lb.setObjectName(name)
        css = []
        if color:
            css.append(f"color: {color};")
        if size:
            css.append(f"font-size: {size}px;")
        if bold:
            css.append("font-weight: 700;")
        if css:
            lb.setStyleSheet(" ".join(css))
        self.body.addWidget(lb)
        return lb

    def rows(self, pairs: list[tuple[str, str]]):
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(6)
        for i, (k, v) in enumerate(pairs):
            kl, vl = QLabel(k), QLabel(v)
            kl.setObjectName("key")
            vl.setObjectName("value")
            vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            vl.setWordWrap(True)
            g.addWidget(kl, i, 0)
            g.addWidget(vl, i, 1)
        g.setColumnStretch(1, 1)
        self.body.addLayout(g)

    def bullet(self, text: str, direction: int):
        h = QHBoxLayout()
        h.setSpacing(8)
        icon = QLabel({1: "▲", -1: "▼"}.get(direction, "●"))
        icon.setStyleSheet(f"color: {TREND_COLORS[direction]}; font-size: 11px;")
        icon.setFixedWidth(14)
        icon.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        t = QLabel(text)
        t.setWordWrap(True)
        h.addWidget(icon)
        h.addWidget(t, 1)
        self.body.addLayout(h)

    def pill(self, text: str, color: str):
        c = QColor(color)
        lb = QLabel(text)
        lb.setStyleSheet(f"background: rgba({c.red()},{c.green()},{c.blue()},40); color: {color};"
                         f" border-radius: 8px; padding: 6px 12px; font-size: 20px; font-weight: 700;")
        lb.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        h = QHBoxLayout()
        h.addWidget(lb)
        h.addStretch(1)
        self.body.addLayout(h)


# ---------------------------------------------------------------- jendela utama
class MainWindow(QMainWindow):
    def __init__(self, autostart: bool = True):
        super().__init__()
        self.setWindowTitle(f"JustPredict {__version__}")
        self.setStyleSheet(STYLE)
        self.setMinimumSize(960, 600)
        self.store = Store(db_path())
        import os
        # Folder data khusus (tes/CI) memakai pengaturan terpisah agar pilihan pengguna tidak tertimpa.
        self.settings = QSettings("JustPredict", "JustPredict-test" if os.environ.get("JUSTPREDICT_HOME") else "JustPredict")
        self._thread: QThread | None = None
        self._cd = pd.DataFrame()
        self._x = np.array([])
        self._trend_open = self.settings.value("trend_details_open", False, type=bool)

        root = QWidget()
        root.setObjectName("root")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 8)
        outer.setSpacing(0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(12)
        outer.addWidget(splitter)
        self.splitter = splitter

        # ---- kiri: daftar aset
        self.asset_list = QListWidget()
        self.asset_list.setMinimumWidth(170)
        self.asset_list.setMaximumWidth(260)
        self.asset_list.currentItemChanged.connect(lambda *_: self.refresh_chart())
        splitter.addWidget(self.asset_list)

        # ---- tengah: toolbar + grafik
        center_w = QWidget()
        center = QVBoxLayout(center_w)
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(8)
        bar = QHBoxLayout()
        bar.setSpacing(0)
        self.tf_group = self._segmented(bar, [(tf.label, k) for k, tf in TIMEFRAMES.items()])
        bar.addSpacing(10)
        self.type_group = self._segmented(bar, [("Garis", "line"), ("Candle", "candle")])
        bar.addSpacing(10)
        # Indikator di menu dropdown agar toolbar tidak memaksa jendela melebar.
        self.ind_btn = QToolButton()
        self.ind_btn.setText("Indikator ▾")
        self.ind_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.ind_btn)
        self.cb_volume = self._toggle(menu, "volume", "Volume", "Jumlah transaksi per batang.")
        self.cb_rsi = self._toggle(menu, "rsi", "RSI",
                                   "RSI mengukur kecepatan naik/turun harga. Di atas 70 = jenuh beli, di bawah 30 = jenuh jual.")
        self.cb_ma = self._toggle(menu, "ma", "MA 20/50",
                                  "Rata-rata harga 20 dan 50 batang terakhir. Harga di atas garis = tren naik.")
        self.cb_sr = self._toggle(menu, "sr", "Support/Resistance",
                                  "Support = level yang sering menahan harga turun. Resistance = level yang sering menahan harga naik.")
        self.cb_past = self._toggle(menu, "past", "Prediksi lalu",
                                    "Titik = prediksi yang sudah jatuh tempo. Hijau arahnya benar, merah salah.")
        menu.setToolTipsVisible(True)
        self.ind_btn.setMenu(menu)
        bar.addWidget(self.ind_btn)
        bar.addStretch(1)
        self.refresh_btn = QPushButton("⟳  Perbarui")
        self.refresh_btn.setToolTip("Ambil data terbaru dan latih ulang model")
        self.refresh_btn.clicked.connect(lambda: self.start_cycle(fetch=True))
        bar.addWidget(self.refresh_btn)
        center.addLayout(bar)

        self.hover = QLabel(" ")
        self.hover.setObjectName("muted")
        self.hover.setTextFormat(Qt.RichText)
        center.addWidget(self.hover)

        self.price_plot = pg.PlotWidget(axisItems={"bottom": DateAxis(orientation="bottom")})
        self.vol_plot = pg.PlotWidget(axisItems={"bottom": DateAxis(orientation="bottom"), "left": VolumeAxis("left")})
        self.rsi_plot = pg.PlotWidget(axisItems={"bottom": DateAxis(orientation="bottom")})
        for p in (self.vol_plot, self.rsi_plot):
            p.setXLink(self.price_plot)
            p.setMaximumHeight(130)
            p.setMouseEnabled(y=False)
        for p in (self.price_plot, self.vol_plot, self.rsi_plot):
            p.showGrid(x=True, y=True, alpha=0.08)
            p.getAxis("left").setWidth(72)
            for ax in ("left", "bottom"):
                p.getAxis(ax).setPen(pg.mkPen(BORDER))
                p.getAxis(ax).setTextPen(pg.mkPen(MUTED))
        legend = self.price_plot.addLegend(offset=(10, 10), labelTextColor=MUTED)
        legend.setBrush(pg.mkBrush(SURFACE + "cc"))
        self.vol_plot.setLabel("left", "Volume")
        self.rsi_plot.setLabel("left", "RSI")
        # Alat gambar: strip vertikal di kiri grafik (seperti TradingView).
        chart_row = QHBoxLayout()
        chart_row.setSpacing(8)
        tools = QVBoxLayout()
        tools.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for key, icon, tip in TOOLS:
            b = QToolButton()
            b.setObjectName("tool")
            b.setText(icon)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setProperty("key", key)
            self.tool_group.addButton(b)
            tools.addWidget(b)
        self.tool_group.buttons()[0].setChecked(True)
        clear_btn = QToolButton()
        clear_btn.setObjectName("tool")
        clear_btn.setText("🗑")
        clear_btn.setToolTip("Hapus semua gambar di aset ini")
        clear_btn.clicked.connect(self._clear_drawings)
        tools.addSpacing(8)
        tools.addWidget(clear_btn)
        tools.addStretch(1)
        chart_row.addLayout(tools)
        plots = QVBoxLayout()
        plots.setSpacing(8)
        plots.addWidget(self.price_plot, 4)
        plots.addWidget(self.vol_plot, 1)
        plots.addWidget(self.rsi_plot, 1)
        chart_row.addLayout(plots, 1)
        center.addLayout(chart_row, 1)
        self.price_plot.scene().sigMouseMoved.connect(self._on_mouse)
        self.drawings = DrawingLayer(self.price_plot, self.store)
        self.tool_group.buttonToggled.connect(
            lambda b, on: on and self.drawings.tool != b.property("key") and self.drawings.set_tool(b.property("key")))
        self.drawings.toolChanged.connect(lambda k: self._check(self.tool_group, k))

        status_row = QHBoxLayout()
        self.status = QLabel("Memulai…")
        self.status.setObjectName("faint")
        self.log_btn = QPushButton("Lihat log")
        self.log_btn.setObjectName("flat")
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
        disclaimer.setObjectName("faint")
        disclaimer.setWordWrap(True)
        center.addWidget(disclaimer)
        center_w.setMinimumWidth(440)
        splitter.addWidget(center_w)

        # ---- kanan: kartu penjelasan
        side = QWidget()
        side.setObjectName("side")
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 4, 0)
        sv.setSpacing(10)
        self.card_trend = Card("ARAH TREN")
        self.card_pred = Card("PERKIRAAN")
        self.card_signal = Card("SARAN")
        self.card_rel = Card("SEBERAPA BISA DIPERCAYA")
        self.card_ta = Card("SINYAL TEKNIKAL")
        for c in (self.card_trend, self.card_pred, self.card_signal, self.card_rel, self.card_ta):
            sv.addWidget(c)
        sv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(460)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(root)
        self._fit_to_screen()

        # ---- pulihkan pilihan terakhir
        self._check(self.tf_group, self.settings.value("timeframe", "harian"))
        self._check(self.type_group, self.settings.value("chart_type", "candle"))
        self.tf_group.buttonToggled.connect(self._on_tf)
        self.type_group.buttonToggled.connect(lambda b, on: on and self.refresh_chart())

        self.refresh_asset_list(select=self.settings.value("asset", None))
        if autostart:
            self.start_cycle(fetch=True)

    # ------------------------------------------------------------ util widget
    def _fit_to_screen(self):
        """Ukuran awal mengikuti layar (maks. 92% lebar, 90% tinggi area yang bisa dipakai), di tengah."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
            return
        geo = screen.availableGeometry()
        w = min(1500, int(geo.width() * 0.92))
        h = min(950, int(geo.height() * 0.90))
        w, h = max(w, min(960, geo.width())), max(h, min(600, geo.height()))
        self.resize(w, h)
        self.move(geo.x() + (geo.width() - w) // 2, geo.y() + (geo.height() - h) // 2)
        side = 340 if w >= 1300 else 300
        left = 220 if w >= 1300 else 180
        self.splitter.setSizes([left, max(440, w - side - left - 60), side])

    def _toggle(self, menu: QMenu, key: str, label: str, tip: str) -> QAction:
        act = QAction(label, menu)
        act.setCheckable(True)
        act.setChecked(self.settings.value(f"ind_{key}", True, type=bool))
        act.setToolTip(tip)
        act.toggled.connect(lambda on, k=key: (self.settings.setValue(f"ind_{k}", on), self.refresh_chart()))
        menu.addAction(act)
        return act

    def _clear_drawings(self):
        if not self.drawings.count():
            return
        import os
        if not os.environ.get("JUSTPREDICT_NO_DIALOG"):
            ans = QMessageBox.question(self, "Hapus gambar", "Hapus semua gambar di aset ini?")
            if ans != QMessageBox.Yes:
                return
        self.drawings.clear_all()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.drawings.set_tool("select")
            return
        super().keyPressEvent(ev)

    def _segmented(self, layout, options) -> QButtonGroup:
        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, (label, key) in enumerate(options):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("seg", True)
            b.setProperty("segFirst", i == 0)
            b.setProperty("segLast", i == len(options) - 1)
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

    @property
    def timeframe(self) -> str:
        return self.tf_group.checkedButton().property("key")

    @property
    def chart_type(self) -> str:
        return self.type_group.checkedButton().property("key")

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
        self.status.setText("Memperbarui data dan melatih model… grafik diperbarui otomatis setelah selesai.")
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
            line2 = money(it["price"], it["currency"]) if it["price"] is not None else "belum ada data"
            if it["change"] is not None:
                arrow = "▲" if it["change"] > 0 else "▼"
                line2 += f"   {arrow} {dec(abs(it['change']) * 100)}%"
            li = QListWidgetItem(f"{it['name']}\n{line2}")
            li.setData(Qt.UserRole, it["asset"])
            if it["signal"]:
                li.setToolTip(f"Saran: {it['signal']}")
            self.asset_list.addItem(li)
            if it["asset"] == select:
                row_to_select = i
        self.asset_list.blockSignals(False)
        self.asset_list.setCurrentRow(row_to_select)
        self.refresh_chart()

    # ------------------------------------------------------------ grafik
    def refresh_chart(self):
        try:
            self._refresh_chart()
        except Exception:
            logging.getLogger("justpredict.gui").exception("Gagal menggambar grafik")
            self.status.setText("Grafik gagal digambar untuk pilihan ini. Detail ada di log.")

    def _refresh_chart(self):
        asset = self.current_asset()
        for p in (self.price_plot, self.vol_plot, self.rsi_plot):
            p.clear()
        if not asset:
            return
        tf_key = self.timeframe
        tf = TIMEFRAMES[tf_key]
        self.settings.setValue("asset", asset)
        self.settings.setValue("timeframe", tf_key)
        self.settings.setValue("chart_type", self.chart_type)

        cd = chart_data(self.store, asset, tf_key)
        self._cd = cd
        s = summarize(self.store, asset, tf_key, cd)
        self.render_side(s, tf)
        if cd.empty:
            self.drawings.set_context(None)
            msg = ("Belum ada data per jam. Data ini diambil saat pembaruan berikutnya."
                   if tf.source == "intraday" else "Belum ada data harga. Tunggu pembaruan selesai.")
            self.hover.setText(msg)
            return

        a = ASSET_BY_KEY[asset]
        index_mode = tf.source == "intraday"
        labels = None
        if index_mode:
            labels = [d.strftime("%d %b %H:%M") for d in cd.index]
            if not s.path.empty:  # label perkiraan untuk batang ke depan
                labels += [f"+{int(h * tf.bar_hours)} jam" for h in range(1, int(s.path.h.max()) + 1)]
        for p in (self.price_plot, self.vol_plot, self.rsi_plot):
            p.getAxis("bottom").labels = labels
        x = np.arange(len(cd), dtype=float) if index_mode else _ts(cd.index)
        self._x = x
        step = float(np.median(np.diff(x))) if len(x) > 1 else 86400.0
        width = step * 0.7
        self.price_plot.setLabel("left", "Rp" if a.currency == "IDR" else "USD")

        if self.chart_type == "candle":
            self.price_plot.addItem(CandlestickItem(x, cd.open.values, cd.high.values, cd.low.values,
                                                    cd.close.values, width))
            self.price_plot.plot([], [], pen=None, symbol="s", symbolSize=8, symbolBrush=UP, symbolPen=None,
                                 name=f"Harga ({tf.label.lower()})")
        else:
            self.price_plot.plot(x, cd.close.values, pen=pg.mkPen(SKY, width=2), name=f"Harga ({tf.label.lower()})")

        if self.cb_ma.isChecked():
            self.price_plot.plot(x, cd.ma20.values, pen=pg.mkPen(AMBER, width=1.4), name="MA 20", connect="finite")
            self.price_plot.plot(x, cd.ma50.values, pen=pg.mkPen("#a78bfa", width=1.4), name="MA 50", connect="finite")

        # ---- support / resistance
        if self.cb_sr.isChecked() and s.ta:
            for lvl in s.ta.supports:
                self.price_plot.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen(UP, width=1, style=Qt.DashLine),
                                                        label=f"Support {money(lvl, a.currency)}",
                                                        labelOpts={"position": 0.55, "color": UP, "fill": BG + "cc",
                                                                   "anchors": [(0.5, 1), (0.5, 1)]}))
            for lvl in s.ta.resistances:
                self.price_plot.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen(DOWN, width=1, style=Qt.DashLine),
                                                        label=f"Resistance {money(lvl, a.currency)}",
                                                        labelOpts={"position": 0.55, "color": DOWN, "fill": BG + "cc",
                                                                   "anchors": [(0.5, 0), (0.5, 0)]}))

        # ---- penanda pola candle (30 batang terakhir)
        if s.ta and s.ta.pattern_marks:
            marks = [(d, n, dr) for d, n, dr in s.ta.pattern_marks if dr != 0 and d in cd.index]
            if marks:
                mx = (np.array([cd.index.get_loc(d) for d, _, _ in marks], dtype=float) if index_mode
                      else _ts([d for d, _, _ in marks]))
                my = [cd.low.loc[d] * 0.995 if dr > 0 else cd.high.loc[d] * 1.005 for d, _, dr in marks]
                spots = [{"pos": (xi, yi), "symbol": "t1" if dr > 0 else "t", "size": 10,
                          "brush": pg.mkBrush(UP if dr > 0 else DOWN), "pen": None, "data": n}
                         for xi, yi, (_, n, dr) in zip(mx, my, marks)]
                sp = pg.ScatterPlotItem(spots=spots, hoverable=True, tip=lambda x, y, data: data)
                self.price_plot.addItem(sp)
                self.price_plot.plot([], [], pen=None, symbol="t1", symbolBrush=UP, symbolPen=None, name="Pola candle")

        # ---- prediksi lalu
        if self.cb_past.isChecked() and not s.past.empty:
            past = s.past
            if index_mode:
                tgt = pd.DatetimeIndex(pd.to_datetime(past.target_date))
                pos = cd.index.get_indexer(tgt)
                past = past[pos >= 0]
                px = pos[pos >= 0].astype(float)
            else:
                px = _ts(pd.to_datetime(past.target_date))
            py = (past.base_price * np.exp(past.q50)).values
            brushes = [pg.mkBrush(UP if h else DOWN) for h in past.direction_hit]
            if len(past):
                self.price_plot.addItem(pg.ScatterPlotItem(px, py, size=6, brush=brushes, pen=pg.mkPen(BG, width=0.5)))
                self.price_plot.plot([], [], pen=None, symbol="o", symbolBrush=UP, symbolPen=None,
                                     name="Prediksi lalu (hijau = benar)")

        # ---- jalur prediksi
        x_end = x[-1]
        if tf.path_key and not s.path.empty:
            path = s.path
            base = path.attrs["base_price"]
            if index_mode:
                # Posisi batang dasar prediksi; jalur digambar per batang setelahnya.
                bi = cd.index.get_indexer([path.attrs["base_date"]])[0]
                bx = float(bi if bi >= 0 else len(cd) - 1)
                px = np.r_[bx, bx + path.h.to_numpy(dtype=float)]
            else:
                bx = _ts([path.attrs["base_date"]])[0]
                px = np.r_[bx, _ts(path.date)]
            lo = self.price_plot.plot(px, np.r_[base, path.lo.values], pen=pg.mkPen(ACCENT + "88", width=1))
            hi = self.price_plot.plot(px, np.r_[base, path.hi.values], pen=pg.mkPen(ACCENT + "88", width=1))
            self.price_plot.addItem(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(139, 92, 246, 45)))
            self.price_plot.plot(px, np.r_[base, path.mid.values], pen=pg.mkPen(ACCENT, width=2.2),
                                 symbol="o", symbolSize=5, symbolBrush=ACCENT, symbolPen=None,
                                 name=f"Perkiraan {len(path)} titik + rentang 80%")
            end = path.iloc[-1]
            label = pg.TextItem(html=f"<div style='background:{ACCENT};color:white;padding:2px 6px;"
                                     f"border-radius:4px'>{money(end['mid'], a.currency)}</div>", anchor=(0, 0.5))
            label.setPos(px[-1], end["mid"])
            self.price_plot.addItem(label)
            x_end = max(x_end, px[-1])

        # ---- volume
        has_vol = bool(cd.volume.notna().any() and (cd.volume.fillna(0) > 0).any())
        self.vol_plot.setVisible(self.cb_volume.isChecked() and has_vol)
        if self.cb_volume.isChecked() and has_vol:
            up = (cd.close >= cd.open).values
            brushes = [pg.mkBrush((UP if u else DOWN) + "aa") for u in up]
            self.vol_plot.addItem(pg.BarGraphItem(x=x, height=cd.volume.fillna(0).values, width=width,
                                                  brushes=brushes, pens=[pg.mkPen(None)] * len(x)))

        # ---- RSI
        self.rsi_plot.setVisible(self.cb_rsi.isChecked())
        if self.cb_rsi.isChecked():
            self.rsi_plot.addItem(pg.LinearRegionItem([30, 70], orientation="horizontal", movable=False,
                                                      brush=pg.mkBrush(139, 92, 246, 20), pen=pg.mkPen(None)))
            for lvl, txt, anchor in ((70, "jenuh beli", (1, 1)), (30, "jenuh jual", (1, 0))):
                self.rsi_plot.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen(FAINT, style=Qt.DashLine),
                                                      label=txt, labelOpts={"position": 0.98, "color": FAINT,
                                                                            "anchors": [anchor, anchor]}))
            self.rsi_plot.plot(x, cd.rsi.values, pen=pg.mkPen(ACCENT, width=1.5), connect="finite")
            self.rsi_plot.setYRange(0, 100)

        # ---- rentang awal: N batang terakhir + area perkiraan
        x0 = x[max(0, len(x) - tf.n_bars)]
        self.price_plot.setXRange(x0, x_end + (x_end - x0) * 0.10, padding=0)
        vis = cd.iloc[max(0, len(cd) - tf.n_bars):]
        ys = [vis.low.min(), vis.high.max()]
        if tf.path_key and not s.path.empty:
            ys += [s.path.lo.min(), s.path.hi.max()]
        self.price_plot.setYRange(min(ys), max(ys), padding=0.06)
        if has_vol:
            self.vol_plot.setYRange(0, float(vis.volume.max() or 1), padding=0.05)
        self._show_hover(len(cd) - 1)

        # ---- gambar pengguna (disimpan per aset)
        if index_mode:
            to_x, from_x = index_mapping(_ts(cd.index))
        else:
            to_x = from_x = None
        self.drawings.set_context(asset, to_x, from_x, fmt=lambda v: money(v, a.currency))
        self.drawings.render()

    def _on_mouse(self, pos):
        if self._cd.empty or not self.price_plot.sceneBoundingRect().contains(pos):
            return
        xv = self.price_plot.plotItem.vb.mapSceneToView(pos).x()
        idx = int(np.argmin(np.abs(self._x - xv)))
        self._show_hover(idx)

    def _show_hover(self, i: int):
        if self._cd.empty:
            return
        r = self._cd.iloc[i]
        cur = ASSET_BY_KEY[self.current_asset()].currency
        intraday = TIMEFRAMES[self.timeframe].source == "intraday"
        when = self._cd.index[i].strftime("%d %b %Y %H:%M" if intraday else "%d %b %Y")
        color = UP if r.close >= r.open else DOWN
        txt = (f"<span style='color:{TEXT}'><b>{when}</b></span> &nbsp; O {money(r.open, cur)} &nbsp; "
               f"H {money(r.high, cur)} &nbsp; L {money(r.low, cur)} &nbsp; "
               f"<span style='color:{color}'><b>C {money(r.close, cur)}</b></span>")
        if not pd.isna(r.volume) and r.volume > 0:
            txt += f" &nbsp; Vol {r.volume:,.0f}".replace(",", ".")
        if not pd.isna(r.rsi):
            txt += f" &nbsp; RSI {r.rsi:.0f}"
        self.hover.setText(txt)

    # ------------------------------------------------------------ panel kanan
    def render_side(self, s, tf):
        for c in (self.card_trend, self.card_pred, self.card_signal, self.card_rel, self.card_ta):
            c.clear()

        # Arah tren
        arrow = {2: "▲", 1: "▲", 0: "◆", -1: "▼", -2: "▼"}[s.trend_level]
        self.card_trend.pill(f"{arrow}  {s.trend}", TREND_COLORS[s.trend_level])
        if "path" in s.lines:
            self.card_trend.label(s.lines["path"])
        self.card_trend.label(s.lines.get("trend_why", ""), "faint")
        if s.trend_details:
            exp = QToolButton()
            exp.setObjectName("expander")
            exp.setCheckable(True)
            exp.setChecked(self._trend_open)
            exp.setCursor(Qt.PointingHandCursor)
            details = QWidget()
            dv = QVBoxLayout(details)
            dv.setContentsMargins(0, 4, 0, 0)
            dv.setSpacing(6)
            g = QGridLayout()
            g.setHorizontalSpacing(10)
            g.setVerticalSpacing(5)
            n = len(s.trend_details)
            for i, (text, pts) in enumerate(s.trend_details):
                summary_row = i >= n - 2  # total teknikal dan arah tren akhir
                t = QLabel(text)
                t.setWordWrap(True)
                t.setStyleSheet(f"color: {TEXT if summary_row else MUTED};" + (" font-weight: 600;" if summary_row else ""))
                v = QLabel(pts)
                color = UP if pts.startswith("+") else DOWN if pts.startswith("-") else NEUTRAL
                v.setStyleSheet(f"color: {color}; font-weight: 700;")
                v.setAlignment(Qt.AlignRight | Qt.AlignTop)
                if i == n - 2:
                    line = QFrame()
                    line.setFixedHeight(1)
                    line.setStyleSheet(f"background: {BORDER};")
                    g.addWidget(line, 2 * i, 0, 1, 2)
                g.addWidget(t, 2 * i + 1, 0)
                g.addWidget(v, 2 * i + 1, 1)
            g.setColumnStretch(0, 1)
            dv.addLayout(g)
            rules = QLabel(TREND_RULES)
            rules.setWordWrap(True)
            rules.setObjectName("faint")
            dv.addWidget(rules)
            details.setVisible(self._trend_open)

            def _toggle(on, b=exp, w=details):
                self._trend_open = on
                self.settings.setValue("trend_details_open", on)
                b.setText(("▾" if on else "▸") + "  Rincian skor")
                w.setVisible(on)

            _toggle(self._trend_open)
            exp.toggled.connect(_toggle)
            self.card_trend.body.addWidget(exp)
            self.card_trend.body.addWidget(details)

        # Perkiraan utama
        if s.has_prediction:
            self.card_pred.label(s.name, "muted")
            self.card_pred.label(s.lines["headline"], "faint")
            self.card_pred.label(money(s.median, s.currency), "big", wrap=False)
            color = UP if s.change_pct > 0 else DOWN if s.change_pct < 0 else NEUTRAL
            self.card_pred.label(s.lines["change"], color=color, bold=True)
            self.card_pred.rows([
                ("Rentang 80%", f"{money(s.low, s.currency)} – {money(s.high, s.currency)}"),
                ("Peluang naik", pct(s.p_up)),
            ])
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(round(s.p_up * 100)))
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background: {UP if s.p_up >= 0.5 else DOWN}; }}")
            self.card_pred.body.addWidget(bar)
            self.card_pred.label(s.lines["p_up"].split(": ", 1)[-1].capitalize(), "muted")
            self.card_pred.label(s.lines["as_of"], "faint")
        else:
            self.card_pred.label(s.name, "muted")
            self.card_pred.label(s.lines["signal"])

        # Saran
        self.card_signal.pill(s.signal, SIGNAL_COLORS[s.signal])
        self.card_signal.label(s.lines["signal"], "muted")

        # Keandalan
        dots = "●" * s.reliability_level + "○" * (3 - s.reliability_level)
        self.card_rel.label(f"{dots}   {s.reliability}", color=RELIABILITY_COLORS[s.reliability_level],
                            size=16, bold=True)
        rows = []
        acc_lines = s.lines.get("accuracy", "").split("\n")
        bt_line = next((l for l in acc_lines if l.startswith("Diuji")), None)
        if bt_line:
            rows.append(("Uji data lama", bt_line.split("benar ", 1)[-1].replace(", tebakan sederhana", " vs tebakan").rstrip(".")))
        if not s.past.empty:
            hits = int(s.past.direction_hit.sum())
            rows.append(("Terbukti nyata", f"{hits}/{len(s.past)} benar ({pct(hits / len(s.past))})"))
            rows.append(("Rata-rata meleset", f"{dec(s.past.abs_pct_error.mean() * 100)}%"))
        if rows:
            self.card_rel.rows(rows)
        if s.past.empty:
            self.card_rel.label("Akurasi nyata muncul setelah aplikasi dipakai beberapa hari.", "faint")
        self.card_rel.label("Patokan: arah benar 52–56% sudah tergolong bagus untuk harga pasar.", "faint")

        # Sinyal teknikal
        if s.ta and s.ta.signals:
            for sig in s.ta.signals:
                self.card_ta.bullet(sig.text, sig.direction)
        else:
            self.card_ta.label("Tidak ada sinyal teknikal yang menonjol saat ini.", "muted")
        if s.ta and (s.ta.supports or s.ta.resistances):
            self.card_ta.rows([
                ("Support", ", ".join(money(v, s.currency) for v in s.ta.supports) or "–"),
                ("Resistance", ", ".join(money(v, s.currency) for v in s.ta.resistances) or "–"),
            ])

    def closeEvent(self, event):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self.store.close()
        super().closeEvent(event)


def make_app() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("JustPredict")
    app.setStyle("Fusion")  # gaya seragam di macOS agar tema gelap terbaca jelas
    app.setPalette(dark_palette())
    return app


def show_error(title: str, details: str) -> None:
    from PySide6.QtWidgets import QMessageBox

    from ..applog import log_file
    box = QMessageBox()
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle("JustPredict")
    box.setText(title)
    box.setInformativeText(f"Log tersimpan di:\n{log_file()}")
    box.setDetailedText(details)
    box.exec()


def _excepthook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    logging.getLogger("justpredict.gui").error("Error tak tertangani:\n%s", text)
    import os
    if os.environ.get("JUSTPREDICT_NO_DIALOG"):
        return
    try:
        show_error("Terjadi kesalahan. Aplikasi tetap berjalan, tapi tampilan mungkin tidak lengkap.", text)
    except Exception:
        pass


def main() -> int:
    log = logging.getLogger("justpredict.gui")
    app = make_app()
    sys.excepthook = _excepthook  # error di dalam slot Qt tidak lagi menutup aplikasi diam-diam
    try:
        win = MainWindow()
    except Exception:
        import os
        text = traceback.format_exc()
        log.error("Gagal membuka jendela:\n%s", text)
        if not os.environ.get("JUSTPREDICT_NO_DIALOG"):
            show_error("Aplikasi gagal dibuka.", text)
        return 1
    win.show()
    win.raise_()
    win.activateWindow()
    return app.exec()
