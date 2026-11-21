"""Alat gambar di grafik ala TradingView: garis tren, garis horizontal, kotak, Fibonacci, catatan.

Setiap gambar disimpan di SQLite per aset dengan koordinat (detik epoch, harga), sehingga tetap ada
setelah aplikasi ditutup dan muncul di tempat yang sama di semua tampilan waktu (harian, per jam, …).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtWidgets import QInputDialog

# (kunci, ikon, penjelasan)
TOOLS: list[tuple[str, str, str]] = [
    ("select", "↖", "Pilih dan geser gambar"),
    ("trend", "╱", "Garis tren: klik titik awal lalu titik akhir"),
    ("hline", "―", "Garis horizontal: klik di level harga"),
    ("rect", "▭", "Kotak area: klik dua sudut yang berseberangan"),
    ("fib", "Fib", "Fibonacci retracement: klik titik awal lalu titik akhir gerakan harga"),
    ("text", "T", "Catatan teks: klik lokasi lalu ketik catatannya"),
    ("erase", "⌫", "Penghapus: klik gambar yang ingin dihapus"),
]
CLICKS = {"trend": 2, "hline": 1, "rect": 2, "fib": 2, "text": 1}
FIB_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

COLOR = "#38bdf8"
FIB_COLOR = "#f59e0b"
TEXT_COLOR = "#e5e7eb"


def _identity(v: float) -> float:
    return float(v)


class DrawingLayer(QObject):
    """Mengelola gambar di satu PlotWidget. Panggil `render()` setiap kali plot dibersihkan."""

    toolChanged = Signal(str)
    changed = Signal()

    def __init__(self, plot: pg.PlotWidget, store, ask_text: Callable[[], str | None] | None = None):
        super().__init__(plot)
        self.plot = plot
        self.vb = plot.plotItem.vb
        self.store = store
        self.asset: str | None = None
        self.tool = "select"
        self.pending: list[tuple[float, float]] = []
        self.preview: pg.PlotDataItem | None = None
        self.items: dict[int, list] = {}
        self.to_x: Callable[[float], float] = _identity    # epoch -> posisi x di grafik
        self.from_x: Callable[[float], float] = _identity  # posisi x -> epoch
        self.fmt: Callable[[float], str] = lambda v: f"{v:,.2f}"
        self.ask_text = ask_text or self._ask_text
        plot.scene().sigMouseClicked.connect(self._on_click)
        plot.scene().sigMouseMoved.connect(self._on_move)

    # ------------------------------------------------------------ konteks
    def set_context(self, asset: str | None, to_x=None, from_x=None, fmt=None):
        self.asset = asset
        self.to_x = to_x or _identity
        self.from_x = from_x or _identity
        if fmt:
            self.fmt = fmt

    def render(self):
        """Gambar ulang semua gambar aset ini (plot sudah dibersihkan oleh pemanggil)."""
        self.items = {}
        self.preview = None
        self.pending = []
        if not self.asset:
            return
        for d in self.store.drawings(self.asset):
            try:
                self._build(d)
            except Exception:  # satu gambar rusak tidak boleh menggagalkan grafik
                import logging
                logging.getLogger(__name__).exception("Gagal menggambar %s", d)

    def set_tool(self, tool: str):
        self.cancel()
        self.tool = tool
        self.vb.setMouseEnabled(x=True, y=True)
        self.plot.setCursor(Qt.ArrowCursor if tool == "select" else Qt.CrossCursor)
        self.toolChanged.emit(tool)

    def cancel(self):
        self.pending = []
        if self.preview is not None:
            self.plot.removeItem(self.preview)
            self.preview = None

    def clear_all(self):
        if not self.asset:
            return
        self.store.clear_drawings(self.asset)
        for its in self.items.values():
            for it in its:
                self.plot.removeItem(it)
        self.items = {}
        self.changed.emit()

    def count(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------ simpan / hapus
    def add(self, kind: str, view_points: list[tuple[float, float]], text: str | None = None) -> int:
        """Simpan gambar baru (koordinat grafik) dan tampilkan."""
        pts = [(self.from_x(x), y) for x, y in view_points]
        did = self.store.save_drawing(self.asset, kind, pts, text)
        self._build({"id": did, "kind": kind, "points": [list(p) for p in pts], "text": text})
        self.changed.emit()
        return did

    def _update(self, d: dict, view_points: list[tuple[float, float]]):
        d["points"] = [[self.from_x(x), y] for x, y in view_points]
        self.store.save_drawing(self.asset, d["kind"], d["points"], d.get("text"), drawing_id=d["id"])
        self.changed.emit()

    def delete(self, did: int):
        for it in self.items.pop(did, []):
            self.plot.removeItem(it)
        self.store.delete_drawing(did)
        self.changed.emit()

    # ------------------------------------------------------------ membangun item grafik
    def _build(self, d: dict):
        kind = d["kind"]
        pts = [(self.to_x(e), float(y)) for e, y in d["points"]]
        pen = pg.mkPen(COLOR, width=2)
        hover = pg.mkPen("#ffffff", width=2)
        its: list = []
        if kind == "trend":
            roi = pg.LineSegmentROI(pts, pen=pen, hoverPen=hover)
            roi.sigRegionChangeFinished.connect(lambda r, d=d: self._update(d, self._roi_points(r)))
            its = [roi]
        elif kind == "hline":
            line = pg.InfiniteLine(pts[0][1], angle=0, movable=True, pen=pen, hoverPen=hover,
                                   label=self.fmt(pts[0][1]),
                                   labelOpts={"position": 0.97, "color": COLOR, "fill": "#0f1115cc",
                                              "anchors": [(1, 1), (1, 1)]})
            line.sigPositionChangeFinished.connect(
                lambda ln, d=d: (ln.label.setFormat(self.fmt(ln.value())),
                                 self._update(d, [(self.to_x(d["points"][0][0]), ln.value())])))
            its = [line]
        elif kind == "rect":
            (x1, y1), (x2, y2) = pts
            roi = pg.RectROI((min(x1, x2), min(y1, y2)), (abs(x2 - x1) or 1e-9, abs(y2 - y1) or 1e-9),
                             pen=pen, hoverPen=hover)
            roi.sigRegionChangeFinished.connect(lambda r, d=d: self._update(d, self._rect_points(r)))
            its = [roi]
        elif kind == "fib":
            roi = pg.LineSegmentROI(pts, pen=pg.mkPen(FIB_COLOR, width=1, style=Qt.DashLine), hoverPen=hover)
            levels = [pg.PlotDataItem(pen=pg.mkPen(FIB_COLOR, width=1)) for _ in FIB_LEVELS]
            labels = [pg.TextItem(color=FIB_COLOR, anchor=(0, 1)) for _ in FIB_LEVELS]
            for it in levels + labels:
                self.plot.addItem(it)
            self._layout_fib(pts, levels, labels)
            roi.sigRegionChanged.connect(lambda r, lv=levels, lb=labels: self._layout_fib(self._roi_points(r), lv, lb))
            roi.sigRegionChangeFinished.connect(lambda r, d=d: self._update(d, self._roi_points(r)))
            its = [roi, *levels, *labels]
        elif kind == "text":
            t = pg.TargetItem(pos=pts[0], size=9, symbol="o", movable=True, pen=pg.mkPen(TEXT_COLOR),
                              brush=pg.mkBrush(COLOR), label=d.get("text") or "",
                              labelOpts={"color": TEXT_COLOR, "offset": (10, -10), "fill": "#171a21dd"})
            t.sigPositionChangeFinished.connect(lambda ti, d=d: self._update(d, [(ti.pos().x(), ti.pos().y())]))
            its = [t]
        else:
            return
        for it in its:
            if it.scene() is None:
                self.plot.addItem(it)
        self.items[d["id"]] = its

    def _layout_fib(self, pts, levels, labels):
        (x1, y1), (x2, y2) = pts
        left, right = min(x1, x2), max(x1, x2)
        for lvl, line, lab in zip(FIB_LEVELS, levels, labels):
            y = y2 + (y1 - y2) * lvl  # 0% di titik akhir, 100% di titik awal (seperti TradingView)
            line.setData([left, right], [y, y])
            pct = f"{lvl * 100:.1f}".removesuffix(".0").replace(".", ",")
            lab.setText(f"{pct}%  {self.fmt(y)}")
            lab.setPos(left, y)

    @staticmethod
    def _roi_points(roi) -> list[tuple[float, float]]:
        out = []
        for h in roi.getHandles():
            p = roi.mapToParent(h.pos())
            out.append((p.x(), p.y()))
        return out

    @staticmethod
    def _rect_points(roi) -> list[tuple[float, float]]:
        p, s = roi.pos(), roi.size()
        return [(p.x(), p.y()), (p.x() + s.x(), p.y() + s.y())]

    # ------------------------------------------------------------ mouse
    def _view_pos(self, scene_pos) -> tuple[float, float] | None:
        if not self.vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = self.vb.mapSceneToView(scene_pos)
        return p.x(), p.y()

    def _on_click(self, ev):
        if self.tool == "select" or ev.button() != Qt.LeftButton or ev.double():
            return
        pos = self._view_pos(ev.scenePos())
        if pos is None or not self.asset:
            return
        if self.tool == "erase":
            hits = set(self.plot.scene().items(ev.scenePos()))
            for did, its in list(self.items.items()):
                if any(it in hits or any(c in hits for c in it.childItems()) for it in its):
                    self.delete(did)
                    ev.accept()
                    return
            return
        self.click_at(*pos)
        ev.accept()

    def click_at(self, x: float, y: float):
        """Tambahkan satu klik untuk alat yang aktif (dipisah agar bisa dites tanpa mouse)."""
        self.pending.append((x, y))
        if len(self.pending) < CLICKS.get(self.tool, 1):
            return
        pts, kind = self.pending, self.tool
        self.cancel()
        text = None
        if kind == "text":
            text = self.ask_text()
            if not text:
                self.set_tool("select")
                return
        self.add(kind, pts, text)
        self.set_tool("select")

    def _on_move(self, scene_pos):
        if not self.pending or self.tool not in ("trend", "rect", "fib"):
            return
        pos = self._view_pos(scene_pos)
        if pos is None:
            return
        (x1, y1), (x2, y2) = self.pending[0], pos
        if self.tool == "rect":
            xs, ys = [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1]
        else:
            xs, ys = [x1, x2], [y1, y2]
        if self.preview is None:
            self.preview = pg.PlotDataItem(pen=pg.mkPen(COLOR, width=1, style=Qt.DashLine))
            self.plot.addItem(self.preview)
        self.preview.setData(xs, ys)

    def _ask_text(self) -> str | None:
        text, ok = QInputDialog.getText(self.plot, "Catatan", "Tulis catatan untuk titik ini:")
        return text.strip() if ok and text.strip() else None


def index_mapping(epochs: np.ndarray):
    """Fungsi konversi epoch <-> nomor batang untuk grafik mode indeks (per jam).

    Di luar rentang data, posisi diperkirakan dengan jarak batang biasa (median).
    """
    ep = np.asarray(epochs, dtype=float)
    n = len(ep)
    idx = np.arange(n, dtype=float)
    step = float(np.median(np.diff(ep))) if n > 1 else 3600.0
    step = step or 3600.0

    def to_x(e: float) -> float:
        e = float(e)
        if n == 0:
            return 0.0
        if e > ep[-1]:
            return (n - 1) + (e - ep[-1]) / step
        if e < ep[0]:
            return (e - ep[0]) / step
        return float(np.interp(e, ep, idx))

    def from_x(x: float) -> float:
        x = float(x)
        if n == 0:
            return 0.0
        if x > n - 1:
            return float(ep[-1] + (x - (n - 1)) * step)
        if x < 0:
            return float(ep[0] + x * step)
        return float(np.interp(x, idx, ep))

    return to_x, from_x


__all__ = ["CLICKS", "DrawingLayer", "TOOLS", "index_mapping", "QPointF"]
