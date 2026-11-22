"""Uji asap GUI tanpa layar (offscreen)."""
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_chart_window_all_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("JUSTPREDICT_HOME", str(tmp_path))
    monkeypatch.setenv("JUSTPREDICT_NO_DIALOG", "1")
    from tests.test_engine import FakeSource
    from justpredict.config import db_path
    from justpredict.db import Store
    from justpredict.engine import run_cycle

    s = Store(db_path())
    src = FakeSource(days=700)
    run_cycle(s, src, assets=["WTI", "GOLD_IDR_G"])
    for _ in range(3):
        src.advance(1)
        run_cycle(s, src, assets=["WTI", "GOLD_IDR_G"])
    s.close()

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from justpredict.gui import app as gui

    QSettings.setDefaultFormat(QSettings.IniFormat)
    monkeypatch.setattr(gui.MainWindow, "start_cycle", lambda self, fetch: None)
    qapp = QApplication.instance() or QApplication([])
    win = gui.MainWindow()
    win.settings.clear()
    assert win.asset_list.count() == 8
    for asset in ("WTI", "GOLD_IDR_G", "BTC"):  # BTC: belum ada data
        for row in range(win.asset_list.count()):
            if win.asset_list.item(row).data(0x0100) == asset:
                win.asset_list.setCurrentRow(row)
        for tf_btn in win.tf_group.buttons():
            tf_btn.setChecked(True)
            for type_btn in win.type_group.buttons():
                type_btn.setChecked(True)
                win.refresh_chart()
    win.grab().save(str(tmp_path / "shot.png"))

    # Regresi: ganti tampilan dengan rincian skor terbuka tidak boleh memunculkan jendela baru.
    win._trend_open = True
    win.show()
    for tf_btn in win.tf_group.buttons():
        tf_btn.setChecked(True)
        qapp.processEvents()
    extra = [w for w in QApplication.topLevelWidgets() if w.isVisible() and w is not win]
    assert extra == [], extra

    # Tampilan tabel prediksi.
    win._check(win.view_group, "table")
    for tf in ("harian", "mingguan", "bulanan", "1jam"):
        win._check(win.tf_group, tf)
        win.refresh_chart()
        qapp.processEvents()
    win._check(win.tf_group, "harian")
    for row in range(win.asset_list.count()):
        if win.asset_list.item(row).data(0x0100) == "WTI":
            win.asset_list.setCurrentRow(row)
    win.refresh_chart()
    assert win.stack.currentIndex() == 1 and win.table.rowCount() == 14
    assert win.table.item(0, 5).text().strip() != ""
    win._check(win.view_group, "chart")
    assert win.stack.currentIndex() == 0

    # Alat gambar: garis tren di WTI harian tersimpan dan muncul lagi setelah jendela dibuka ulang.
    for row in range(win.asset_list.count()):
        if win.asset_list.item(row).data(0x0100) == "WTI":
            win.asset_list.setCurrentRow(row)
    win._check(win.tf_group, "harian")
    win.refresh_chart()
    x = win._x
    y = float(win._cd.close.iloc[-1])
    win.drawings.set_tool("trend")
    win.drawings.click_at(x[-30], y)
    win.drawings.click_at(x[-1], y * 1.02)
    win.drawings.set_tool("fib")
    win.drawings.click_at(x[-20], y)
    win.drawings.click_at(x[-5], y * 0.97)
    win.drawings.set_tool("hline")
    win.drawings.click_at(x[-10], y)
    assert win.drawings.tool == "select" and win.drawings.count() == 3
    win._check(win.tf_group, "1jam")  # gambar ikut tampil di mode per jam
    assert win.drawings.count() == 3
    win.close()
    win.deleteLater()
    qapp.processEvents()
    del win

    win = gui.MainWindow()
    assert len(Store(db_path()).drawings("WTI")) == 3
    assert win.current_asset() == "WTI" and win.drawings.count() == 3
    win._clear_drawings()
    assert win.drawings.count() == 0
    win.close()
    win.deleteLater()
    qapp.processEvents()
    del win


def test_gui_smoke_command(monkeypatch):
    from justpredict.gui.smoke import run
    assert run() == 0
