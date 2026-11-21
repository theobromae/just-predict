"""Uji asap GUI tanpa layar (offscreen)."""
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_chart_window_all_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("JUSTPREDICT_HOME", str(tmp_path))
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
    win.close()
    win.deleteLater()
    qapp.processEvents()
    del win
