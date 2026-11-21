"""Uji asap GUI tanpa layar (offscreen)."""
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_renders(tmp_path, monkeypatch):
    monkeypatch.setenv("JUSTPREDICT_HOME", str(tmp_path))
    from tests.test_engine import FakeSource
    from justpredict.config import db_path
    from justpredict.db import Store
    from justpredict.engine import run_cycle

    s = Store(db_path())
    run_cycle(s, FakeSource(days=700), assets=["WTI", "TLKM"])
    s.close()

    from PySide6.QtWidgets import QApplication
    from justpredict.gui import app as gui

    monkeypatch.setattr(gui.MainWindow, "start_cycle", lambda self, fetch: None)
    qapp = QApplication.instance() or QApplication([])
    win = gui.MainWindow()
    assert win.pred_table.rowCount() == 6
    assert win.acc_table.rowCount() == 6
    win.asset_combo.setCurrentIndex(win.asset_combo.findData("TLKM"))
    win.refresh_chart()
    win.grab().save(str(tmp_path / "shot.png"))
    win.close()
    win.deleteLater()
    qapp.processEvents()
    del win
