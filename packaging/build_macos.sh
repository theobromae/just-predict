#!/usr/bin/env bash
# Build JustPredict.app di macOS (Apple Silicon). Jalankan dari folder repo:
#   bash packaging/build_macos.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v brew >/dev/null; then
  echo "Homebrew dibutuhkan untuk libomp (dipakai LightGBM): https://brew.sh" >&2
  exit 1
fi
brew list libomp >/dev/null 2>&1 || brew install libomp

python3 -m venv .venv-build
source .venv-build/bin/activate
pip install --upgrade pip
pip install -r requirements.txt pyinstaller

pyinstaller --noconfirm --clean --windowed \
  --name JustPredict \
  --osx-bundle-identifier com.theobromae.justpredict \
  --collect-all lightgbm \
  --collect-all yfinance \
  --collect-all curl_cffi \
  --collect-all pyqtgraph \
  --exclude-module sklearn \
  --exclude-module matplotlib \
  --exclude-module IPython \
  --exclude-module tkinter \
  --exclude-module PyQt5 \
  --exclude-module PyQt6 \
  --exclude-module OpenGL \
  --exclude-module pyqtgraph.opengl \
  --exclude-module pyqtgraph.examples \
  --exclude-module pyqtgraph.jupyter \
  --exclude-module PySide6.QtWebEngineCore \
  --exclude-module PySide6.QtWebEngineWidgets \
  --exclude-module PySide6.QtWebEngineQuick \
  --exclude-module PySide6.QtWebChannel \
  --exclude-module PySide6.QtQml \
  --exclude-module PySide6.QtQuick \
  --exclude-module PySide6.QtQuickWidgets \
  --exclude-module PySide6.Qt3DCore \
  --exclude-module PySide6.QtMultimedia \
  --exclude-module PySide6.QtPdf \
  --exclude-module PySide6.QtCharts \
  --exclude-module PySide6.QtDataVisualization \
  --exclude-module PySide6.QtDesigner \
  --paths . \
  packaging/launcher.py

./dist/JustPredict.app/Contents/MacOS/JustPredict selftest
# Buka jendela aplikasi hasil build dengan data sintetis, coba semua tampilan, lalu tutup.
# Batas waktu 15 menit (macOS tidak punya perintah `timeout`).
perl -e 'alarm 900; exec @ARGV' ./dist/JustPredict.app/Contents/MacOS/JustPredict gui-smoke
# Laporan ukuran (tampil sebagai catatan di GitHub Actions).
APP_MB=$(du -sm dist/JustPredict.app | cut -f1)
echo "::notice title=Ukuran app::JustPredict.app ${APP_MB} MB"
du -sm dist/JustPredict.app/Contents/Frameworks/* 2>/dev/null | sort -rn | head -12 | \
  while read -r mb path; do echo "::notice title=Isi terbesar::${mb} MB $(basename "$path")"; done
(cd dist && ditto -c -k --keepParent JustPredict.app JustPredict-macos-arm64.zip)
echo "Selesai: dist/JustPredict.app dan dist/JustPredict-macos-arm64.zip"
