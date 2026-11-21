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
  --paths . \
  packaging/launcher.py

./dist/JustPredict.app/Contents/MacOS/JustPredict selftest
(cd dist && ditto -c -k --keepParent JustPredict.app JustPredict-macos-arm64.zip)
echo "Selesai: dist/JustPredict.app dan dist/JustPredict-macos-arm64.zip"
