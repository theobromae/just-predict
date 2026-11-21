# JustPredict

Aplikasi lokal (macOS) untuk memprediksi arah dan rentang harga minyak WTI, IHSG, Rupiah/USD,
emas per gram (Rupiah), saham bluechip Indonesia (TLKM, ADRO, BRPT), dan Bitcoin, untuk horizon
harian (1 hari bursa), mingguan (5) dan bulanan (21).

> Prediksi bersifat probabilistik dan bisa salah. Bukan saran investasi.

## Cara kerja

Setiap kali aplikasi dibuka:

1. **Update data**: hanya data baru yang diambil dari Yahoo Finance (gratis, tanpa API key).
   Emas Rp/gram = emas USD/oz × kurs USD/IDR ÷ 31,1035.
2. **Evaluasi**: prediksi lama yang tanggal targetnya sudah lewat dibandingkan dengan harga aktual.
   Galat, ketepatan arah, dan apakah harga aktual masuk rentang disimpan.
3. **Latih ulang + auto adjust** (`justpredict/adaptive.py`):
   - data aktual terbaru langsung jadi data latih (bobot sampel lebih besar untuk data baru);
   - **bobot ansambel adaptif**: model yang galatnya kecil belakangan mendapat bobot lebih besar;
   - **koreksi galat**: kalau prediksi konsisten meleset ke satu arah (mis. prediksi 20rb, aktual 25rb),
     median berikutnya digeser sebesar rata-rata galat terbaru; rentang diperlebar/dipersempit
     sampai sekitar 80% harga aktual jatuh di dalamnya.
4. **Prediksi**: peluang naik, median, dan rentang 80% (kuantil 10–90%) untuk tiap horizon.
5. **Sinyal BELI/JUAL/TAHAN** hanya diberikan kalau di backtest model benar arahnya ≥52% *dan*
   mengalahkan tebakan naif ("harga tetap"). Selain itu sinyal ditahan dan ditandai "Belum terbukti".

Semua data disimpan di `~/Library/Application Support/JustPredict/justpredict.sqlite`.

## Struktur

| File | Isi |
|---|---|
| `justpredict/config.py` | daftar aset, ticker, horizon |
| `justpredict/data/prices.py` | pengambilan harga inkremental |
| `justpredict/features.py` | fitur teknikal (return, volatilitas, MA, MACD, RSI, dll.) |
| `justpredict/models/` | model plug-in: `naive` (baseline) dan `lightgbm` (kuantil + arah) |
| `justpredict/adaptive.py` | bobot ansambel dan koreksi galat |
| `justpredict/engine.py` | siklus update → evaluasi → latih → prediksi |
| `justpredict/gui/app.py` | aplikasi desktop PySide6 |

Menambah model baru: buat kelas turunan `Forecaster` (lihat `models/base.py`) lalu daftarkan di
`models/registry.py`. Ansambel otomatis memberi bobot sesuai akurasinya.

Menambah saham: tambahkan baris `Asset(...)` di `config.py` dengan ticker `.JK`.

## Menjalankan dari kode (Mac)

```bash
brew install libomp            # dibutuhkan LightGBM
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m justpredict          # buka aplikasi
python -m justpredict run      # mode teks tanpa GUI
python -m pytest -q            # tes
```

## Membuat JustPredict.app

- **Otomatis**: setiap push ke `main` atau pull request, GitHub Actions (`build-macos`) membangun
  `JustPredict-macos-arm64.zip` untuk Apple Silicon. Unduh dari tab **Actions** → run terbaru →
  **Artifacts**. Push tag `v0.1.0` untuk melampirkannya ke halaman Release.
- **Manual di Mac**: `bash packaging/build_macos.sh`, hasilnya di `dist/JustPredict.app`.

Aplikasi belum ditandatangani Apple, jadi saat pertama dibuka macOS akan menolaknya. Klik kanan
`JustPredict.app` → **Open** → **Open**, atau jalankan sekali:

```bash
xattr -dr com.apple.quarantine /Applications/JustPredict.app
```

## Rencana berikutnya

- v0.2: fundamental (laporan keuangan, makro), sentimen berita (RSS + IndoBERT), Reddit,
  Fear & Greed Index; modul opsional X via AIsa dan Stockbit dengan batas biaya harian.
- v0.3: model hasil riset jurnal (XGBoost, ansambel lanjutan) dan kalibrasi peluang.
