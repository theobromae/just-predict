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

## Tampilan

Saat dibuka, aplikasi langsung menampilkan **grafik** dari data tersimpan, lalu memperbarui data dan
prediksi di belakang. Satu layar berisi:

- tema gelap, desain flat;
- daftar aset di kiri, dengan harga terakhir dan perkiraan perubahan;
- pilihan **1 Jam / 4 Jam / Harian / Mingguan / Bulanan**. Data per jam dari Yahoo (±2 tahun terakhir),
  4 jam disusun dari data per jam;
- **jalur prediksi**: 8 titik (1 jam), 10 titik (4 jam), 14 titik (harian), 8 titik (mingguan),
  6 titik (bulanan) dengan rentang 80% yang melebar makin jauh ke depan. Atur di `PATH_POINTS` (`engine.py`).
  Model per jam dilatih terpisah (`INTRADAY` di `engine.py`) dan juga dievaluasi serta dikoreksi dari galatnya;
- pilihan **Garis / Candle**, menu **Indikator ▾**: Volume, RSI, MA 20/50, Support/Resistance, dan titik
  Prediksi lalu (hijau = arah benar, merah = salah), plus penanda **pola candle**;
- **alat gambar** di kiri grafik: garis tren, garis horizontal, kotak, Fibonacci retracement, catatan teks,
  dan penghapus. Gambar disimpan per aset di database (tabel `drawings`), tetap ada setelah aplikasi ditutup,
  dan ikut tampil di semua tampilan waktu. Tekan Esc untuk membatalkan;
- panel kanan berbahasa awam: **arah tren** (Bullish/Bearish/Sideways, gabungan model + analisis teknikal)
  dengan dropdown **Rincian skor**, perkiraan harga dan rentangnya, saran, seberapa bisa dipercaya, dan
  daftar sinyal teknikal;
- ukuran jendela mengikuti layar, dan batas antara daftar aset, grafik, dan panel kanan bisa digeser.

Analisis teknikal (`technicals.py`) juga menjadi fitur model: jarak ke support/resistance, breakout/breakdown,
pola candle (hammer, engulfing, shooting star, doji), kemiringan dan divergensi RSI, golden/death cross,
Bollinger %B, ATR, kemiringan tren, dan rasio volume.

### Fundamental dan sentimen (v0.2)

- **Saham** (`data/fundamentals.py`): laporan keuangan dari Yahoo Finance: laba dan pendapatan 12 bulan,
  ekuitas, utang, jumlah saham. Dari situ dihitung PER/PBV dibanding rata-rata 5 tahun (Murah/Wajar/Mahal),
  pertumbuhan laba, ROE, dan DER. Laporan berdenominasi USD (mis. ADRO) dikonversi ke Rupiah.
- **Makro**: indeks dolar (DXY) dan imbal hasil obligasi AS 10 tahun (Yahoo), suku bunga The Fed,
  suku bunga Indonesia, dan stok minyak mentah AS/EIA (FRED, CSV publik). **Bitcoin**: Fear & Greed
  (alternative.me) dan hash rate (blockchain.com).
- Semua angka dicatat sejak tanggal **dianggap terbit** (kuartal +45 hari, tahunan +90 hari, data bulanan
  +45 hari), jadi model tidak memakai angka sebelum publik bisa melihatnya. Penanda **L** di grafik saham
  menunjukkan perkiraan tanggal terbit itu.
- Fundamental **tidak mengubah angka harga** (median dan rentang tetap dari model teknikal). Fundamental
  hanya **menggeser peluang naik**, paling banyak 10 poin (`MAX_FUND_SHIFT`), dan hanya kalau di uji data lama
  peluangnya jadi lebih tepat (log loss `lightgbm_fund` < `lightgbm_base`). Geseran tampil di kartu Perkiraan.
- **Sentimen berita** (`data/news.py`): judul dari Google News RSS, dinilai dengan kamus kata keuangan
  Indonesia/Inggris plus negasi. Karena RSS hanya berisi beberapa minggu, sentimen dipakai di skor Arah
  Tren, belum sebagai fitur model.
- **Arah Tren** = 4 kelompok (model ML, teknikal, fundamental/makro, sentimen), masing-masing +1/0/−1.
  Rinciannya, termasuk judul berita yang bisa diklik, ada di dropdown **Rincian skor**.

### Gabungan prediksi multi-rentang dan tampilan tabel

- `reconcile.py` menyelaraskan prediksi semua rentang harian (1–126 hari bursa) ke satu arah dasar
  bersama (laju return μ·h dan sinyal s·√h), dengan bobot 1/galat² tiap rentang dari uji data lama.
  λ (campuran prediksi langsung vs arah bersama) dipilih di paruh pertama data uji, lalu dinilai di paruh
  kedua; penyelarasan hanya dipakai di rentang yang galatnya lebih kecil dan arahnya tidak lebih buruk.
  Hasilnya disimpan sebagai model `recon` dan dievaluasi seperti prediksi lain.
- Kalau arah harian, mingguan, dan bulanan jelas berlawanan, saran menjadi TAHAN dan keandalan turun satu
  tingkat.
- Tombol **Grafik / Tabel** menukar layar tengah dengan tabel prediksi: tanggal, prediksi harga, rentang
  80%, perubahan, peluang naik, dan tingkat keyakinan (Tinggi/Sedang/Rendah), mengikuti pilihan
  1 Jam/4 Jam/Harian/Mingguan/Bulanan.

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
| `justpredict/technicals.py` | pola candle, support/resistance, sinyal teknikal, fitur TA |
| `justpredict/viewmodel.py` | data grafik (resample, indikator) dan teks penjelasan |
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
