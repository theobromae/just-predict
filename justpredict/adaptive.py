"""Auto adjust: belajar dari galat prediksi sebelumnya.

Tiga mekanisme:
1. Latih ulang setiap kali aplikasi dibuka (harga aktual baru otomatis jadi data latih).
2. Bobot ansambel adaptif: model yang galatnya kecil belakangan ini mendapat bobot lebih besar.
3. Koreksi galat: kalau prediksi gabungan konsisten meleset ke satu arah (mis. selalu terlalu rendah),
   median digeser sebesar rata-rata galat terbaru; kalau rentang terlalu sempit/lebar, rentang diskalakan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIOR_STRENGTH = 20          # berapa evaluasi live "senilai" metrik backtest
BIAS_HALFLIFE = 10           # evaluasi terbaru lebih berbobot
BIAS_SHRINK = 10             # koreksi baru penuh setelah banyak evaluasi
TARGET_COVERAGE = 0.8        # rentang q10–q90 idealnya memuat 80% harga aktual


def member_weights(live: dict[str, pd.Series], backtest_mae: dict[str, float]) -> dict[str, float]:
    """Bobot per model dari galat absolut median (live) dicampur galat backtest sebagai prior.

    live: nama model -> deret |aktual - q50| dari evaluasi live, urut waktu.
    """
    scores = {}
    for name, prior in backtest_mae.items():
        errs = live.get(name, pd.Series(dtype=float)).dropna().iloc[-60:]
        n = len(errs)
        live_mae = float(errs.mean()) if n else prior
        scores[name] = (n * live_mae + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH)
    if not scores:
        return {}
    inv = {k: 1.0 / max(v, 1e-9) ** 2 for k, v in scores.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def combine(preds: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    out = sum(preds[k] * weights[k] for k in preds)
    q = np.sort([out["q10"], out["q50"], out["q90"]])
    out["q10"], out["q50"], out["q90"] = q
    return out


@dataclass
class Correction:
    bias: float = 0.0
    width_scale: float = 1.0
    n: int = 0

    def apply(self, pred: pd.Series) -> pd.Series:
        p = pred.copy()
        mid = p["q50"] + self.bias
        p["q10"] = mid - (pred["q50"] - pred["q10"]) * self.width_scale
        p["q90"] = mid + (pred["q90"] - pred["q50"]) * self.width_scale
        p["q50"] = mid
        return p


def error_correction(ensemble_evals: pd.DataFrame) -> Correction:
    """Hitung koreksi dari evaluasi prediksi gabungan sebelumnya.

    Contoh: prediksi Rp20rb, aktual Rp25rb -> raw_error = ln(25/20) = +0,22.
    Kalau galat seperti ini berulang, median berikutnya digeser naik.
    """
    if ensemble_evals.empty:
        return Correction()
    df = ensemble_evals.sort_values("base_date")
    errs = df["raw_error"].fillna(df["error"]).astype(float)
    n = len(errs)
    w = 0.5 ** (np.arange(n)[::-1] / BIAS_HALFLIFE)
    ewma = float(np.sum(w * errs) / np.sum(w))
    shrink = n / (n + BIAS_SHRINK)
    bias = ewma * shrink
    # Jangan menggeser lebih dari setengah sebaran galat agar tidak "mengejar" noise.
    cap = 0.5 * float(errs.abs().median()) if n else 0.0
    bias = float(np.clip(bias, -cap, cap))

    coverage = float(df["in_range"].iloc[-60:].mean())
    width = 1.0 + (TARGET_COVERAGE - coverage) * shrink
    return Correction(bias=bias, width_scale=float(np.clip(width, 0.7, 2.0)), n=n)
