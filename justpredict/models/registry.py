"""Daftar model anggota ansambel. Tambahkan model hasil riset di sini."""
from __future__ import annotations

from typing import Callable

from .base import Forecaster, NaiveForecaster
from .gbm import LightGBMForecaster

MODEL_FACTORIES: dict[str, Callable[[], Forecaster]] = {
    "naive": NaiveForecaster,
    "lightgbm": LightGBMForecaster,
}


def make_models(names: list[str] | None = None) -> list[Forecaster]:
    names = names or list(MODEL_FACTORIES)
    return [MODEL_FACTORIES[n]() for n in names]
