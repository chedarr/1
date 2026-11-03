# -*- coding: utf-8 -*-
"""Реестр экспериментов и абляций."""
from __future__ import annotations

CONFIGS: dict[str, dict] = {
    "A": {"label": "A_базовая", "arch": "lstm1", "units": 64, "dense": 32, "dropout": 0.1},
    "B": {"label": "B_одномерная", "arch": "lstm1", "units": 64, "dense": 32, "dropout": 0.1, "univariate": True},
    "C": {"label": "C_двухслойная", "arch": "lstm2", "units": 64, "dense": None, "dropout": 0.1},
    "D": {"label": "D_большая", "arch": "lstm1", "units": 128, "dense": 64, "dropout": 0.2},
    "E": {"label": "E_GRU", "arch": "gru", "units": 64, "dense": 32, "dropout": 0.1},
    "F": {"label": "F_L2", "arch": "l2", "units": 64, "dense": 32, "dropout": 0.1},
    "G": {"label": "G_BatchNorm", "arch": "batchnorm", "units": 64, "dense": 32, "dropout": 0.1},
    "E0": {"label": "E_без_dropout", "arch": "lstm1", "units": 64, "dense": 32, "dropout": 0.0},
}

ABLATIONS: dict[str, dict] = {
    "lb72": {"label": "lookback_72", "base": "A", "lookback": 72},
    "lb288": {"label": "lookback_288", "base": "A", "lookback": 288},
    "hz72": {"label": "horizon_72", "base": "A", "horizon": 72},
    "hz28": {"label": "horizon_28", "base": "A", "horizon": 28},
}


def resolve(key: str) -> dict:
    """Возвращает параметры запуска по ключу (A–G, E0, lb72…)."""
    if key in CONFIGS:
        cfg = dict(CONFIGS[key])
        cfg["key"] = key
        return cfg
    if key in ABLATIONS:
        spec = ABLATIONS[key]
        cfg = dict(CONFIGS[spec["base"]])
        cfg.update({k: v for k, v in spec.items() if k in ("lookback", "horizon", "stride")})
        cfg["label"] = spec["label"]
        cfg["key"] = key
        return cfg
    raise KeyError(f"Неизвестный эксперимент: {key}")
