# -*- coding: utf-8 -*-
"""Сохранение и загрузка артефактов: модель, метрики, история обучения."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config


def _to_builtin(obj):
    """Приводит numpy-типы к встроенным, чтобы json был корректным."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj, path: Path | str) -> Path:
    """Записывает словарь в json в кодировке UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model(model, path: Path | str = config.MODEL_PATH) -> Path:
    """Сохраняет сеть в формате Keras."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def save_history(history, path: Path | str = config.HISTORY_PATH) -> Path:
    """Сохраняет историю обучения без служебных объектов."""
    data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    return save_json(data, path)
