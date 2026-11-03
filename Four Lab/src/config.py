# -*- coding: utf-8 -*-
"""Конфигурация проекта: пути, seed, каналы, окна и горизонт."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "goce"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = PROJECT_ROOT / "docs"

SEED = 42

# Целевая переменная — напряжение батареи (BAT_VOL_R).
TARGET_CHANNEL = "PHT10180"
TARGET_COLUMN = "value_mean"
# Признаки — 5 каналов: заряд, разряд, напряжение, TMP_1, TMP_2.
CHANNELS = ["PHT10080", "PHT10100", "PHT10180", "PHT10260", "PHT10300"]
FEATURE_COLUMNS = [f"{c}_mean" for c in CHANNELS]
TARGET_FEATURE = f"{TARGET_CHANNEL}_mean"

# 10-минутная сетка: 144 шага = 24 часа.
LOOKBACK = 144
HORIZON = 144
MAX_GAP_STEPS = 6  # NaN-серия длиннее 1 часа => разрыв
STEP_MINUTES = 10

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
STRIDE = 12  # шаг между окнами при сборке выборки (2 часа)

BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 1e-3

USE_GPU = False
MIXED_PRECISION = False
INTRA_OP_THREADS = 6
INTER_OP_THREADS = 1


def configure_runtime() -> None:
    """Фиксирует потоки CPU и скрывает GPU: обучение идёт на CPU."""
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
    tf.keras.utils.set_random_seed(SEED)
