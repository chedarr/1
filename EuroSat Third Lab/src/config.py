# -*- coding: utf-8 -*-
"""Единая точка входа для путей, флагов и констант задачи.

Все пути строятся только через pathlib.Path: в коде нет ручных строк
с именем проекта (в имени есть пробелы).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "eurosat_rgb"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = PROJECT_ROOT / "docs"

SEED = 42
BATCH_SIZE = 64

# Окружение: TensorFlow 2.20.0 на Windows — CPU-сборка, GPU не используется.
TF_VERSION = "2.20.0"
PYTHON_VERSION = "3.13"
USE_GPU = False
MIXED_PRECISION = False
INTRA_OP_THREADS = 6
INTER_OP_THREADS = 1

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
N_CLASSES = len(CLASS_NAMES)

IMAGE_SHAPE = (64, 64, 3)
IMAGE_DTYPE = "uint8"

# Стратифицированный split: 70 / 15 / 15.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

TFDS_NAME = "eurosat/rgb"
TFDS_VERSION = "2.0.0"


def configure_runtime() -> None:
    """Фиксирует seed и потоки CPU. Вызывается один раз до работы с tf.data."""
    import random

    import numpy as np
    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
