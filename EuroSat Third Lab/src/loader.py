# -*- coding: utf-8 -*-
"""Загрузка EuroSAT из локального кэша TFDS (без сети)."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from src.config import DATA_RAW, TFDS_NAME


def load_dataset(data_dir=DATA_RAW, download: bool = False) -> tf.data.Dataset:
    """Возвращает tf.data.Dataset из (image uint8 (64,64,3), label int64)."""
    return tfds.load(
        TFDS_NAME,
        split="train",
        as_supervised=True,
        data_dir=str(data_dir),
        download=download,
    )


def load_arrays(data_dir=DATA_RAW) -> tuple[np.ndarray, np.ndarray]:
    """Материализует все снимки в uint8-массивы: (N, 64, 64, 3) и (N,)."""
    dataset = load_dataset(data_dir=data_dir)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(int(label))
    return np.stack(images), np.asarray(labels, dtype=np.int64)
