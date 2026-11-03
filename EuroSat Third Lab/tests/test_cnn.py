# -*- coding: utf-8 -*-
"""Тесты CNN: архитектура, форма выхода, подготовка tf.data."""
from __future__ import annotations

import numpy as np

from src import config
from src.cnn import build_cnn, make_dataset

STATS = {
    "mean_rgb": [86.7, 97.0, 102.7],
    "std_rgb": [52.0, 35.1, 29.7],
}


def test_output_shape_and_params():
    model = build_cnn()
    assert model.output_shape == (None, config.N_CLASSES)
    assert model.input_shape == (None, 64, 64, 3)
    assert model.count_params() > 100_000


def test_dataset_batch_is_normalized():
    images = np.full((5, 64, 64, 3), 100, dtype=np.uint8)
    labels = np.arange(5, dtype=np.int64)
    ds = make_dataset(images, labels, STATS, training=False, batch_size=2)
    batch_images, batch_labels = next(iter(ds))
    assert batch_images.shape == (2, 64, 64, 3)
    assert batch_images.dtype == np.float32
    assert batch_labels.shape == (2,)
    expected = (100.0 - STATS["mean_rgb"][0]) / STATS["std_rgb"][0]
    assert abs(float(batch_images[0, 0, 0, 0]) - expected) < 1e-4
