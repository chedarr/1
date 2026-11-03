# -*- coding: utf-8 -*-
"""Augmentation: только для train, val/test идут без преобразований."""
from __future__ import annotations

import tensorflow as tf

from . import config


def build_augmentation(seed: int = config.SEED) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.05, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
            tf.keras.layers.RandomBrightness(0.1, seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmentation",
    )


def apply_augmentation(dataset, augmentation, training: bool):
    """Возвращает датасет с augmentation только при training=True."""
    if not training:
        return dataset
    return dataset.map(
        lambda image, label: (augmentation(image, training=True), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
