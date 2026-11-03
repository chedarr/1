# -*- coding: utf-8 -*-
"""CNN для EuroSAT: сборка модели и подготовка tf.data."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from . import config
from .augment import apply_augmentation, build_augmentation
from .normalize import normalize


def build_cnn(n_classes: int = config.N_CLASSES) -> tf.keras.Model:
    """CNN с нуля: Conv2D(16/32/64) + BatchNorm + MaxPooling + Dropout."""
    inputs = tf.keras.Input(shape=config.IMAGE_SHAPE, name="image")
    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="label")(x)
    model = tf.keras.Model(inputs, outputs, name="eurosat_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    stats: dict,
    training: bool,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """tf.data из uint8-массивов; нормализация по train, augmentation только в train."""
    mean = tf.constant(stats["mean_rgb"], dtype=tf.float32)
    std = tf.constant(stats["std_rgb"], dtype=tf.float32)
    augmentation = build_augmentation() if training else None

    def _prepare(image, label):
        image = tf.cast(image, tf.float32)
        if training:
            image = augmentation(image, training=True)
        image = (image - mean) / std
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(images), 5000), seed=config.SEED)
    ds = ds.map(_prepare, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
