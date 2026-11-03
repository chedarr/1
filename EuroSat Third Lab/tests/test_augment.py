# -*- coding: utf-8 -*-
"""Проверки augmentation: применяется только к train."""
import numpy as np
import tensorflow as tf

from src.augment import apply_augmentation, build_augmentation


def _dataset(n=8):
    images = np.random.default_rng(0).integers(0, 256, size=(n, 8, 8, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    return tf.data.Dataset.from_tensor_slices((images, labels))


def test_augmentation_layer_is_built():
    aug = build_augmentation()
    assert isinstance(aug, tf.keras.Sequential)
    names = [type(layer).__name__ for layer in aug.layers]
    assert "RandomFlip" in names
    assert "RandomRotation" in names
    assert "RandomZoom" in names


def test_validation_dataset_is_unchanged():
    ds = _dataset()
    out = apply_augmentation(ds, build_augmentation(), training=False)
    original = list(ds.as_numpy_iterator())
    processed = list(out.as_numpy_iterator())
    assert len(original) == len(processed)
    for (x0, y0), (x1, y1) in zip(original, processed):
        assert np.array_equal(x0, x1)
        assert np.array_equal(y0, y1)


def test_training_augmentation_keeps_shape_and_dtype():
    ds = _dataset()
    out = apply_augmentation(ds, build_augmentation(), training=True)
    x, y = next(iter(out.as_numpy_iterator()))
    assert x.shape == (8, 8, 3)
    assert x.dtype == np.float32
    assert y.dtype == np.int64
