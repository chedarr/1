# -*- coding: utf-8 -*-
"""Проверка загруженного TFDS EuroSAT: 27k, 10 классов, форма, dtype, распределение."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from src import config

ds = tfds.load(
    config.TFDS_NAME,
    split="train",
    data_dir=str(config.DATA_RAW),
    download=False,
    as_supervised=True,
)

counts = np.zeros(config.N_CLASSES, dtype=np.int64)
n = 0
first_shape = None
first_dtype = None
for image, label in tfds.as_numpy(ds):
    counts[int(label)] += 1
    if n == 0:
        first_shape = image.shape
        first_dtype = image.dtype
    n += 1

print("n_images:", n)
print("shape:", first_shape, "dtype:", first_dtype)
print("classes match:", list(ds.element_spec) if False else "see builder")
for i, name in enumerate(config.CLASS_NAMES):
    print(f"  {i:2d} {name:22s} {counts[i]}")
print("min/max:", counts.min(), counts.max(), "ratio:", round(float(counts.max()) / counts.min(), 3))
print("total:", int(counts.sum()))
