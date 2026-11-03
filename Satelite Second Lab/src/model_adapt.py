# -*- coding: utf-8 -*-
"""Нейронная сеть для классификации окон телеметрии.

Архитектура задана в постановке: вход 1620 -> 64 -> прореживание 0.2 ->
32 -> 1 (сигмоида). Обучение на CPU, поэтому потоки ограничиваются
средствами config.
"""
from __future__ import annotations

from tensorflow import keras

from src import config


def build_mlp(n_features: int = config.N_FEATURES, seed: int = config.SEED) -> keras.Model:
    """Собирает и компилирует полносвязную сеть."""
    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,), name="sensors_in"),
            keras.layers.Dense(64, activation="relu", name="hidden_1"),
            keras.layers.Dropout(0.2, name="dropout"),
            keras.layers.Dense(32, activation="relu", name="hidden_2"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="mlp_adapt",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def callbacks(patience: int = 20) -> list:
    """Ранняя остановка и снижение шага обучения по валидационной потере."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(5, patience // 3), verbose=0
        ),
    ]


def count_parameters(model: keras.Model) -> int:
    """Число обучаемых параметров — для отчёта."""
    return int(sum(int(v.shape.num_elements()) for v in model.trainable_weights))
