# -*- coding: utf-8 -*-
"""Архитектуры LSTM/GRU для многомерного прогноза."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def build(cfg: dict, lookback: int, n_features: int, horizon: int) -> tf.keras.Model:
    """Собирает модель по описанию конфигурации."""
    inputs = tf.keras.Input(shape=(lookback, n_features), name="window")
    kind = cfg.get("arch", "lstm1")
    units = cfg.get("units", 64)
    dropout = cfg.get("dropout", 0.1)
    dense_units = cfg.get("dense", 32)

    if kind == "lstm2":
        x = layers.LSTM(units, return_sequences=True)(inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(max(units // 2, 8))(x)
    elif kind == "gru":
        x = layers.GRU(units)(inputs)
    elif kind == "l2":
        x = layers.LSTM(units, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    elif kind == "batchnorm":
        x = layers.LSTM(units)(inputs)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.LSTM(units)(inputs)

    if dropout:
        x = layers.Dropout(dropout)(x)
    if dense_units:
        x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(horizon, name="forecast")(x)
    model = tf.keras.Model(inputs, outputs, name=cfg.get("name", "lstm"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.get("lr", 1e-3)),
        loss="mse",
        metrics=["mae"],
    )
    return model
