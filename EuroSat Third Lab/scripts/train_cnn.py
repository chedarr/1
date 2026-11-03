# -*- coding: utf-8 -*-
"""Обучение CNN на EuroSAT и оценка на test.

Архитектура и режим взяты из ТЗ: Conv2D(16/32/64) + BatchNorm + ReLU + MaxPool,
Dropout(0.5), Adam, early stopping (patience=3), batch_size=64, до 20 эпох.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src import config
from src.augment import build_augmentation
from src.cnn import build_cnn, make_dataset
from src.loader import load_arrays
from src.metrics import evaluate
from src.normalize import load_stats


def main() -> int:
    config.configure_runtime()
    t0 = time.time()

    images, labels = load_arrays()
    info = json.load(open(config.REPORTS / "split_info.json", encoding="utf-8"))
    idx = np.asarray(info["index"], dtype=np.int64)
    split = np.asarray(info["split"])
    y = np.asarray(info["class_id"], dtype=np.int64)
    train_idx = idx[split == "train"]
    val_idx = idx[split == "val"]
    test_idx = idx[split == "test"]
    stats = load_stats()

    augmentation = build_augmentation()
    train_ds = make_dataset(images[train_idx], y[train_idx], stats, training=True)
    val_ds = make_dataset(images[val_idx], y[val_idx], stats, training=False)
    test_ds = make_dataset(images[test_idx], y[test_idx], stats, training=False)
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

    model = build_cnn()
    model.summary(print_fn=lambda s: None)
    print(f"параметров: {model.count_params()}", flush=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        )
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
    )
    epochs_ran = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    prob_val = model.predict(val_ds, verbose=0)
    prob_test = model.predict(test_ds, verbose=0)
    pred_val = np.argmax(prob_val, axis=1)
    pred_test = np.argmax(prob_test, axis=1)

    test_metrics = evaluate(y[test_idx], pred_test, config.CLASS_NAMES)
    val_metrics = evaluate(y[val_idx], pred_val, config.CLASS_NAMES)
    errors = np.flatnonzero(pred_test != y[test_idx])[:30]

    payload = {
        "model": "cnn",
        "architecture": "Conv2D(16,32,64)+BatchNorm+ReLU+MaxPool -> Dense(128) -> Dropout(0.5) -> Dense(10)",
        "params": int(model.count_params()),
        "epochs_requested": 20,
        "epochs_ran": epochs_ran,
        "early_stopping_epoch": best_epoch,
        "batch_size": config.BATCH_SIZE,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "test": test_metrics,
        "val": val_metrics,
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "accuracy": [float(v) for v in history.history["accuracy"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "error_indices": [int(test_idx[i]) for i in errors],
        "error_true": [int(y[test_idx][i]) for i in errors],
        "error_pred": [int(pred_test[i]) for i in errors],
        "train_seconds": round(time.time() - t0, 1),
    }
    out = config.REPORTS / "metrics_eurosat.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    model.save(config.REPORTS / "eurosat_cnn.keras")

    summary_path = config.REPORTS / "summary_baselines.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cnn"] = {
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"test accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"epochs_ran={epochs_ran} best_epoch={best_epoch} params={model.count_params()}")
    print(f"ИТОГ: CNN обучена за {payload['train_seconds']} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
