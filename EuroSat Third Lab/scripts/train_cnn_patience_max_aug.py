# -*- coding: utf-8 -*-
"""Прогон CNN aug v2: patience=50, epochs=50, restore_best_weights=True.

Отличие от aug v1 (control_patience_max): добавлены RandomBrightness(0.1) и
RandomContrast(0.1) только для train, до нормализации. Архитектура, batch_size,
lr, optimizer, seed, split и нормализация — те же.

Тест открывается один раз, после обучения и восстановления лучших весов.
Артефакты aug v1 (reports/control_patience_max/*) не перезаписываются.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src import config
from src.cnn import build_cnn, make_dataset
from src.loader import load_arrays
from src.metrics import evaluate
from src.normalize import load_stats

CONTROL_DIR = config.REPORTS / "control_patience_max_aug"
FIGURES_DIR = CONTROL_DIR / "figures"
CHECKPOINT = CONTROL_DIR / "eurosat_cnn_patience_max_aug.keras"
EPOCHS = 50
PATIENCE = 50


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class EpochTimer(tf.keras.callbacks.Callback):
    """Время каждой эпохи — для лога и JSON."""

    def __init__(self) -> None:
        super().__init__()
        self.seconds: list[float] = []
        self._start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(round(time.time() - self._start, 2))


def main() -> int:
    if CHECKPOINT.exists() or (CONTROL_DIR / "metrics_eurosat_patience_max_aug.json").exists():
        print("ОСТАНОВ: артефакты aug v2 уже существуют — не перезаписываю.", flush=True)
        return 2

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config.configure_runtime()
    set_seeds(config.SEED)
    t0 = time.time()

    images, labels = load_arrays()
    info = json.loads((config.REPORTS / "split_info.json").read_text(encoding="utf-8"))
    index = np.asarray(info["index"], dtype=np.int64)
    split = np.asarray(info["split"])
    y = np.asarray(info["class_id"], dtype=np.int64)
    train_idx = index[split == "train"]
    val_idx = index[split == "val"]
    test_idx = index[split == "test"]
    stats = load_stats()

    train_ds = make_dataset(images[train_idx], y[train_idx], stats, training=True)
    val_ds = make_dataset(images[val_idx], y[val_idx], stats, training=False)
    # test_ds создаётся, но используется один раз — после обучения
    test_ds = make_dataset(images[test_idx], y[test_idx], stats, training=False)

    model = build_cnn()
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"params={model.count_params()}", flush=True)
    print(f"aug v2: patience={PATIENCE}, epochs={EPOCHS}, "
          f"restore_best_weights=True", flush=True)

    timer = EpochTimer()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[timer, early_stopping, checkpoint],
        verbose=2,
        shuffle=False,
    )
    total_epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    early_stopping_triggered = int(getattr(early_stopping, "stopped_epoch", 0)) > 0

    # Гарантия весов лучшей эпохи: checkpoint пишется по тому же val_loss
    model.load_weights(CHECKPOINT)

    # Тест открывается один раз
    prob_test = model.predict(test_ds, verbose=0)
    pred_test = np.argmax(prob_test, axis=1)
    prob_val = model.predict(val_ds, verbose=0)
    pred_val = np.argmax(prob_val, axis=1)

    test_metrics = evaluate(y[test_idx], pred_test, config.CLASS_NAMES)
    val_metrics = evaluate(y[val_idx], pred_val, config.CLASS_NAMES)
    errors = np.flatnonzero(pred_test != y[test_idx])[:30]

    per_class = [
        {
            "class_id": class_id,
            "class_name": name,
            **test_metrics["per_class"][name],
        }
        for class_id, name in enumerate(config.CLASS_NAMES)
    ]

    payload = {
        "run": "control_patience_max_aug",
        "augmentation_version": "aug_v2",
        "model": "cnn",
        "architecture": "Conv2D(16,32,64)+BatchNorm+ReLU+MaxPool -> Dense(128) -> Dropout(0.5) -> Dense(10)",
        "params": int(model.count_params()),
        "config": {
            "batch_size": config.BATCH_SIZE,
            "lr": 1e-3,
            "optimizer": "Adam",
            "seed": config.SEED,
            "patience": PATIENCE,
            "epochs": EPOCHS,
            "monitor": "val_loss",
            "restore_best_weights": True,
            "model_checkpoint": str(CHECKPOINT.relative_to(config.PROJECT_ROOT)),
            "augmentation": "flip + rotation + zoom + brightness + contrast (train only, before normalization)",
            "normalization": "stats from train only",
            "use_gpu": config.USE_GPU,
            "mixed_precision": config.MIXED_PRECISION,
            "intra_op_threads": config.INTRA_OP_THREADS,
            "inter_op_threads": config.INTER_OP_THREADS,
        },
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "total_epochs_run": total_epochs_run,
        "best_epoch": best_epoch,
        "early_stopping_triggered": early_stopping_triggered,
        "restore_best_weights": True,
        "checkpoint_path": str(CHECKPOINT.relative_to(config.PROJECT_ROOT)),
        "training_time_sec": round(time.time() - t0, 1),
        "epoch_seconds": timer.seconds,
        "history": {
            "epoch": list(range(total_epochs_run)),
            "train_loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "train_accuracy": [float(v) for v in history.history["accuracy"]],
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class": per_class,
            "confusion_matrix": test_metrics["confusion_matrix"],
            "n_samples": test_metrics["n_samples"],
        },
        "val": {
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
        },
        "error_indices": [int(test_idx[i]) for i in errors],
        "error_true": [int(y[test_idx][i]) for i in errors],
        "error_pred": [int(pred_test[i]) for i in errors],
    }

    out = CONTROL_DIR / "metrics_eurosat_patience_max_aug.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"test accuracy={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"total_epochs_run={total_epochs_run} best_epoch={best_epoch} "
          f"early_stopping_triggered={early_stopping_triggered}")
    print(f"ИТОГ: aug v2 завершён за {payload['training_time_sec']} c", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
