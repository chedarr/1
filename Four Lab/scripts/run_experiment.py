# -*- coding: utf-8 -*-
"""Обучение одного эксперимента LSTM/GRU и оценка на val/test."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src import config
from src.experiments import resolve
from src.metrics import evaluate
from src.model import build
from src.pipeline import build_dataset


class TimerCallback(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.seconds: list[float] = []
        self._start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(round(time.time() - self._start, 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    config.configure_runtime()
    cfg = resolve(args.name)
    lookback = int(cfg.get("lookback", config.LOOKBACK))
    horizon = int(cfg.get("horizon", config.HORIZON))
    univariate = bool(cfg.get("univariate", False))

    t0 = time.time()
    data = build_dataset(lookback, horizon, args.stride, univariate)
    x, y = data["x"], data["y"]
    masks = data["masks"]
    x_train, y_train = x[masks["train"]], y[masks["train"]]
    x_val, y_val = x[masks["val"]], y[masks["val"]]
    x_test, y_test = x[masks["test"]], y[masks["test"]]
    t_mean, t_std = data["target_mean"], data["target_std"]

    y_train_s = (y_train - t_mean) / t_std
    y_val_s = (y_val - t_mean) / t_std

    model = build(cfg, lookback, x.shape[2], horizon)
    timer = TimerCallback()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
        ),
        timer,
    ]
    history = model.fit(
        x_train, y_train_s,
        validation_data=(x_val, y_val_s),
        batch_size=config.BATCH_SIZE,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    epochs_ran = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    pred_val = model.predict(x_val, verbose=0) * t_std + t_mean
    pred_test = model.predict(x_test, verbose=0) * t_std + t_mean

    target_pos = 0 if univariate else data["target_index"]
    last = x_test[:, -1, target_pos] * data["scaler"].scale_[data["target_index"]] + data["scaler"].mean_[data["target_index"]]
    persistence = np.repeat(last[:, None], horizon, axis=1)
    baseline_mae = float(np.abs(y_test - persistence).mean())

    test_metrics = evaluate(y_test, pred_test, baseline_mae=baseline_mae)
    val_metrics = evaluate(y_val, pred_val)

    out_dir = config.REPORTS / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": args.name,
        "label": cfg["label"],
        "config": {
            "arch": cfg.get("arch", "lstm1"),
            "units": cfg.get("units", 64),
            "dense": cfg.get("dense"),
            "dropout": cfg.get("dropout", 0.1),
            "lookback": lookback,
            "horizon": horizon,
            "stride": args.stride,
            "univariate": univariate,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "patience": args.patience,
            "epochs_requested": args.epochs,
        },
        "params": int(model.count_params()),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "early_stopping_triggered": bool(epochs_ran < args.epochs),
        "training_time_sec": round(time.time() - t0, 1),
        "epoch_seconds": timer.seconds,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "val": val_metrics,
        "test": test_metrics,
        "target_mean": t_mean,
        "target_std": t_std,
    }
    (out_dir / f"{args.name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    np.save(out_dir / f"{args.name}_pred_test.npy", pred_test.astype(np.float32))
    np.save(out_dir / f"{args.name}_y_test.npy", y_test.astype(np.float32))
    if args.save_model:
        model.save(out_dir / f"{args.name}.keras")

    print(
        f"RESULT {args.name} mae={test_metrics['mae']:.5f} rmse={test_metrics['rmse']:.5f} "
        f"r2={test_metrics['r2']:.5f} skill={test_metrics['skill_score']:.4f} "
        f"epochs={epochs_ran} best={best_epoch} params={payload['params']} "
        f"time={payload['training_time_sec']}c"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
