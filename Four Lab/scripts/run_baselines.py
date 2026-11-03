# -*- coding: utf-8 -*-
"""Baseline на том же test: persistence, constant, linear."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import config
from src.baselines import constant, linear, persistence
from src.metrics import evaluate
from src.pipeline import build_dataset


def main() -> int:
    t0 = time.time()
    data = build_dataset()
    x, y, masks = data["x"], data["y"], data["masks"]
    x_train, y_train = x[masks["train"]], y[masks["train"]]
    x_test, y_test = x[masks["test"]], y[masks["test"]]
    idx = data["target_index"]
    mean = data["target_mean"]

    preds = {
        "persistence": persistence(x_test, idx, data["horizon"], mean, data["target_std"]),
        "constant": constant(x_test, mean, data["horizon"]),
        "linear": linear(x_train, y_train, x_test),
    }
    persistence_mae = float(np.abs(y_test - preds["persistence"]).mean())

    results = {name: evaluate(y_test, preds[name], baseline_mae=persistence_mae) for name in preds}
    payload = {
        "target": config.TARGET_FEATURE,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "horizon": data["horizon"],
        "lookback": data["lookback"],
        "stride": data["stride"],
        "linear_alpha": 1.0,
        "results": results,
    }
    out = config.REPORTS / "summary_baselines.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, m in results.items():
        print(f"{name}: MAE={m['mae']:.5f} RMSE={m['rmse']:.5f} R2={m['r2']:.5f} "
              f"skill={m['skill_score']:.4f}")
    print(f"ИТОГ: baseline готов за {time.time() - t0:.1f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
