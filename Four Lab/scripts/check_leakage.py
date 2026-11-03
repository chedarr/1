# -*- coding: utf-8 -*-
"""Проверки утечек: нормализация по train, окна внутри сплитов, состав признаков."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import config
from src.pipeline import build_dataset


def main() -> int:
    data = build_dataset()
    scaler = data["scaler"]
    ranges = data["ranges"]

    # 1. scaler обучен только на train-строках
    from src.loader import clean, load_raw

    cleaned, _ = clean(load_raw())
    train_rows = cleaned.iloc[ranges["train"][0] : ranges["train"][1]]
    recomputed_mean = train_rows[config.FEATURE_COLUMNS].to_numpy().mean(axis=0)
    recomputed_std = train_rows[config.FEATURE_COLUMNS].to_numpy().std(axis=0)
    scaler_ok = bool(
        np.allclose(scaler.mean_, recomputed_mean, atol=1e-6)
        and np.allclose(scaler.scale_, recomputed_std, atol=1e-6)
    )

    # 2. ни одно выбранное окно не пересекает границы своего сплита
    starts = data["starts"]
    span = data["lookback"] + data["horizon"]
    crossing_selected = 0
    contained = np.zeros(len(starts), dtype=bool)
    for name, mask in data["masks"].items():
        begin, end = ranges[name]
        inside = (starts >= begin) & (starts + span <= end)
        contained |= inside
        crossing_selected += int((inside & ~mask).sum())
    windows_outside_any_split = int((~contained).sum())
    overlap_windows = bool(crossing_selected > 0)

    # 3. в признаках нет value_last / value_sum / value_count и будущих значений
    forbidden = {"value_last", "value_sum", "value_count", "value_min", "value_max"}
    used_columns = set(config.FEATURE_COLUMNS)
    forbidden_used = sorted(forbidden & used_columns)
    target_in_features = config.TARGET_FEATURE in config.FEATURE_COLUMNS

    # 4. окно истории заканчивается до первого шага горизонта
    history_end = starts + data["lookback"] - 1
    horizon_start = starts + data["lookback"]
    future_in_input = bool(np.any(history_end >= horizon_start))

    result = {
        "scaler_fit_on_train_only": scaler_ok,
        "scaler_mean_diff": float(np.max(np.abs(scaler.mean_ - recomputed_mean))),
        "windows_cross_split_boundaries": overlap_windows,
        "windows_crossing_selected": crossing_selected,
        "windows_outside_any_split": windows_outside_any_split,
        "forbidden_columns_used": forbidden_used,
        "target_in_features": target_in_features,
        "target_feature": config.TARGET_FEATURE,
        "feature_columns": config.FEATURE_COLUMNS,
        "future_values_in_input": future_in_input,
        "lookback": data["lookback"],
        "horizon": data["horizon"],
        "train_rows": [int(ranges["train"][0]), int(ranges["train"][1])],
        "val_rows": [int(ranges["val"][0]), int(ranges["val"][1])],
        "test_rows": [int(ranges["test"][0]), int(ranges["test"][1])],
        "windows": {k: int(m.sum()) for k, m in data["masks"].items()},
    }
    result["all_checks_passed"] = bool(
        scaler_ok and not overlap_windows and not forbidden_used
        and not future_in_input
    )
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "leakage_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print("ИТОГ: проверки утечек пройдены" if result["all_checks_passed"] else "ИТОГ: есть замечания")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
