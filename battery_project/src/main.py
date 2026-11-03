"""Оркестрация трёх экспериментов (Лаб. №1): RUL / напряжение / ёмкость.

Последовательность:
  1. parser (34 .mat → raw_cycles.parquet)
  2. clean (минимальная чистка)
  3. features.build_feature_table → data/processed/battery_features.parquet
  4. rul.make_rul_dataset → battery_rul.parquet (ровно 8 батарей)
  5. датасеты B и C (34 батареи, dropna по целям/признакам)
  6. обучение MLP для трёх задач → model_*.keras, metrics_*.json
  7. графики → reports/figures/
  8. экспорт: csv/xlsx/json, battery_features_report.md, lab1_report.md
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from . import parser as parser_mod
from .clean import save_clean
from .config import (
    CAPACITY_FEATURES,
    CAPACITY_TARGET,
    CAPACITY_DATASET_PATH,
    CLEAN_CYCLES_PATH,
    LAB1_REPORT_HTML_PATH,
    LAB1_REPORT_PATH,
    METRICS_CAPACITY_PATH,
    METRICS_RUL_PATH,
    METRICS_VOLTAGE_PATH,
    MODEL_CAPACITY_PATH,
    MODEL_RUL_PATH,
    MODEL_VOLTAGE_PATH,
    RUL_DATASET_PATH,
    RUL_FEATURES,
    RUL_TARGET,
    STYLE_CSS_PATH,
    VOLTAGE_DATASET_PATH,
    VOLTAGE_FEATURES,
    VOLTAGE_TARGET,
)
from .export import build_lab1_report, export_datasets_report, export_feature_table
from .features import (
    make_capacity_dataset,
    make_voltage_dataset,
    save_feature_table,
)
from .model import train_task
from .plots import make_task_figures, plot_capacity_degradation, plot_rul_distribution
from .rul import save_rul_dataset
from .validate import (
    validate_next_targets,
    validate_raw_counts,
    validate_rul_dataset,
)

TASKS = {
    "rul": (RUL_FEATURES, RUL_TARGET),
    "voltage": (VOLTAGE_FEATURES, VOLTAGE_TARGET),
    "capacity": (CAPACITY_FEATURES, CAPACITY_TARGET),
}


def run_pipeline() -> dict:
    print("\n=== 1. Парсинг .mat ===")
    raw = parser_mod.parse_all(out_path=parser_mod.RAW_CYCLES_PATH)
    assert validate_raw_counts(raw)

    print("\n=== 2. Чистка ===")
    cleaned = save_clean(raw)
    print(f"  строк после чистки: {len(cleaned)} "
          f"(было {len(raw)}), батарей: {cleaned['battery_id'].nunique()}")

    print("\n=== 3. Таблица признаков (34 батареи) ===")
    feat = save_feature_table(cleaned)
    print(f"  строк: {len(feat)}, батарей: {feat['battery_id'].nunique()}")
    assert validate_next_targets(feat, VOLTAGE_TARGET)
    assert validate_next_targets(feat, CAPACITY_TARGET)

    print("\n=== 4. RUL-датасет (8 батарей) ===")
    rul_df = save_rul_dataset(feat)
    assert validate_rul_dataset(rul_df)
    print(f"  строк: {len(rul_df)}, батарей: {rul_df['battery_id'].nunique()}")

    print("\n=== 5. Датасеты B и C (34 батареи) ===")
    voltage_df = make_voltage_dataset(feat, VOLTAGE_FEATURES, VOLTAGE_TARGET)
    capacity_df = make_capacity_dataset(feat, CAPACITY_FEATURES, CAPACITY_TARGET)
    voltage_df.to_parquet(VOLTAGE_DATASET_PATH, index=False)
    capacity_df.to_parquet(CAPACITY_DATASET_PATH, index=False)
    print(f"  voltage: {len(voltage_df)} строк, {voltage_df['battery_id'].nunique()} батарей")
    print(f"  capacity: {len(capacity_df)} строк, {capacity_df['battery_id'].nunique()} батарей")

    datasets = {
        "rul": {"rows": len(rul_df), "batteries": sorted(rul_df["battery_id"].unique()),
                "features": RUL_FEATURES, "target": RUL_TARGET},
        "voltage": {"rows": len(voltage_df), "batteries": sorted(voltage_df["battery_id"].unique()),
                    "features": VOLTAGE_FEATURES, "target": VOLTAGE_TARGET},
        "capacity": {"rows": len(capacity_df), "batteries": sorted(capacity_df["battery_id"].unique()),
                     "features": CAPACITY_FEATURES, "target": CAPACITY_TARGET},
    }

    print("\n=== 6. Обучение MLP (3 задачи) ===")
    model_paths = {
        "rul": (MODEL_RUL_PATH, METRICS_RUL_PATH, rul_df),
        "voltage": (MODEL_VOLTAGE_PATH, METRICS_VOLTAGE_PATH, voltage_df),
        "capacity": (MODEL_CAPACITY_PATH, METRICS_CAPACITY_PATH, capacity_df),
    }
    results, test_frames, histories = {}, {}, {}
    for name, (feat_cols, target) in TASKS.items():
        mp, metp, dset = model_paths[name]
        print(f"\n--- Задача {name} ({target}) ---")
        res, test_df, hist = train_task(
            dset, feat_cols, target, run_name=name,
            model_path=mp, metrics_path=metp,
        )
        results[name] = res
        test_frames[name] = test_df
        histories[name] = hist

    # --- нормированная цель RUL_frac: интерпретация в циклы ---------
    rul_res = results["rul"]
    n_bats_test = rul_res["n_batteries"]["test"]
    test_bats_rul = set(rul_res["split"]["test"])
    # средняя длина серии тестовых батарей (по батареям, а не по строкам)
    mean_total = float(
        rul_df[rul_df["battery_id"].isin(test_bats_rul)]
        .groupby("battery_id")["total_cycles"].first().mean()
    )
    rul_res["task"] = "rul_frac"
    rul_res["target"] = "RUL_frac"
    rul_res["mae_frac"] = rul_res["test"]["mae"]
    rul_res["rmse_frac"] = rul_res["test"]["rmse"]
    rul_res["r2_frac"] = rul_res["test"]["r2"]
    rul_res["mae_cycles"] = rul_res["test"]["mae"] * mean_total
    rul_res["mean_total_cycles_test"] = mean_total
    rul_res["per_battery_mae_frac"] = dict(rul_res["per_battery_mae_test"])
    with open(METRICS_RUL_PATH, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in rul_res.items() if k != "history"},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"  RUL-интерпретация: mae_frac={rul_res['mae_frac']:.4f}, "
          f"r2_frac={rul_res['r2_frac']:.4f}, "
          f"mae_cycles≈{rul_res['mae_cycles']:.2f} "
          f"(mean total_cycles теста ({n_bats_test} батарей) = {mean_total:.1f})")

    print("\n=== 7. Графики ===\n")
    make_task_figures("rul", histories["rul"], test_frames["rul"], stem="rul_frac",
                      target_units="доли (RUL_frac)", xlim=(0, 1), ylim=(0, 1))
    make_task_figures("voltage", histories["voltage"], test_frames["voltage"])
    make_task_figures("capacity", histories["capacity"], test_frames["capacity"])
    plot_capacity_degradation(feat)
    plot_rul_distribution(rul_df)

    print("\n=== 7b. Инфографики (10 шт.) ===")
    from .infographics import build_all
    build_all(results=results, datasets=datasets, test_frames=test_frames,
              histories=histories, feat=feat, cleaned=cleaned, rul_df=rul_df)

    print("\n=== 8. Экспорт ===")
    export_feature_table(feat)
    export_datasets_report(datasets)
    build_lab1_report(results, datasets,
                      n_total=cleaned["battery_id"].nunique(),
                      n_rul=rul_df["battery_id"].nunique())

    # --- сводная таблица метрик трёх задач (summary_metrics.json) ---
    from .config import SUMMARY_METRICS_PATH
    summary = {
        "tasks": [
            {"task": "RUL_frac", "mae": results["rul"]["test"]["mae"],
             "rmse": results["rul"]["test"]["rmse"],
             "r2": results["rul"]["test"]["r2"], "unit": "доли",
             "mae_cycles": results["rul"]["mae_cycles"]},
            {"task": "Напряжение", "mae": results["voltage"]["test"]["mae"],
             "rmse": results["voltage"]["test"]["rmse"],
             "r2": results["voltage"]["test"]["r2"], "unit": "В"},
            {"task": "Ёмкость", "mae": results["capacity"]["test"]["mae"],
             "rmse": results["capacity"]["test"]["rmse"],
             "r2": results["capacity"]["test"]["r2"], "unit": "Ahr"},
        ],
        "n_batteries": {"RUL": int(rul_df["battery_id"].nunique()),
                        "Напряжение": 34, "Ёмкость": 33},
        "split": "GroupShuffleSplit по батареям",
        "architecture": "Dense(64) → Dropout(0.2) → Dense(32) → Dense(1)",
    }
    with open(SUMMARY_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Сводка метрик: {SUMMARY_METRICS_PATH.name}")

    print("\n=== 9. HTML-отчёт ===")
    from .html_report import build_html
    build_html(LAB1_REPORT_PATH, STYLE_CSS_PATH, LAB1_REPORT_HTML_PATH)

    print("\n=== ИТОГ ===")
    for name in ("rul", "voltage", "capacity"):
        m = results[name]["test"]
        extra = ""
        if name == "rul":
            extra = (f"  (≈{results[name]['mae_cycles']:.1f} циклов в интерпретации)")
        print(f"  {name:8s}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}{extra}")

    return {"results": results, "datasets": datasets,
            "test_frames": test_frames, "histories": histories}


if __name__ == "__main__":
    run_pipeline()
    print("Готово.")