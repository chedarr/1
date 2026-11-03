"""Baseline-сравнение для трёх задач (Лаб. №1).

Для каждой задачи (RUL_frac / voltage / capacity):
  - берётся тот же сплит по батареям, что у MLP (списки батарей из
    data/processed/metrics_{task}.json);
  - на train-строках обучаются три baseline (константа, LinearRegression,
    RandomForest), оценка — MAE / RMSE / R² на test-строках;
  - результаты сохраняются в data/processed/metrics_baselines_{task}.json;
  - сводка и сравнение с MLP — summary_baselines.json, comparison_baselines.json;
  - три графика: reports/figures/baselines_comparison.png (MAE),
    reports/figures/baselines_r2.png (R²),
    reports/figures/comparison_baselines.png (MAE в лог-шкале: без неё столбец
    Linear для RUL_frac, MAE = 11.26, на порядки выше остальных).

Запуск:
    py -X utf8 scripts/run_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import MODELS, MODEL_NAMES, evaluate_baselines
from src.config import (
    CAPACITY_DATASET_PATH,
    CAPACITY_FEATURES,
    CAPACITY_TARGET,
    FIGURES_DIR,
    METRICS_CAPACITY_PATH,
    METRICS_RUL_PATH,
    METRICS_VOLTAGE_PATH,
    PROCESSED_DIR,
    RUL_DATASET_PATH,
    RUL_FEATURES,
    RUL_TARGET,
    VOLTAGE_DATASET_PATH,
    VOLTAGE_FEATURES,
    VOLTAGE_TARGET,
)

TASKS = (
    # (task_key, cmp_key, датасет, признаки, цель, metrics-json с MLP)
    # task_key — имя файла metrics_baselines_{task_key}.json (как metrics_{task}.json);
    # cmp_key  — ключ в comparison_baselines.json (сравнение с ключами export.py).
    ("rul", "rul_frac", RUL_DATASET_PATH, RUL_FEATURES, RUL_TARGET, METRICS_RUL_PATH),
    ("voltage", "voltage", VOLTAGE_DATASET_PATH, VOLTAGE_FEATURES, VOLTAGE_TARGET, METRICS_VOLTAGE_PATH),
    ("capacity", "capacity", CAPACITY_DATASET_PATH, CAPACITY_FEATURES, CAPACITY_TARGET, METRICS_CAPACITY_PATH),
)

TASK_LABELS = {"rul_frac": "RUL_frac", "voltage": "voltage", "capacity": "capacity"}
TASK_UNITS = {"rul_frac": "доли", "voltage": "В", "capacity": "Ahr"}


def _save_json(name: str, data: dict):
    path = PROCESSED_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  сохранено: {path.name}")
    return path


def run() -> dict:
    """Прогнать baseline по всем задачам; вернуть comparison (для графиков)."""
    comparison: dict = {}
    summary: dict = {}
    for task, cmp_key, data_path, feats, target, metrics_path in TASKS:
        df = pd.read_parquet(data_path)
        with metrics_path.open(encoding="utf-8") as f:
            mlp = json.load(f)

        split = mlp["split"]
        print(f"\n=== {task} ({len(df)} строк, {df['battery_id'].nunique()} батарей) ===")
        print(f"  train батарей: {len(split['train'])}, test: {len(split['test'])}")

        res = evaluate_baselines(df, target, feats, split)
        print(f"  MLP      (test): " + _fmt_row(mlp["test"]))
        for model in MODELS:
            print(f"  {MODEL_NAMES[model]:7s} (test): " + _fmt_row(res[model]))

        # metrics_baselines_{task}.json: метрики baseline + справочная информация
        task_json = {model: res[model] for model in MODELS}
        task_json["task"] = cmp_key
        task_json["target"] = target
        task_json["n_rows"] = {
            "train": int(df[df["battery_id"].isin(split["train"])].shape[0]),
            "test": int(df[df["battery_id"].isin(split["test"])].shape[0]),
        }
        _save_json(f"metrics_baselines_{task}.json", task_json)

        comparison[cmp_key] = {"mlp": mlp["test"], **res}
        summary[cmp_key] = {"mlp": mlp["test"], **res}

    comparison_path = _save_json("comparison_baselines.json", comparison)
    _save_json("summary_baselines.json", {"tasks": summary})
    print(f"\nСравнение с MLP: {comparison_path}")
    return comparison


def _fmt_row(m: dict) -> str:
    return f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}"


def _bar_figure(comparison: dict, metric: str, out_name: str, title: str):
    """3 subplot (по задаче); bar chart одного метрика для MLP + 3 baseline."""
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 20})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    order = ("rul_frac", "voltage", "capacity")
    colors = {"mlp": "#c44e52", "constant": "#7f7f7f", "linear": "#4c72b0", "rf": "#55a868"}
    for ax, task in zip(axes, order):
        data = comparison[task]
        labels = ["MLP", "const", "LR", "RF"]
        vals = [data[m][metric] for m in ("mlp", "constant", "linear", "rf")]
        bars = ax.bar(labels, vals, color=[colors[m] for m in ("mlp", "constant", "linear", "rf")],
                      width=0.6)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=13)
        ax.set_title(f"{TASK_LABELS[task]} ({TASK_UNITS[task]})")
        if metric == "r2":
            ax.axhline(0, color="black", linewidth=1)
        ax.margins(y=0.2)
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График: {out}")


def _log_mae_figure(comparison: dict):
    """Один сгруппированный bar chart MAE (лог-шкала): MLP + 3 baseline × 3 задачи.

    Лог-шкала по оси Y обязательна: для RUL_frac линейная регрессия даёт
    MAE = 11.26 при целях в [0, 1] — на линейной оси её столбец на порядки
    выше остальных, и всё качество сравнения теряется.
    """
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 20})
    order = ("rul_frac", "voltage", "capacity")
    xlabels = [TASK_LABELS[t] for t in order]
    models = ("mlp", "constant", "linear", "rf")
    model_labels = ["MLP", "Const", "Linear", "RF"]
    colors = {"mlp": "#c44e52", "constant": "#7f7f7f", "linear": "#4c72b0", "rf": "#55a868"}

    x = np.arange(len(order))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, model in enumerate(models):
        vals = [comparison[t][model]["mae"] for t in order]
        bars = ax.bar(x + i * width, vals, width, label=model_labels[i],
                      color=colors[model])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v * 1.35),
                        ha="center", fontsize=12)

    ax.set_xlabel("Задача", fontsize=16)
    ax.set_ylabel("MAE (test, лог-шкала)", fontsize=16)
    ax.set_title("Сравнение MLP с baseline: MAE", fontsize=18)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(xlabels, fontsize=14)
    ax.legend(fontsize=14)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.margins(y=0.15)
    fig.tight_layout()
    out = FIGURES_DIR / "comparison_baselines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График: {out}")


def plot(comparison: dict):
    _bar_figure(comparison, "mae", "baselines_comparison.png",
                "MAE (test): MLP против baseline по трём задачам")
    _bar_figure(comparison, "r2", "baselines_r2.png",
                "R² (test): MLP против baseline по трём задачам")
    _log_mae_figure(comparison)


if __name__ == "__main__":
    comparison = run()
    plot(comparison)
    print("\nГотово.")