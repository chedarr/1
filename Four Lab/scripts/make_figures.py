# -*- coding: utf-8 -*-
"""Четыре рисунка для отчёта: ряд, пример прогноза, MAE по шагам, сравнение."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.loader import clean, load_raw
from src.plots import comparison_plot, forecast_plot, mae_by_step


def series_plot(path: Path) -> None:
    cleaned, _ = clean(load_raw())
    info = json.loads((config.DATA_PROCESSED / "split_info.json").read_text(encoding="utf-8"))
    ranges = info["ranges"]
    index = cleaned.index
    values = cleaned[config.TARGET_FEATURE].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(index, values, linewidth=0.45, color="C0", label="PHT10180 (BAT_VOL_R)")
    for name, color, alpha in (("train", "C2", 0.06), ("val", "C1", 0.10), ("test", "C3", 0.10)):
        begin, end = ranges[name]
        ax.axvspan(index[begin], index[end - 1], color=color, alpha=alpha, label=f"{name} ({end - begin} строк)")
    ax.set_title("Напряжение батареи GOCE, 10-минутный ряд", pad=12)
    ax.set_xlabel("время", labelpad=10)
    ax.set_ylabel("напряжение, В", labelpad=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    series_plot(config.FIGURES / "01_target_series.png")

    exp_dir = config.REPORTS / "experiments"
    a = json.loads((exp_dir / "A.json").read_text(encoding="utf-8"))
    pred = np.load(exp_dir / "A_pred_test.npy")
    truth = np.load(exp_dir / "A_y_test.npy")

    forecast_plot(pred, truth, config.FIGURES / "02_forecast_example.png")

    mae_by_step(pred, truth, config.FIGURES / "03_mae_by_step.png")

    baselines = json.loads((config.REPORTS / "summary_baselines.json").read_text(encoding="utf-8"))["results"]
    rows = [(name, float(baselines[name]["mae"])) for name in ("persistence", "constant", "linear")]
    main_keys = {"A", "B", "C", "D", "E", "F", "G", "E0"}
    for path in sorted(exp_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["key"] not in main_keys:
            continue
        rows.append((payload["label"], float(payload["test"]["mae"])))
    comparison_plot(rows, config.FIGURES / "04_experiments.png")

    print(f"рисунки: 4, экспериментов в сравнении: {len(rows)}")
    print("ИТОГ: рисунки готовы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
