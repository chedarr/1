"""График сравнения версий V1 / V2 (Лаб. №1).

Строит reports/figures/comparison_versions.png: test R² по двум версиям набора
признаков для трёх задач основной линии. Обучение не выполняется — метрики
читаются из data/processed/:
    V1 — metrics_*_with_discharge_num.json (с служебным `discharge_num`);
    V2 — metrics_*.json (итоговый честный состав без служебных признаков).
Отдельный эксперимент V3 (абляция `temp_*`) в этот график не входит и описан
в подразделе 7.5 отчёта.

Запуск:
    py -X utf8 scripts/run_versions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, PROCESSED_DIR

TASKS = (
    # (метка на графике, файл метрик V1, файл метрик V2)
    ("RUL_frac", "metrics_rul_with_discharge_num.json", "metrics_rul.json"),
    ("voltage", "metrics_voltage_with_discharge_num.json", "metrics_voltage.json"),
    ("capacity", "metrics_capacity_with_discharge_num.json", "metrics_capacity.json"),
)


def _load_test_r2(fname: str) -> float:
    with (PROCESSED_DIR / fname).open(encoding="utf-8") as f:
        return json.load(f)["test"]["r2"]


def plot():
    labels = [t[0] for t in TASKS]
    v1 = [_load_test_r2(t[1]) for t in TASKS]
    v2 = [_load_test_r2(t[2]) for t in TASKS]

    plt.rcParams.update({"font.size": 15})
    x = np.arange(len(TASKS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.5))
    b1 = ax.bar(x - width / 2, v1, width, label="V1 (с discharge_num)", color="#c44e52")
    b2 = ax.bar(x + width / 2, v2, width, label="V2 (без служебных, итог)", color="#4c72b0")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=13)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Test R²")
    ax.set_title("Test R² по версиям V1 / V2 для трёх задач",
                 fontsize=17, fontweight="bold")
    ax.legend(fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    ax.margins(y=0.18)
    fig.tight_layout()
    out = FIGURES_DIR / "comparison_versions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График: {out}")


if __name__ == "__main__":
    plot()
