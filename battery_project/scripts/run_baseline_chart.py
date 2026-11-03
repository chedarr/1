# -*- coding: utf-8 -*-
"""График `baseline_comparison.png`: R² (test) для MLP и трёх baseline по трём задачам.

Данные читаются из готового `data/processed/summary_baselines.json` — baseline
не пересчитывается, обучение MLP не запускается. Запуск:

    python scripts/run_baseline_chart.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, PROCESSED_DIR  # noqa: E402

SUMMARY_PATH = PROCESSED_DIR / "summary_baselines.json"
OUT_NAME = "baseline_comparison.png"

TASKS = ("rul_frac", "voltage", "capacity")
TASK_LABELS = {"rul_frac": "A: RUL_frac", "voltage": "B: Напряжение", "capacity": "C: Ёмкость"}
MODELS = ("mlp", "constant", "linear", "rf")
MODEL_LABELS = {
    "mlp": "MLP",
    "constant": "константа",
    "linear": "линейная\nрегрессия",
    "rf": "Random\nForest",
}
# MLP — синий, baseline — серые оттенки
COLORS = {
    "mlp": "#1f77b4",
    "constant": "#bdbdbd",
    "linear": "#8c8c8c",
    "rf": "#5a5a5a",
}
# столбцы, уходящие далеко за пределы шкалы (R² линейной регрессии на RUL_frac)
CLIP_LIMIT = 5.0


def load_tasks() -> dict:
    with SUMMARY_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data.get("tasks"), dict):
        data = data["tasks"]
    missing = [t for t in TASKS if t not in data]
    if missing:
        raise SystemExit(f"В {SUMMARY_PATH.name} нет задач: {missing}")
    return data


def panel_limits(tasks_data: dict) -> tuple[float, float]:
    """Общие пределы панелей: R² = 0 и 1 внутри, выбросы за пределами обрезаются.

    Пределы считаются по всем задачам сразу, чтобы панели сравнивались честно:
    при разных шкалах столбцы на разных панелях нельзя сравнивать на глаз.
    """
    kept: list[float] = []
    for task in TASKS:
        kept += [float(tasks_data[task][m]["r2"]) for m in MODELS
                 if abs(float(tasks_data[task][m]["r2"])) <= CLIP_LIMIT]
    lo = min(min(kept), 0.0) - 0.15
    hi = max(max(kept), 1.0) + 0.12
    return lo, hi


def build() -> Path:
    data = load_tasks()
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 14,
        "axes.titlesize": 17, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 13, "figure.titlesize": 20,
        "lines.linewidth": 2.2, "axes.linewidth": 1.4,
    })
    lo, hi = panel_limits(data)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for ax, task in zip(axes, TASKS):
        vals = [float(data[task][m]["r2"]) for m in MODELS]
        best = max(range(len(vals)), key=lambda i: vals[i])

        for i, (model, value) in enumerate(zip(MODELS, vals)):
            clipped = max(lo, min(hi, value))
            ax.bar(i, clipped - lo, bottom=lo, width=0.62,
                   color=COLORS[model], edgecolor="#333333", linewidth=1.2,
                   zorder=2)
            out_of_scale = value < lo or value > hi
            if out_of_scale:
                # значение вне шкалы: зазубренный столбец у края + пометка
                ax.bar(i, 0.03, bottom=(hi - 0.03) if value > hi else lo,
                       width=0.62, color="#ffffff", edgecolor="#333333",
                       hatch="///", linewidth=1.2, zorder=3)
            if out_of_scale:
                label, size = f"{value:.1f} — вне шкалы", 11
            else:
                label, size = f"{value:.4f}", 13
            ax.annotate(label, (i, min(max(clipped, lo), hi)),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=size, fontweight="bold",
                        color="#1a1a1a", zorder=4)
            if i == best:
                ax.annotate("лучший", (i, min(max(clipped, lo), hi)),
                            textcoords="offset points", xytext=(0, 30),
                            ha="center", fontsize=13, fontweight="bold",
                            color="#0b5394", zorder=4)

        ax.axhline(0.0, color="black", ls="--", lw=1.6, zorder=1)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_ylim(lo, hi)
        if ax is axes[0]:
            ax.set_ylabel("R² (test)")
        ax.set_title(f"{TASK_LABELS[task]} — лучший: "
                     f"{'MLP' if MODELS[best] == 'mlp' else MODEL_LABELS[MODELS[best]].replace(chr(10), ' ')}")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[m], edgecolor="#333333")
               for m in MODELS]
    fig.legend(handles, [MODEL_LABELS[m].replace(chr(10), " ") for m in MODELS],
               loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("R² на тесте: MLP против baseline (пунктир — уровень R² = 0 "
                 "константного предсказателя)", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / OUT_NAME
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"График: {out}")
    return out


if __name__ == "__main__":
    build()
