# -*- coding: utf-8 -*-
r"""Три недостающих рисунка отчёта собираются из готовых артефактов.

Что рисуется (обучение не запускается, сеть не читается, сеть не используется):
* 07_baseline_comparison.png — столбчатая диаграмма F-меры на тесте по всем
  моделям сравнения; данные берутся как есть из reports\summary_baselines.json
  (раздел «сравнение»);
* 13_pipeline_schema.png — схема пайплайна: от исходных измерений 16 записей
  через подготовку, нарезку окон 20 × 81 = 1620, разбиение по экспериментам,
  стандартизацию по обучению, сеть 128–64–32, подбор порога по валидации и
  метрики на тесте до отчёта;
* 14_model_architecture.png — схема итоговой сети. Слои берутся ровно те, что
  реально обучались и сохранены (reports\model_adapt.keras, конфигурация
  «D_ёмкость»: Dense 128 → BatchNorm → ReLU → Dropout 0.3, далее 64 и 32, затем
  Dense 1 с сигмоидой); число параметров и регуляризация — из
  reports\metrics_adapt.json.

Итоговый рисунок 07 и две схемы — только matplotlib (backend «Agg»).

Запуск:  .venv\Scripts\python.exe scripts\make_extra_figures.py
Код возврата 1 — если нет входного артефакта.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # без окна: сохранение прямо в файл

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src import config  # noqa: E402
from src.export_adapt import load_json  # noqa: E402
from src.plots_adapt import plot_baseline_comparison  # noqa: E402

# Итоговая конфигурация «D_ёмкость» из src\experiments_adapt.py: скрытые слои
# 128–64–32, прореживание 0.3, нормализация слоёв и L2 = 0.0001. Именно такую
# сеть строит experiments_adapt.build_model и именно она сохранена в .keras
# (проверено: 218305 обучаемых параметров, 448 необучаемых у нормализации).
HIDDEN_WIDTHS = (128, 64, 32)
DROPOUT = 0.3
L2 = 0.0001

BLUE = "#2b6cb0"
GREY = "#b8b8b8"
GREEN = "#2f855a"
ORANGE = "#c05621"
EDGE = "#4a5568"


# ------------------------------------------------------------------ утилиты
def _save(fig, name: str) -> Path:
    """Сохраняет рисунок в reports\figures под заданным именем."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _box(ax, cx: float, cy: float, w: float, h: float, text: str,
         facecolor: str, fontsize: float = 10.5) -> None:
    """Прямоугольный блок со скруглёнными углами и подписью по центру."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.4, edgecolor=EDGE, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="#1a202c")


def _arrow_down(ax, x: float, y_from: float, y_to: float) -> None:
    """Вертикальная стрелка сверху вниз."""
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.8))


def _arrow_right(ax, x_from: float, x_to: float, y: float) -> None:
    """Горизонтальная стрелка слева направо."""
    ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.8))


# ------------------------------------------------------ 07: сравнение с baseline
def make_baseline_comparison(log) -> Path:
    """Рисунок 07: F-мера на тесте по моделям сравнения."""
    if not config.BASELINES_PATH.exists():
        raise FileNotFoundError(config.BASELINES_PATH)
    data = load_json(config.BASELINES_PATH)
    comparison = data["сравнение"]
    path = plot_baseline_comparison(comparison)
    log.info("рисунок сравнения с baseline: %s", path)
    return path


# ---------------------------------------------------------- 13: схема пайплайна
def make_pipeline_schema(log) -> Path:
    """Рисунок 13: блок-схема обработки данных и обучения."""
    steps = [
        ("Исходные измерения: 16 экспериментов телеметрии\n"
         "(записи ADAPT и журнал отказов)", "#e6f4ea"),
        ("Подготовка и фильтрация: чистка, отбраковка,\n"
         "разметка окон меткой «норма» / «отказ»", "#e6f4ea"),
        ("Нарезка окон: 20 шагов × 81 сенсор = 1620 признаков", "#e8f0fb"),
        ("Разбиение по экспериментам:\n"
         "обучение — 12 записей, тест — 4 записи", "#e8f0fb"),
        ("Стандартизация: среднее и разброс\n"
         "считаются только по обучающей части", "#e8f0fb"),
        (f"Сеть 128–64–32 (прореживание {DROPOUT:g}, "
         f"нормализация слоёв, L2 {L2:g})", "#fdebd8"),
        ("Подбор порога решения по валидации\n"
         "(тест в выборе не участвует)", "#fdebd8"),
        ("Метрики на тестовой выборке:\n"
         "F-мера, полнота, точность, доля верных", "#e8f0fb"),
        ("Отчёт: рисунки и таблицы", "#ededed"),
    ]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(8.6, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h = 0.82, 0.072
    top, bottom = 0.955, 0.045
    step = (top - bottom) / (n - 1)
    centers = [top - i * step for i in range(n)]
    for i, (text, color) in enumerate(steps):
        _box(ax, 0.5, centers[i], box_w, box_h, text, color)
        if i < n - 1:
            _arrow_down(ax, 0.5, centers[i] - box_h / 2,
                        centers[i + 1] + box_h / 2)
    ax.set_title("Пайплайн: от измерений до отчёта", fontsize=13)
    path = _save(fig, "13_pipeline_schema.png")
    log.info("схема пайплайна: %s", path)
    return path


# -------------------------------------------------- 14: схема архитектуры сети
def make_model_architecture(log) -> Path:
    """Рисунок 14: слои итоговой сети с размерностями и регуляризацией."""
    if not config.METRICS_PATH.exists():
        raise FileNotFoundError(config.METRICS_PATH)
    metrics = load_json(config.METRICS_PATH)
    n_features = int(metrics["признаков_в_модели"])
    n_params = int(metrics["параметров_сети"])
    additions = metrics.get("дополнения", "")

    blocks = [
        (f"Вход\n{n_features} признаков\n(20 шагов × 81 сенсор)", GREEN),
        (f"Слой 1: {HIDDEN_WIDTHS[0]}\nDense (L2 {L2:g})\n"
         f"норм. слоёв, ReLU\nпрореживание {DROPOUT:g}", ORANGE),
        (f"Слой 2: {HIDDEN_WIDTHS[1]}\nDense (L2 {L2:g})\n"
         f"норм. слоёв, ReLU\nпрореживание {DROPOUT:g}", ORANGE),
        (f"Слой 3: {HIDDEN_WIDTHS[2]}\nDense (L2 {L2:g})\n"
         f"норм. слоёв, ReLU\nпрореживание {DROPOUT:g}", ORANGE),
        ("Выход: 1\nDense\nсигмоида\nвероятность отказа", GREEN),
    ]

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centers = [0.09, 0.295, 0.50, 0.705, 0.91]
    box_w, box_h = 0.17, 0.40
    cy = 0.58
    for cx, (text, color) in zip(centers, blocks):
        _box(ax, cx, cy, box_w, box_h, text, color, fontsize=9.5)
    for x_from, x_to in zip(centers[:-1], centers[1:]):
        _arrow_right(ax, x_from + box_w / 2, x_to - box_w / 2, cy)

    footer = (
        f"Итого обучаемых параметров: {n_params}. "
        f"Регуляризация: {additions}; прореживание {DROPOUT:g}. "
        f"Оптимизатор Adam (шаг 1e-3), потеря — бинарная перекрёстная энтропия, "
        f"ранняя остановка по валидационной потере."
    )
    ax.text(0.5, 0.18, footer, ha="center", va="center", fontsize=9.5,
            color="#2d3748", wrap=True)
    ax.set_title("Архитектура сети: вход → "
                 + " → ".join(str(w) for w in HIDDEN_WIDTHS) + " → 1",
                 fontsize=13)
    path = _save(fig, "14_model_architecture.png")
    log.info("схема архитектуры сети: %s", path)
    return path


# ------------------------------------------------------------------- вход
def main() -> int:
    config.setup_logging()
    log = config.get_logger()
    log.info("сборка дополнительных рисунков: старт")
    try:
        paths = [
            make_baseline_comparison(log),
            make_pipeline_schema(log),
            make_model_architecture(log),
        ]
    except FileNotFoundError as exc:
        log.error("нет входного артефакта: %s", exc)
        return 1
    for path in paths:
        log.info("готово: %s", path)
    log.info("сборка дополнительных рисунков завершена")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
