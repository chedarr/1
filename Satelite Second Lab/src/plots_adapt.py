# -*- coding: utf-8 -*-
"""Графики для отчёта: кривые обучения, матрица ошибок, ROC, сравнение моделей."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: сохранение прямо в файл

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from src import config  # noqa: E402
from src.metrics_adapt import compute_metrics  # noqa: E402

plt.rcParams["figure.figsize"] = (8, 4.5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def _save(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training(history, name: str = "06_training_curves.png") -> Path:
    """Потеря и точность по эпохам: обучающая и валидационная части."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, hist["loss"], label="обучение")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="валидация")
    axes[0].set_xlabel("эпоха")
    axes[0].set_ylabel("потеря")
    axes[0].set_title("Потеря")
    axes[0].legend()

    axes[1].plot(epochs, hist.get("accuracy", []), label="обучение")
    if "val_accuracy" in hist:
        axes[1].plot(epochs, hist["val_accuracy"], label="валидация")
    axes[1].set_xlabel("эпоха")
    axes[1].set_ylabel("точность")
    axes[1].set_title("Доля верных ответов")
    axes[1].legend()

    return _save(fig, name)


def plot_confusion(y_true, y_pred, name: str = "03_confusion_matrix.png") -> Path:
    """Матрица ошибок с абсолютными числами."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["норма", "отказ"])
    ax.set_yticks([0, 1], labels=["норма", "отказ"])
    ax.set_xlabel("предсказание")
    ax.set_ylabel("истина")
    ax.set_title("Матрица ошибок")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, name)


def plot_roc(y_true, y_prob, name: str = "05_roc_curve.png") -> Path:
    """Кривая ошибок с указанием площади под ней."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(fpr, tpr, label=f"площадь = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="случайное угадывание")
    ax.set_xlabel("доля ложных тревог")
    ax.set_ylabel("доля верно найденных отказов")
    ax.set_title("Кривая ошибок")
    ax.legend(loc="lower right")
    return _save(fig, name)


def plot_model_comparison(results: dict, name: str = "08_model_comparison.png") -> Path:
    """Столбчатая диаграмма метрик по моделям."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    titles = ["доля верных", "точность", "полнота", "F1"]
    models = list(results)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, m in enumerate(models):
        vals = [results[m]["metrics"].get(k, 0.0) for k in metrics]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2, labels=titles)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("значение метрики")
    ax.set_title("Сравнение моделей на тестовой выборке")
    # легенда вынесена за область осей: подписи моделей длинные и иначе
    # перекрывают столбцы диаграммы
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, fontsize=8)
    return _save(fig, name)


def plot_pr_curve(y_true, y_prob, threshold: float, name: str = "04_pr_curve.png") -> Path:
    """Кривая «точность–полнота» по порогу и выбранная рабочая точка."""
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve отдаёт на одну точку больше (порог = бесконечность)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    points = np.array(
        [compute_metrics(y_true, y_prob, float(t))["f1"] for t in thresholds]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].plot(thresholds, precision[:-1], label="точность")
    axes[0].plot(thresholds, recall[:-1], label="полнота")
    axes[0].plot(thresholds, points, label="F-мера", linestyle="--")
    axes[0].axvline(threshold, color="crimson", linestyle=":", label=f"рабочий порог {threshold:.2f}")
    axes[0].set_xlabel("порог решения")
    axes[0].set_ylabel("значение")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Метрики в зависимости от порога")
    axes[0].legend(fontsize=8)

    axes[1].plot(recall, precision, marker=".", markersize=3, label="кривая")
    m = compute_metrics(y_true, y_prob, float(threshold))
    axes[1].scatter([m["recall"]], [m["precision"]], color="crimson", zorder=5,
                    label=f"рабочая точка: полнота {m['recall']:.2f}, точность {m['precision']:.2f}")
    axes[1].set_xlabel("полнота (доля найденных отказов)")
    axes[1].set_ylabel("точность (доля верных среди тревог)")
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Кривая «точность–полнота»")
    axes[1].legend(fontsize=8)

    return _save(fig, name)


def plot_loeo_folds(folds: list[dict], mean: float, name: str = "12_loeo_folds.png") -> Path:
    """F-мера по каждому исключённому эксперименту и среднее по ним."""
    labels = [f["эксперимент"] for f in folds]
    values = [f["f1"] for f in folds]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, 0.7, color="#2b6cb0")
    for bar, value in zip(bars, values):
        if value < 0.5:
            bar.set_color("#b3261e")
    ax.axhline(mean, color="crimson", linestyle="--",
               label=f"среднее по фолдам {mean:.3f}")
    ax.set_xticks(x, labels=labels, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("исключённый эксперимент")
    ax.set_ylabel("F-мера")
    ax.set_title("Оценка «по одному эксперименту»: разброс между записями")
    ax.legend()
    return _save(fig, name)


def plot_class_balance(y_train, y_test, name: str = "01_class_balance.png") -> Path:
    """Распределение классов в обучающей и тестовой выборках."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    tr = [int((y_train == 0).sum()), int((y_train == 1).sum())]
    te = [int((y_test == 0).sum()), int((y_test == 1).sum())]
    ax.bar(x - 0.18, tr, 0.36, label="обучение")
    ax.bar(x + 0.18, te, 0.36, label="тест")
    ax.set_xticks(x, labels=["норма", "отказ"])
    ax.set_ylabel("число окон")
    ax.set_title("Распределение классов")
    # запас сверху: легенда не должна налезать на верхние столбцы
    ax.set_ylim(0, max(tr + te) * 1.25)
    ax.legend(loc="upper right")
    return _save(fig, name)


def _distinct_top_sensors(row, k: int = 3, rel_tol: float = 0.01) -> list[int]:
    """Индексы k самых изменчивых столбцов без почти совпадающих кривых."""
    variance = np.nanstd(row, axis=0)
    order = np.argsort(variance)[::-1]
    scale = float(np.nanmax(np.abs(row))) if row.size else 0.0
    tol = max(1e-9, rel_tol * scale)
    chosen: list[int] = []
    for j in order:
        if any(np.nanmax(np.abs(row[:, j] - row[:, i])) <= tol for i in chosen):
            continue
        chosen.append(int(j))
        if len(chosen) == k:
            break
    return chosen


def plot_example_window(X, y, features, name: str = "02_example_window.png") -> Path:
    """Фрагмент телеметрии до и после отказа для наглядности."""
    fault = np.flatnonzero(y == 1)
    if fault.size == 0:
        return _save(plt.subplots()[0], name)
    row = X[fault[0]].reshape(config.WINDOW_SIZE, config.N_SENSORS)

    # Почти совпадающие кривые не берём: иначе одна линия полностью
    # перекрывает другую и в легенде оказывается больше записей, чем видно.
    top = _distinct_top_sensors(row)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j in top:
        name_j = features[j].split("_", 1)[-1]
        ax.plot(range(1, config.WINDOW_SIZE + 1), row[:, j], marker="o", label=name_j)
    ax.set_xlabel("шаг окна")
    ax.set_ylabel("значение сенсора")
    ax.set_xticks(range(1, config.WINDOW_SIZE + 1, 2))
    ax.set_title("Пример окна с отказом: три самых изменчивых сенсора")
    ax.legend()
    return _save(fig, name)


FINAL_MODEL_NAME = "Итоговая модель"


def _f1_of(value) -> float:
    """Достаёт F-меру из записи модели: metrics.f1 либо f1 на верхнем уровне."""
    if isinstance(value, dict):
        metrics = value.get("metrics")
        if isinstance(metrics, dict) and "f1" in metrics:
            return float(metrics["f1"])
        if "f1" in value:
            return float(value["f1"])
    raise KeyError("в записи модели нет F-меры: ожидается value['metrics']['f1'] или value['f1']")


def plot_baseline_comparison(comparison: dict,
                             name: str = "07_baseline_comparison.png") -> Path:
    """Столбчатая диаграмма F-меры на тесте по всем моделям сравнения.

    Итоговая модель выделяется синим, остальные — серым; значения подписаны
    над столбцами, пунктирная линия отмечает уровень 0.5.
    """
    models = list(comparison)
    values = [_f1_of(comparison[m]) for m in models]
    colors = ["#2b6cb0" if m == FINAL_MODEL_NAME else "#b8b8b8" for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(models))
    bars = ax.bar(x, values, 0.62, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="crimson", linestyle="--", label="уровень 0.5")
    ax.set_xticks(x, labels=models, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F-мера")
    ax.set_title("Сравнение F-меры на тестовой выборке")
    ax.legend()
    return _save(fig, name)
