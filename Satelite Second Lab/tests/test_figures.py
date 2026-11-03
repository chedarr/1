# -*- coding: utf-8 -*-
"""Рисунки: легенды не должны перекрывать данные, содержимое — упираться в край.

Проверки геометрические: фигура собирается тем же кодом, что и в отчёте,
после чего сравниваются рамка легенды и рамки данных (линии, точки, столбцы).
Обучение не запускается и сеть не читается: берутся только готовые артефакты.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import config, plots_adapt
from src.export_adapt import load_json
from src.loader_adapt import load_split

TOLERANCE = 0.02  # доля площади столбца, которую легенде разрешено накрыть


def _capture(monkeypatch, call) -> "object":
    """Собирает фигуру, перехватывая её до сохранения."""
    holder: dict = {}

    def fake_save(fig, name):
        holder["fig"] = fig
        return Path(name)

    monkeypatch.setattr(plots_adapt, "_save", fake_save)
    call()
    return holder["fig"]


def _legend_problems(fig) -> list[str]:
    """Что легенда перекрывает: линии, точки диаграмм, столбцы, подписи."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems: list[str] = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        box = legend.get_window_extent(renderer)
        for line in ax.get_lines():
            xs, ys = line.get_xdata(), line.get_ydata()
            if len(xs) == 0:
                continue
            points = line.get_transform().transform(np.column_stack([xs, ys]))
            inside = ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
                      & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)).sum()
            if inside:
                problems.append(f"легенда накрывает {inside} точек линии «{line.get_label()}»")
        for patch in ax.patches:
            patch_box = patch.get_window_extent(renderer)
            if not patch_box.overlaps(box):
                continue
            overlap = ((min(patch_box.x1, box.x1) - max(patch_box.x0, box.x0))
                       * (min(patch_box.y1, box.y1) - max(patch_box.y0, box.y0)))
            share = overlap / (patch_box.width * patch_box.height or 1.0)
            if share > TOLERANCE:
                problems.append(f"легенда накрывает {share:.0%} площади столбца")
        for text in ax.texts:
            if not text.get_text().strip():
                continue
            if text.get_window_extent(renderer).overlaps(box):
                problems.append(f"легенда накрывает подпись «{text.get_text()[:30]}»")
        for title, artist in (("заголовок", ax.title), ("подпись оси X", ax.xaxis.label),
                              ("подпись оси Y", ax.yaxis.label)):
            if artist.get_text().strip() and \
                    artist.get_window_extent(renderer).overlaps(box):
                problems.append(f"легенда пересекает {title}")
    return problems


def _close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_class_balance_legend_is_clear(monkeypatch):
    """Легенда распределения классов не задевает столбцы."""
    if not config.TRAIN_PATH.exists():
        pytest.skip("нет подготовленных окон")
    train, test = load_split()
    fig = _capture(monkeypatch, lambda: plots_adapt.plot_class_balance(train.y, test.y))
    problems = _legend_problems(fig)
    _close(fig)
    assert problems == []


def test_model_comparison_legend_is_clear(monkeypatch):
    """Легенда сравнения моделей не задевает столбцы."""
    if not config.BASELINES_PATH.exists():
        pytest.skip("нет сравнения с простыми моделями")
    comparison = {model: {"metrics": data}
                  for model, data in load_json(config.BASELINES_PATH)["сравнение"].items()}
    fig = _capture(monkeypatch, lambda: plots_adapt.plot_model_comparison(comparison))
    problems = _legend_problems(fig)
    _close(fig)
    assert problems == []


def test_baseline_comparison_legend_is_clear(monkeypatch):
    """Легенда сравнения по F-мере не задевает столбцы и подписи значений."""
    if not config.BASELINES_PATH.exists():
        pytest.skip("нет сравнения с простыми моделями")
    comparison = load_json(config.BASELINES_PATH)["сравнение"]
    fig = _capture(monkeypatch, lambda: plots_adapt.plot_baseline_comparison(comparison))
    problems = _legend_problems(fig)
    _close(fig)
    assert problems == []


def test_loeo_folds_legend_is_clear(monkeypatch):
    """Легенда среднего по фолдам не задевает столбцы."""
    if not config.LOEO_PATH.exists():
        pytest.skip("нет отчёта LOEO")
    loeo = load_json(config.LOEO_PATH)
    fig = _capture(monkeypatch, lambda: plots_adapt.plot_loeo_folds(
        loeo["фолды"], loeo["сводка"]["f1_среднее"]))
    problems = _legend_problems(fig)
    _close(fig)
    assert problems == []


def test_example_window_legend_is_clear(monkeypatch):
    """Легенда примера окна не накрывает точки сенсоров."""
    if not config.TRAIN_PATH.exists():
        pytest.skip("нет подготовленных окон")
    train, _ = load_split()
    fig = _capture(monkeypatch, lambda: plots_adapt.plot_example_window(
        train.X, train.y, train.features))
    problems = _legend_problems(fig)
    _close(fig)
    assert problems == []


def test_training_curves_legends_are_clear(monkeypatch):
    """Легенды кривых обучения не накрывают сами кривые."""
    if not config.HISTORY_PATH.exists():
        pytest.skip("нет истории обучения")
    history = SimpleNamespace(history=load_json(config.HISTORY_PATH))
    fig = _capture(monkeypatch, lambda: plots_adapt.plot_training(history))
    problems = _legend_problems(fig)
    _close(fig)
    assert problems == []


def test_example_window_shows_three_distinct_curves():
    """Три линии примера окна не дублируют друг друга: иначе одна скрыта."""
    if not config.TRAIN_PATH.exists():
        pytest.skip("нет подготовленных окон")
    train, _ = load_split()
    row = train.X[np.flatnonzero(train.y == 1)[0]].reshape(
        config.WINDOW_SIZE, config.N_SENSORS)
    top = plots_adapt._distinct_top_sensors(row)
    assert len(top) == 3
    scale = float(np.nanmax(np.abs(row)))
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            diff = np.nanmax(np.abs(row[:, top[a]] - row[:, top[b]]))
            assert diff > 0.01 * scale, "две линии почти совпадают и скрывают друг друга"


def test_figures_have_margin():
    """Ни один готовый рисунок не обрезан краем: со всех сторон есть поля."""
    from PIL import Image

    if not config.FIGURES_DIR.exists():
        pytest.skip("нет папки рисунков")
    files = sorted(config.FIGURES_DIR.glob("*.png"))
    assert files, "в папке рисунков нет ни одного png"
    for path in files:
        pixels = np.asarray(Image.open(path).convert("RGB"))
        content = pixels.sum(axis=2) < 3 * 250
        rows = np.where(content.any(axis=1))[0]
        cols = np.where(content.any(axis=0))[0]
        assert rows.size and cols.size, f"{path.name}: рисунок пустой"
        top, bottom = rows.min(), pixels.shape[0] - 1 - rows.max()
        left, right = cols.min(), pixels.shape[1] - 1 - cols.max()
        assert min(top, bottom, left, right) >= 5, (
            f"{path.name}: содержимое упирается в край "
            f"(поля: top={top}, bottom={bottom}, left={left}, right={right})")
