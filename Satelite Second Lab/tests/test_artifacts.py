# -*- coding: utf-8 -*-
"""Готовые артефакты: модель, метрики, baseline, графики, отчёт."""
from __future__ import annotations

import pytest

from src import config
from src.export_adapt import load_json

COMPARISON_KEYS = (
    "Итоговая модель",
    "Модель до улучшений",
    "Логистическая регрессия",
    "Постоянный ответ «норма»",
    "Постоянный ответ «отказ»",
)

EXPECTED_FIGURES = (
    "01_class_balance.png",
    "02_example_window.png",
    "03_confusion_matrix.png",
    "04_pr_curve.png",
    "05_roc_curve.png",
    "06_training_curves.png",
    "07_baseline_comparison.png",
    "08_model_comparison.png",
    "09_diag_f1_vs_frac_positive.png",
    "10_diag_f1_vs_dist_to_train.png",
    "11_diag_f1_vs_first_fault.png",
    "12_loeo_folds.png",
    "13_pipeline_schema.png",
    "14_model_architecture.png",
)


def test_model_saved():
    if not config.MODEL_PATH.exists():
        pytest.skip("модель ещё не обучена")
    assert config.MODEL_PATH.stat().st_size > 0


def test_metrics_json_structure():
    if not config.METRICS_PATH.exists():
        pytest.skip("метрики ещё не собраны")
    data = load_json(config.METRICS_PATH)
    assert data["зерно"] == config.SEED
    assert data["окно"]["признаков"] == config.N_FEATURES
    assert 0.0 <= data["порог_решения"] <= 1.0
    test_metrics = data["тестовая_выборка"]
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= test_metrics[key] <= 1.0
    assert data["кросс_валидация"]["фолдов"] == config.CV_FOLDS
    assert len(data["фолды"]) == config.CV_FOLDS
    assert len(data["тест_по_экспериментам"]) == config.N_TEST_EXPERIMENTS
    assert data["проверка_пересборки_окон"]["совпадает_размер"] is True
    assert data["проверка_пересборки_окон"]["тест_совпадают_метки"] is True
    assert "до_улучшений" in data


def test_baselines_json_has_all_models():
    if not config.BASELINES_PATH.exists():
        pytest.skip("baseline ещё не посчитаны")
    data = load_json(config.BASELINES_PATH)
    names = set(data["сравнение"])
    assert set(COMPARISON_KEYS).issubset(names)
    assert data["сравнение"]["Постоянный ответ «отказ»"]["recall"] == 1.0
    assert data["сравнение"]["Постоянный ответ «норма»"]["f1"] == 0.0


def test_improvement_json_lists_configurations():
    path = config.REPORTS_DIR / "improvement_adapt.json"
    if not path.exists():
        pytest.skip("поиск улучшений ещё не выполнен")
    data = load_json(path)
    assert data["победитель"]
    rows = data["вклад_методов"]
    assert len(rows) >= 5
    assert any(r["имя"] == "A_базовая" for r in rows)
    assert data["проверка_пересборки_окон"]["совпадают_метки"] is True
    ladder = data["разложение_прироста"]
    assert len(ladder) == 5
    assert ladder[-1]["метрики"]["f1"] >= ladder[0]["метрики"]["f1"]
    assert "контрольные_эксперименты" in data["порядок_обучения_из_первого_отчёта"]


def test_split_info_has_no_overlap():
    if not config.SPLIT_PATH.exists():
        pytest.skip("разбиение ещё не сохранено")
    data = load_json(config.SPLIT_PATH)
    assert data["пересечение"] == []
    assert len(data["обучение_эксперименты"]) == config.N_TRAIN_EXPERIMENTS
    assert len(data["тест_эксперименты"]) == config.N_TEST_EXPERIMENTS


def test_figures_exist():
    if not config.REPORT_HTML.exists():
        pytest.skip("отчёт ещё не собран")
    figures = sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))
    for expected in EXPECTED_FIGURES:
        assert expected in figures


def test_figures_present_and_nonempty():
    """Все 14 рисунков отчёта на месте и непустые (дешёвая проверка)."""
    for name in EXPECTED_FIGURES:
        path = config.FIGURES_DIR / name
        assert path.exists(), f"нет рисунка {name}"
        assert path.stat().st_size > 0, f"пустой рисунок {name}"


def test_loeo_json_has_all_folds():
    if not config.LOEO_PATH.exists():
        pytest.skip("честная оценка ещё не выполнена")
    data = load_json(config.LOEO_PATH)
    assert data["фолдов"] == config.N_EXPERIMENTS
    assert len(data["фолды"]) == config.N_EXPERIMENTS
    split = load_json(config.SPLIT_PATH)
    expected = set(split["обучение_эксперименты"]) | set(split["тест_эксперименты"])
    assert {f["эксперимент"] for f in data["фолды"]} == expected
    summary = data["сводка"]
    for key in ("f1_среднее", "f1_отклонение", "f1_минимум", "f1_максимум"):
        assert 0.0 <= summary[key] <= 1.0
    assert summary["f1_минимум"] <= summary["f1_среднее"] <= summary["f1_максимум"]
    assert 0.0 <= data["сводка_0.5"]["f1_0.5_среднее"] <= 1.0
    assert data["эталонный_тест"]["f1"] > 0.0
    assert data["эталонный_тест"]["экспериментов"] == config.N_TEST_EXPERIMENTS
    assert "не используется для выбора" in data["предупреждение"]


def test_step5_dataset_recorded_in_metrics():
    if not config.METRICS_PATH.exists():
        pytest.skip("метрики ещё не собраны")
    data = load_json(config.METRICS_PATH)
    step5 = data["проверка_датасета_с_шагом_5"]
    if not step5.get("есть_файл"):
        pytest.skip("датасет с частой сеткой ещё не собран")
    assert step5["проверка_пройдена"] is True
    assert step5["пересечение_с_тестом"] == []
    assert step5["нарушений_границ"] == 0
    assert step5["шаг"] == config.STRIDE_SMALL
    assert step5["окон"] > 300
    loeo = data["честная_оценка_по_одному_эксперименту"]
    assert loeo["фолдов"] == config.N_EXPERIMENTS


def test_report_is_self_contained():
    if not config.REPORT_HTML.exists():
        pytest.skip("отчёт ещё не собран")
    html = config.REPORT_HTML.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    for term in config.BANNED_IN_TEXT:
        assert term not in html
