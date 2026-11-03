# -*- coding: utf-8 -*-
"""Сборка HTML-отчёта: одиннадцать разделов, подписи рисунков, офлайн."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src import config, report_text
from src.html_report import _FIGURES, build_report

DIAG_PATH = config.REPORTS_DIR / "loeo_diagnostics.json"
LOEO_PATH = config.LOEO_PATH

# Точные названия одиннадцати разделов (титул задаётся заголовком первого уровня).
SECTIONS = [
    "2. Постановка задачи",
    "3. Данные",
    "4. Методология",
    "5. Результаты",
    "6. Сравнение",
    "7. Анализ",
    "8. Ограничения",
    "9. Выводы",
    "10. Приложение",
    "11. Использованные методы",
]
SUBSECTIONS = [
    "3.1. Проверки целостности данных",
    "4.1. Теоретическая основа",
    "4.2. Признаки",
    "4.3. Пайплайн",
    "5.1. Метрики на тестовой выборке",
    "5.2. Сравнение с baseline",
    "7.1. Разбор ошибок",
    "7.2. Важность признаков и корреляции",
    "7.3. Стабильность",
    "10.1. Схема пайплайна",
    "10.2. Схема архитектуры",
    "10.3. Выходные файлы",
    "10.4. Проверенные конфигурации",
    "10.5. Разложение прироста",
    "10.6. Вопросы, требующие уточнения",
]


def _metrics() -> dict:
    return {
        "accuracy": 0.9032, "precision": 1.0, "recall": 0.8125, "f1": 0.8966,
        "roc_auc": 0.926, "верно_норма": 30, "ложная_тревога": 0,
        "пропуск_отказа": 6, "верно_отказ": 26, "объектов": 62, "порог": 0.3952,
    }


def _loeo() -> dict:
    folds = [
        {"фолд": 1, "эксперимент": "304", "объектов": 20, "порог": 0.56,
         "f1": 1.0, "f1_в_пороге_0.5": 1.0, "recall": 1.0, "precision": 1.0,
         "accuracy": 1.0},
        {"фолд": 8, "эксперимент": "441", "объектов": 17, "порог": 0.76,
         "f1": 0.0, "f1_в_пороге_0.5": 1.0, "recall": 0.0, "precision": 0.0,
         "accuracy": 0.35},
    ]
    summary = {
        "f1_среднее": 0.642, "f1_отклонение": 0.406, "f1_минимум": 0.0,
        "f1_максимум": 1.0, "recall_среднее": 0.667, "recall_отклонение": 0.437,
        "recall_минимум": 0.0, "recall_максимум": 1.0,
        "precision_среднее": 0.725, "precision_отклонение": 0.374,
        "precision_минимум": 0.0, "precision_максимум": 1.0,
        "accuracy_среднее": 0.786, "accuracy_отклонение": 0.216,
        "accuracy_минимум": 0.355, "accuracy_максимум": 1.0,
        "провалов_ниже_0.5": 5, "полных_попаданий": 5, "отказов_всего": 132,
    }
    half = {
        "f1_0.5_среднее": 0.752, "f1_0.5_отклонение": 0.329,
        "f1_0.5_минимум": 0.0, "f1_0.5_максимум": 1.0,
        "recall_0.5_среднее": 0.766, "recall_0.5_отклонение": 0.345,
        "recall_0.5_минимум": 0.0, "recall_0.5_максимум": 1.0,
        "precision_0.5_среднее": 0.756, "precision_0.5_отклонение": 0.333,
        "precision_0.5_минимум": 0.0, "precision_0.5_максимум": 1.0,
        "accuracy_0.5_среднее": 0.842, "accuracy_0.5_отклонение": 0.175,
        "accuracy_0.5_минимум": 0.429, "accuracy_0.5_максимум": 1.0,
        "провалов_ниже_0.5": 3, "полных_попаданий": 7,
    }
    return {
        "фолдов": 16, "окон_всего": 266,
        "фолды": folds, "сводка": summary, "сводка_0.5": half,
        "объединённо_подобранный_порог": {**_metrics(), "порог_средний": 0.515},
    }


def _diagnostics() -> dict | None:
    if not DIAG_PATH.exists():
        return None
    return json.loads(DIAG_PATH.read_text(encoding="utf-8"))


def _figures() -> dict:
    return {key: config.FIGURES_DIR / filename
            for key, (_, filename, _) in _FIGURES.items()}


def _sample_ctx(with_improvement: bool = True) -> dict:
    m = _metrics()
    comparison = {
        "Итоговая модель": {"metrics": m},
        "Модель до улучшений": {"metrics": dict(m, f1=0.7692, recall=0.625)},
        "Модель из первого отчёта": {"metrics": dict(m, f1=0.6415, recall=0.5313)},
        "Логистическая регрессия": {"metrics": dict(m, f1=0.6383, recall=0.4688)},
        "Постоянный ответ «норма»": {"metrics": dict(m, f1=0.0, recall=0.0)},
        "Постоянный ответ «отказ»": {"metrics": dict(m, precision=0.5161, f1=0.6809)},
    }
    ctx = {
        "title": "Лабораторная работа №2 — Классификация состояния спутника (ADAPT)",
        "seed": 42,
        "n_train_exp": 12,
        "n_test_exp": 4,
        "train": {"окон": 204, "норма": 104, "отказ": 100, "экспериментов": 12},
        "test": {"окон": 62, "норма": 30, "отказ": 32, "экспериментов": 4},
        "checks": [{"проверка": "Число признаков", "результат": "OK", "детали": "1620"}],
        "architecture": "вход → 128 → 64 → 32 → 1 (прореживание 0.3, нормализация слоёв, L2 0.0001)",
        "mlp_metrics": m,
        "before": dict(m, f1=0.7692, recall=0.625),
        "cv": {"f1_среднее": 0.7935, "f1_отклонение": 0.1787},
        "folds": [{"фолд": 1, "accuracy": 0.85, "precision": 0.85, "recall": 0.77,
                   "f1": 0.79, "эксперименты": ["309"]}],
        "test_by_experiment": [
            {"эксперимент": "441", "объектов": 17, "accuracy": 1.0, "precision": 1.0,
             "recall": 1.0, "f1": 1.0},
            {"эксперимент": "442", "объектов": 15, "accuracy": 0.6, "precision": 1.0,
             "recall": 0.25, "f1": 0.4},
        ],
        "comparison": comparison,
        "figures": _figures(),
        "loeo": _loeo(),
        "diagnostics": _diagnostics(),
        "verdicts": ["вывод один", "вывод два"],
        "limitations": ["ограничение"],
        "questions": ["вопрос"],
        "loeo_notes": ["пояснение к разбросу"],
    }
    if with_improvement:
        ctx["improvement"] = {
            "winner": "D_ёмкость",
            "rows": [
                {"имя": "A_базовая", "дополнения": "как в базовой модели",
                 "окон_обучения": 204, "признаков": 1620, "порог_f1": 0.452,
                 "валидация_f1_0.5": 0.821, "валидация_f1": 0.829,
                 "тест_f1": 0.769, "тест_полнота": 0.625, "прирост_f1_на_тесте": 0.0},
                {"имя": "D_ёмкость", "дополнения": "нормализация, L2=0.0001",
                 "окон_обучения": 204, "признаков": 1620, "порог_f1": 0.395,
                 "валидация_f1_0.5": 0.833, "валидация_f1": 0.868,
                 "тест_f1": 0.897, "тест_полнота": 0.8125, "прирост_f1_на_тесте": 0.127},
            ],
            "разложение_прироста": [
                {"этап": "Первый отчёт: прежний порядок обучения, порог 0.5",
                 "метрики": dict(m, f1=0.6415, recall=0.5313)},
                {"этап": "Прежний порядок обучения, подобранный порог",
                 "метрики": dict(m, f1=0.6154, recall=0.5)},
                {"этап": "Новый порядок обучения, базовая архитектура, порог 0.5",
                 "метрики": dict(m, f1=0.7692, recall=0.625)},
                {"этап": "Новый порядок обучения, базовая архитектура, подобранный порог",
                 "метрики": dict(m, f1=0.7692, recall=0.625)},
                {"этап": "Итоговая модель: новый порядок и итоговая архитектура",
                 "метрики": m},
            ],
        }
        ctx["failed_methods"] = ["приём без прироста"]
        ctx["decomposition"] = ["разложение прироста"]
    return ctx


def test_report_has_eleven_sections():
    html = build_report(_sample_ctx())
    assert ("<h1>Лабораторная работа №2 — Классификация состояния спутника "
            "(ADAPT)</h1>") in html
    for name in SECTIONS:
        assert f"<h2>{name}</h2>" in html
    for name in SUBSECTIONS:
        assert f"<h3>{name}</h3>" in html


def test_all_fourteen_figures_are_embedded():
    html = build_report(_sample_ctx())
    assert html.count("data:image/png;base64") == 14


def test_each_figure_has_a_preceding_note():
    html = build_report(_sample_ctx())
    parts = html.split("<figure>")
    assert len(parts) - 1 == 14
    for index in range(1, len(parts)):
        earlier = parts[index - 1]
        assert ("Что на графике" in earlier) or ("Что на схеме" in earlier)


def test_figure_captions_are_numbered():
    html = build_report(_sample_ctx())
    assert "Рис. 7.4" in html
    assert "Рис. 10.2" in html
    for number in ("3.1", "3.2", "5.1", "5.2", "5.3", "5.4", "5.5", "6.1",
                   "7.1", "7.2", "7.3", "7.4", "10.1", "10.2"):
        assert f"Рис. {number} —" in html


def test_report_without_improvement_still_builds():
    html = build_report(_sample_ctx(with_improvement=False))
    assert "<h2>10. Приложение</h2>" in html
    assert "<h2>9. Выводы</h2>" in html
    assert html.count("data:image/png;base64") == 14


def test_report_is_offline():
    html = build_report(_sample_ctx())
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html
    assert not re.findall(r'(?:src|href)="(?:https?:)[^"]+"', html)


def test_report_has_no_banned_terms():
    html = build_report(_sample_ctx())
    for term in config.BANNED_IN_TEXT:
        assert term not in html


def test_report_uses_english_metric_terms():
    """В таблицах метрик не остаётся русских названий вместо английских."""
    html = build_report(_sample_ctx())
    for banned in ('<th class="num">Полнота</th>', '<th class="num">F-мера</th>',
                   '<th class="num">Доля верных</th>',
                   '<th class="num">Точность</th>'):
        assert banned not in html
    for expected in ('<th class="num">accuracy</th>',
                     '<th class="num">precision</th>',
                     '<th class="num">recall</th>',
                     '<th class="num">F1</th>'):
        assert expected in html
    assert "<td>accuracy</td>" in html and "<td>recall</td>" in html


def test_verdicts_have_seven_points():
    loeo = _loeo()
    ctx = _sample_ctx()
    ctx["loeo"] = loeo
    verdicts = report_text.verdicts(ctx)
    assert len(verdicts) == 7
    assert len(report_text.limitations(ctx)) in range(5, 8)
    assert len(report_text.questions()) == 4


def test_pooled_summary_numbers_are_filled():
    """Числа объединённого расчёта подставляются, а не остаются пустыми."""
    text = " ".join(report_text.loeo_verdict(_sample_ctx()))
    pooled = text.split("Объединённый расчёт", 1)
    assert len(pooled) == 2, "нет абзаца объединённого расчёта"
    assert "—" not in pooled[1], "остались незаполненные места"
    assert "при recall" in pooled[1] and "precision" in pooled[1]


def test_stability_paragraphs_from_real_artifact():
    """Абзацы устойчивости читают ключи так же, как они записаны в артефакте."""
    if not LOEO_PATH.exists():
        pytest.skip("оценка «по одному эксперименту» ещё не посчитана")
    loeo = json.loads(LOEO_PATH.read_text(encoding="utf-8"))
    ctx = _sample_ctx()
    ctx["loeo"] = loeo
    text = " ".join(report_text.loeo_verdict(ctx))
    pooled = text.split("Объединённый расчёт", 1)
    assert len(pooled) == 2, "нет абзаца объединённого расчёта"
    assert "—" not in pooled[1], "остались незаполненные места"
    assert "0.748" in pooled[1], "нет объединённой F-меры из артефакта"


def test_report_marks_missing_figure():
    ctx = _sample_ctx()
    ctx["figures"] = {}
    html = build_report(ctx)
    assert "нет файла графика" in html
    assert html.count("data:image/png;base64") == 0


@pytest.mark.parametrize("key", list(_FIGURES))
def test_figure_registry_has_number_and_file(key):
    number, filename, caption = _FIGURES[key]
    assert re.fullmatch(r"\d+\.\d+", number)
    assert filename.endswith(".png")
    assert caption
    assert Path(config.FIGURES_DIR / filename).exists(), f"нет файла {filename}"
