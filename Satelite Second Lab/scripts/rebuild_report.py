# -*- coding: utf-8 -*-
"""Пересборка HTML-отчёта лабораторной №2 только из готовых артефактов.

Обучение не запускается и сеть не читается: контекст собирается из сохранённых
файлов (метрики, оценка «по одному эксперименту», сравнение с простыми моделями,
поиск улучшений, диагностика разброса), а рисунки берутся готовыми файлами из
папки рисунков и встраиваются в HTML как data-URI. Описание архитектуры
запрашивается у модуля экспериментов, где хранится её словесное представление.

Запуск:  .venv\\Scripts\\python.exe scripts\\rebuild_report.py
Результат: reports\\lab2_report.html
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, export_adapt, html_report, report_text, validate_adapt  # noqa: E402
from src.experiments_adapt import architecture_text, find_config  # noqa: E402
from src.loader_adapt import load_split  # noqa: E402


def _figure_paths() -> dict[str, Path]:
    """Готовые файлы рисунков: ключ отчёта -> путь к png."""
    return {
        key: config.FIGURES_DIR / filename
        for key, (_, filename, _) in html_report._FIGURES.items()
    }


def build_context(log) -> dict:
    """Собирает контекст отчёта из сохранённых артефактов."""
    train, test = load_split()
    checks = validate_adapt.full_report(train, test)

    metrics = export_adapt.load_json(config.METRICS_PATH)
    loeo = export_adapt.load_json(config.LOEO_PATH)
    baselines = export_adapt.load_json(config.BASELINES_PATH)
    improvement_raw = export_adapt.load_json(config.REPORTS_DIR / "improvement_adapt.json")
    diagnostics = export_adapt.load_json(config.REPORTS_DIR / "loeo_diagnostics.json")

    config_name = metrics["итоговая_конфигурация"]
    cfg = find_config(config_name)
    architecture = architecture_text(cfg) if cfg else config_name

    improvement = {
        "winner": improvement_raw["победитель"],
        "rows": improvement_raw["вклад_методов"],
        "разложение_прироста": improvement_raw["разложение_прироста"],
    }

    figures = _figure_paths()
    missing = [path.name for path in figures.values() if not path.exists()]
    if missing:
        log.warning("нет файлов рисунков: %s", ", ".join(sorted(missing)))

    ctx = {
        "title": "Лабораторная работа №2 — Классификация состояния спутника (ADAPT)",
        "seed": metrics.get("зерно", config.SEED),
        "n_train_exp": len(train.experiments),
        "n_test_exp": len(test.experiments),
        "train": train.summary(),
        "test": test.summary(),
        "checks": checks,
        "architecture": architecture,
        "mlp_metrics": metrics["тестовая_выборка"],
        "before": metrics["до_улучшений"],
        "cv": metrics["кросс_валидация"],
        "folds": metrics["фолды"],
        "test_by_experiment": metrics["тест_по_экспериментам"],
        "comparison": baselines["сравнение"],
        "figures": figures,
        "loeo": loeo,
        "diagnostics": diagnostics,
        "improvement": improvement,
        "questions": report_text.questions(),
    }
    ctx["verdicts"] = report_text.verdicts(ctx)
    ctx["limitations"] = report_text.limitations(ctx)
    ctx["loeo_notes"] = report_text.loeo_verdict(ctx)
    ctx["failed_methods"] = report_text.failed_methods(improvement)
    ctx["decomposition"] = report_text.decomposition(improvement)
    return ctx


def main() -> int:
    log = config.setup_logging()
    try:
        ctx = build_context(log)
    except FileNotFoundError as exc:
        log.error("нет обязательного артефакта: %s", exc)
        return 1

    html = html_report.build_report(ctx)
    images = html.count("data:image/png;base64")
    path = html_report.write_report(html, config.REPORT_HTML)
    log.info("рисунков, попавших в отчёт: %d", images)
    log.info("отчёт сохранён: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
