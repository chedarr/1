# -*- coding: utf-8 -*-
"""Полный пайплайн лабораторной №2: данные -> проверки -> поиск улучшений ->
итоговая модель -> сравнение с простыми моделями -> графики -> HTML-отчёт.

Выбор конфигурации и порога решения выполняется ТОЛЬКО по валидации внутри
обучающей выборки. Тестовая выборка используется лишь для итогового отчёта.

Запуск:  .venv\\Scripts\\python.exe -m src.main_adapt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config, export_adapt, html_report, plots_adapt, report_text, validate_adapt
from src.evaluate_adapt import aggregate_folds, baseline_majority
from src.experiments_adapt import (
    build_windows,
    find_config,
    load_measurements,
    original_protocol_probe,
    run_search,
    verify_against_reference,
)
from src.loader_adapt import load_split
from src.loeo_adapt import run_loeo
from src.metrics_adapt import compute_metrics
from src.model_adapt import count_parameters
from src.split_adapt import assert_disjoint

log = config.get_logger()
REFERENCE_CONFIG = "A_базовая"
BASELINE_MODEL_PATH = config.REPORTS_DIR / "model_baseline.keras"
EXP_COL = config.META_PREFIX + "experiment"

# Диагностические и схемные рисунки готовятся отдельным шагом; в отчёт они
# попадают, если файл уже существует.
EXTRA_FIGURES = {
    "diag_frac_positive": "09_diag_f1_vs_frac_positive.png",
    "diag_dist_to_train": "10_diag_f1_vs_dist_to_train.png",
    "diag_first_fault": "11_diag_f1_vs_first_fault.png",
    "pipeline_schema": "13_pipeline_schema.png",
    "model_architecture": "14_model_architecture.png",
}


def _find(results: list[dict], name: str | None) -> dict | None:
    for item in results:
        if item["имя"] == name and "ошибка" not in item:
            return item
    return None


def _contributions(results: list[dict], base_name: str) -> list[dict]:
    """Вклад каждого набора методов относительно базовой конфигурации."""
    base = _find(results, base_name)
    if base is None:
        return []
    rows = []
    for item in results:
        if "ошибка" in item:
            rows.append({"имя": item["имя"], "описание": item["описание"],
                         "дополнения": item["дополнения"], "ошибка": item["ошибка"]})
            continue
        rows.append(
            {
                "имя": item["имя"],
                "описание": item["описание"],
                "дополнения": item["дополнения"],
                "окон_обучения": item["окон_обучения"],
                "признаков": item["признаков"],
                "порог_f1": item["порог_f1"],
                "порог_f2": item["порог_f2"],
                "валидация_f1_0.5": item["валидация_порог_0.5"]["f1"],
                "валидация_f1": item["валидация_порог_f1"]["f1"],
                "валидация_полнота": item["валидация_порог_f1"]["recall"],
                "тест_f1_0.5": item["тест_порог_0.5"]["f1"],
                "тест_f1": item["тест_порог_f1"]["f1"],
                "тест_полнота": item["тест_порог_f1"]["recall"],
                "тест_точность": item["тест_порог_f1"]["precision"],
                "тест_доля_верных": item["тест_порог_f1"]["accuracy"],
                "прирост_f1_на_тесте": item["тест_порог_f1"]["f1"]
                - base["тест_порог_f1"]["f1"],
                "прирост_f1_на_валидации": item["валидация_порог_f1"]["f1"]
                - base["валидация_порог_f1"]["f1"],
            }
        )
    return rows


def _verify_step5_dataset(meas, sensors, test_experiments: list[str]) -> dict:
    """Проверяет готовый датасет обучения с частой сеткой окон.

    Смысл проверки — исключить утечку: обучающие окна не должны пересекаться с
    тестовыми ни по экспериментам, ни по координатам, а сами данные должны
    совпадать с пересборкой из исходных измерений.
    """
    if not config.STEP5_PATH.exists():
        return {"есть_файл": False, "проверка_пройдена": False}
    frame = pd.read_parquet(config.STEP5_PATH)
    x5, y5, _ = build_windows(meas, sensors, config.STRIDE_SMALL, (0,), test=False)
    flat = x5.reshape(len(x5), -1)
    feats = [c for c in frame.columns
             if not c.startswith(config.META_PREFIX) and c != config.TARGET]
    same_size = len(frame) == len(y5) and len(feats) == flat.shape[1]
    same_labels = bool(
        same_size
        and np.array_equal(frame[config.TARGET].to_numpy(dtype=np.int32), y5)
    )
    max_diff = (
        float(np.abs(frame[feats].to_numpy(dtype=np.float32) - flat).max())
        if same_size
        else float("nan")
    )
    experiments = sorted(set(frame["meta__experiment"]))
    overlap = sorted(set(experiments) & set(test_experiments))
    bad_bounds = int(
        (frame["meta__end_index"] != frame["meta__start_index"] + config.WINDOW_SIZE - 1).sum()
    )
    ok = bool(
        same_size and same_labels and max_diff == 0.0 and not overlap and bad_bounds == 0
    )
    return {
        "есть_файл": True,
        "шаг": config.STRIDE_SMALL,
        "окон": int(len(frame)),
        "экспериментов": len(experiments),
        "совпадает_размер": same_size,
        "совпадают_метки": same_labels,
        "максимальное_расхождение": max_diff,
        "пересечение_с_тестом": overlap,
        "нарушений_границ": bad_bounds,
        "проверка_пройдена": ok,
    }


def main() -> int:
    config.setup_logging()
    config.set_seed()
    config.limit_resources()
    for folder in (config.REPORTS_DIR, config.FIGURES_DIR, config.DOCS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    log.info("1) датасет и проверки корректности")
    train, test = load_split()
    assert_disjoint(train.meta, test.meta)
    checks = validate_adapt.full_report(train, test)
    for c in checks:
        log.info(f"   [{c['результат']}] {c['проверка']}: {c['детали']}")
    if validate_adapt.problems(checks):
        log.error("!! обучение остановлено: есть критические проблемы данных")
        return 1

    log.info("2) сверка пересборки окон из сырых измерений")
    meas, sensors = load_measurements()
    x_ref, y_ref, meta_ref = build_windows(meas, sensors, 10, (0,), test=False)
    rebuild = verify_against_reference(x_ref, y_ref, meta_ref)
    x_test, y_test, meta_test = build_windows(meas, sensors, 10, (0,), test=True)
    rebuild["тест_совпадают_метки"] = bool(np.array_equal(y_test, test.y))
    rebuild["тест_максимальное_расхождение"] = float(
        np.abs(x_test.reshape(len(x_test), -1) - test.X).max()
    )
    log.info(f"   {rebuild}")
    rebuild_ok = (
        rebuild["совпадает_размер"]
        and rebuild["совпадают_метки"]
        and rebuild["совпадают_начала_окон"]
        and rebuild["максимальное_расхождение"] == 0.0
        and rebuild["тест_совпадают_метки"]
        and rebuild["тест_максимальное_расхождение"] == 0.0
    )
    if not rebuild_ok:
        log.error("!! пересборка окон не совпала с принятым датасетом — работа остановлена")
        return 1

    step5 = _verify_step5_dataset(meas, sensors, test.experiments)
    log.info(f"   датасет с шагом {config.STRIDE_SMALL}: {step5}")
    if step5["есть_файл"] and not step5["проверка_пройдена"]:
        log.error("!! датасет с частой сеткой окон не согласован — работа остановлена")
        return 1

    log.info("3) поиск улучшений (выбор только по валидации)")
    search = run_search(meas, sensors, progress=log.info)
    results = search["конфигурации"]
    for item in results:
        if "ошибка" in item:
            log.info(f"     {item['имя']}: ошибка {item['ошибка']}")
            continue
        log.info(
            f"     {item['имя']:<16} валидация F1 {item['валидация_порог_f1']['f1']:.3f} "
            f"| тест F1 {item['тест_порог_f1']['f1']:.3f} "
            f"полнота {item['тест_порог_f1']['recall']:.3f} "
            f"порог {item['порог_f1']:.3f}"
        )

    winner = _find(results, search["победитель"])
    reference = _find(results, REFERENCE_CONFIG)
    if winner is None or reference is None:
        log.error("!! не найдено ни одной успешной конфигурации")
        return 1
    logreg_item = search["логистическая_регрессия"]

    final_metrics = winner["тест_порог_f1"]
    final_at_half = winner["тест_порог_0.5"]
    final_recall_oriented = winner["тест_порог_f2"]
    base_metrics = reference["тест_порог_0.5"]
    majority = baseline_majority(test.y)
    always_fault = compute_metrics(y_test, np.ones(len(y_test), dtype=float))

    original = original_protocol_probe(x_ref, y_ref, meta_ref, x_test, y_test)
    log.info(
        f"   прежний порядок обучения: тест F1 {original['тест_порог_0.5']['f1']:.3f}, "
        f"полнота {original['тест_порог_0.5']['recall']:.3f} "
        f"(контроль {original['контрольные_эксперименты']}, эпох {original['эпох']})"
    )

    comparison = {
        "Итоговая модель": {"metrics": final_metrics},
        "Модель до улучшений": {"metrics": base_metrics},
        "Модель из первого отчёта": {"metrics": original["тест_порог_0.5"]},
        "Логистическая регрессия": {"metrics": logreg_item["тест_порог_f1"]},
        "Постоянный ответ «норма»": {"metrics": majority},
        "Постоянный ответ «отказ»": {"metrics": always_fault},
    }
    contributions = _contributions(results, REFERENCE_CONFIG)
    ladder = [
        {"этап": "Первый отчёт: прежний порядок обучения, порог 0.5",
         "метрики": original["тест_порог_0.5"]},
        {"этап": "Прежний порядок обучения, подобранный порог",
         "метрики": original["тест_порог_f1"]},
        {"этап": "Новый порядок обучения, базовая архитектура, порог 0.5",
         "метрики": reference["тест_порог_0.5"]},
        {"этап": "Новый порядок обучения, базовая архитектура, подобранный порог",
         "метрики": reference["тест_порог_f1"]},
        {"этап": "Итоговая модель: новый порядок и итоговая архитектура",
         "метрики": final_metrics},
    ]
    improvement = {
        "winner": winner["имя"],
        "rows": contributions,
        "разложение_прироста": ladder,
    }

    log.info(f"4) победитель по валидации: {winner['имя']}, порог {winner['порог_f1']:.3f}")
    log.info(f"   тест: F1 {final_metrics['f1']:.3f}, полнота {final_metrics['recall']:.3f}, "
          f"точность {final_metrics['precision']:.3f}")

    prob_test = winner["_test"]
    by_exp = []
    for exp in sorted(set(meta_test[EXP_COL])):
        mask = (meta_test[EXP_COL] == exp).to_numpy()
        m = compute_metrics(y_test[mask], prob_test[mask], winner["порог_f1"])
        m["эксперимент"] = exp
        by_exp.append(m)

    log.info("5) честная оценка по одному эксперименту (все %d записей)",
             config.N_EXPERIMENTS)
    loeo_cfg = find_config(winner["имя"]) or find_config(REFERENCE_CONFIG)
    loeo = run_loeo(
        loeo_cfg,
        np.concatenate([x_ref, x_test]),
        np.concatenate([y_ref, y_test]),
        pd.concat([meta_ref, meta_test], ignore_index=True),
        logger=log,
    )
    loeo["эталонный_тест"] = {
        "f1": final_metrics["f1"],
        "recall": final_metrics["recall"],
        "precision": final_metrics["precision"],
        "accuracy": final_metrics["accuracy"],
        "порог": final_metrics["порог"],
        "окон": int(len(y_test)),
        "экспериментов": int(len(test.experiments)),
    }
    loeo["окон_всего"] = int(len(y_ref) + len(y_test))
    export_adapt.save_json(loeo, config.LOEO_PATH)
    log.info(f"   сохранено: {config.LOEO_PATH}")

    log.info("6) сохранение артефактов")
    export_adapt.save_model(winner["_model"], config.MODEL_PATH)
    export_adapt.save_model(reference["_model"], BASELINE_MODEL_PATH)
    export_adapt.save_history(winner["_history"], config.HISTORY_PATH)

    cv_summary = aggregate_folds(winner["фолды"])
    export_adapt.save_json(
        {
            "задача": "бинарная классификация",
            "метки": {"0": "норма", "1": "отказ"},
            "зерно": config.SEED,
            "устройство": config.DEVICE,
            "параметров_сети": count_parameters(winner["_model"]),
            "порог_решения": winner["порог_f1"],
            "порог_полнота_ориентированный": winner["порог_f2"],
            "правило_порога": "максимум F-меры на валидации; тест не используется",
            "окно": {"размер": config.WINDOW_SIZE, "шаг": config.STRIDE,
                     "сенсоров": config.N_SENSORS, "признаков": config.N_FEATURES},
            "итоговая_конфигурация": winner["имя"],
            "дополнения": winner["дополнения"],
            "признаков_в_модели": winner["признаков"],
            "окон_обучения": winner["окон_обучения"],
            "тестовая_выборка": final_metrics,
            "тест_порог_0.5": final_at_half,
            "тест_полнота_ориентированный": final_recall_oriented,
            "до_улучшений": base_metrics,
            "тест_по_экспериментам": by_exp,
            "кросс_валидация": cv_summary,
            "фолды": winner["фолды"],
            "проверка_пересборки_окон": rebuild,
            "проверка_датасета_с_шагом_5": step5,
            "честная_оценка_по_одному_эксперименту": {
                "фолдов": loeo["фолдов"],
                "конфигурация": loeo["конфигурация"],
                "правило_порога": loeo["правило_порога"],
                "сводка": loeo["сводка"],
                "сводка_при_пороге_0.5": loeo["сводка_0.5"],
                "объединённо": loeo["объединённо_подобранный_порог"],
                "эталонный_тест": loeo["эталонный_тест"],
            },
        },
        config.METRICS_PATH,
    )
    export_adapt.save_json(
        {
            "сравнение": {
                "Итоговая модель": final_metrics,
                "Модель до улучшений": base_metrics,
                "Модель из первого отчёта": original["тест_порог_0.5"],
                "Логистическая регрессия": logreg_item["тест_порог_f1"],
                "Постоянный ответ «норма»": majority,
                "Постоянный ответ «отказ»": always_fault,
            },
            "пороги": {
                "итоговая модель": winner["порог_f1"],
                "модель до улучшений": 0.5,
                "модель из первого отчёта": 0.5,
                "логистическая регрессия": logreg_item["порог_f1"],
                "постоянный ответ «норма»": 0.5,
                "постоянный ответ «отказ»": 0.5,
            },
            "разложение_прироста": [
                {"этап": item["этап"], "метрики": item["метрики"]} for item in ladder
            ],
            "условия": "одно разделение по экспериментам; порог выбран по валидации "
                       "для каждой модели отдельно",
        },
        config.BASELINES_PATH,
    )
    export_adapt.save_json(
        {
            "проверка_пересборки_окон": rebuild,
            "победитель": winner["имя"],
            "правило_выбора": search["правило_выбора"],
            "вклад_методов": contributions,
            "порядок_обучения_из_первого_отчёта": {
                "контрольные_эксперименты": original["контрольные_эксперименты"],
                "эпох": original["эпох"],
                "порог_f1": original["порог_f1"],
                "валидация_порог_0.5": original["валидация_порог_0.5"],
                "тест_порог_0.5": original["тест_порог_0.5"],
                "тест_порог_f1": original["тест_порог_f1"],
            },
            "разложение_прироста": [
                {"этап": item["этап"], "метрики": item["метрики"]} for item in ladder
            ],
            "логистическая_регрессия": {
                "порог_f1": logreg_item["порог_f1"],
                "порог_f2": logreg_item["порог_f2"],
                "валидация": logreg_item["валидация_порог_f1"],
                "тест": logreg_item["тест_порог_f1"],
                "тест_порог_0.5": logreg_item["тест_порог_0.5"],
            },
        },
        config.REPORTS_DIR / "improvement_adapt.json",
    )
    export_adapt.save_json(
        {
            "обучение_эксперименты": train.experiments,
            "тест_эксперименты": test.experiments,
            "пересечение": sorted(set(train.experiments) & set(test.experiments)),
        },
        config.SPLIT_PATH,
    )

    log.info("7) графики")
    figures = {
        "pr_curve": plots_adapt.plot_pr_curve(
            y_ref, winner["_oof"], winner["порог_f1"], name="04_pr_curve.png"
        ),
        "confusion": plots_adapt.plot_confusion(
            y_test, (prob_test >= winner["порог_f1"]).astype(int),
            name="03_confusion_matrix.png",
        ),
        "roc": plots_adapt.plot_roc(y_test, prob_test, name="05_roc_curve.png"),
        "training": plots_adapt.plot_training(winner["_history"], name="06_training_curves.png"),
        "model_comparison": plots_adapt.plot_model_comparison(
            comparison, name="08_model_comparison.png"
        ),
        "baseline_comparison": plots_adapt.plot_baseline_comparison(comparison),
        "loeo_folds": plots_adapt.plot_loeo_folds(
            loeo["фолды"], loeo["сводка"]["f1_среднее"], name="12_loeo_folds.png"
        ),
        "class_balance": plots_adapt.plot_class_balance(
            train.y, test.y, name="01_class_balance.png"
        ),
        "example_window": plots_adapt.plot_example_window(
            train.X, train.y, train.features, name="02_example_window.png"
        ),
    }
    for key, filename in EXTRA_FIGURES.items():
        path = config.FIGURES_DIR / filename
        if path.exists():
            figures[key] = path
    for name, path in figures.items():
        log.info(f"   {name}: {Path(path).name}")

    log.info("8) отчёт")
    diagnostics = None
    diag_path = config.REPORTS_DIR / "loeo_diagnostics.json"
    if diag_path.exists():
        diagnostics = export_adapt.load_json(diag_path)

    ctx = {
        "title": "Лабораторная работа №2 — Классификация состояния спутника (ADAPT)",
        "subtitle": "Обнаружение отказа по окну телеметрии: нейронная сеть, "
                    "подбор порога и сравнение с простыми моделями",
        "seed": config.SEED,
        "n_train_exp": len(train.experiments),
        "n_test_exp": len(test.experiments),
        "train": train.summary(),
        "test": test.summary(),
        "checks": checks,
        "architecture": winner["архитектура"],
        "mlp_metrics": final_metrics,
        "before": base_metrics,
        "cv": cv_summary,
        "folds": winner["фолды"],
        "test_by_experiment": by_exp,
        "comparison": comparison,
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

    html_report.write_report(html_report.build_report(ctx), config.REPORT_HTML)
    log.info(f"   отчёт: {config.REPORT_HTML}")
    log.info("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
