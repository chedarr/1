# -*- coding: utf-8 -*-
"""Честная оценка «по одному эксперименту» (Leave-One-Experiment-Out) по всем 16 записям.

Зачем это нужно. Итоговое число отчёта получено на четырёх тестовых экспериментах,
и оно ничего не говорит о разбросе между записями. Оценка «оставить один эксперимент»
прогоняет процедуру обучения шестнадцать раз: каждая запись по очереди исключается,
модель обучается на остальных пятнадцати, а порог решения подбирается на одном
эксперименте, отведённом внутри этих пятнадцати.

Честность. Этот расчёт НЕ участвует в выборе архитектуры и порога: он служит только
мерой устойчивости. Для каждой записи фиксируется своё зерно генератора, поэтому
результат воспроизводим. Модель, которая поставляется как итоговая, обучается на
двенадцати обучающих экспериментах и о четырёх тестовых не знает; поэтому оценка по
шестнадцати описывает процедуру в целом, а не сохранённую модель.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.experiments_adapt import (
    Config,
    architecture_text,
    best_thresholds,
    build_features,
    build_model,
    class_weights,
    fit_with_early_stopping,
    threshold_curve,
)
from src.metrics_adapt import compute_metrics
from src.split_adapt import holdout_by_experiment

EXP_COL = config.META_PREFIX + "experiment"
THRESHOLD_RULE = (
    "порог подбирается внутри каждого фолда по одному контрольному эксперименту, "
    "отведённому из пятнадцати обучающих; тестовая запись фолда в подборе не участвует"
)
WARNING = (
    "оценка по одному эксперименту не используется для выбора модели или порога и "
    "служит только мерой разброса между записями"
)


def fold_seed(index: int) -> int:
    """Зерно генератора для фолда: фиксировано и воспроизводимо."""
    return config.SEED + index


def loeo_splits(meta: pd.DataFrame) -> list[dict]:
    """Шестнадцать фолдов: по одному исключённому эксперименту в каждом.

    Возвращаются позиционные индексы строк, а не метки, чтобы вызывающий код
    мог обращаться к массивам окон напрямую.
    """
    groups = meta[EXP_COL].to_numpy()
    experiments = sorted(set(groups.tolist()), key=lambda e: (len(str(e)), str(e)))
    splits: list[dict] = []
    for i, exp in enumerate(experiments, start=1):
        holdout = np.flatnonzero(groups == exp)
        rest = np.flatnonzero(groups != exp)
        if holdout.size == 0 or rest.size == 0:
            raise ValueError(f"эксперимент {exp}: пустая часть разбиения")
        splits.append(
            {
                "фолд": i,
                "эксперимент": str(exp),
                "индексы_контроля": holdout,
                "индексы_обучения": rest,
                "зерно": fold_seed(i),
            }
        )
    return splits


def summarize(folds: list[dict], suffix: str = "") -> dict:
    """Среднее, отклонение, минимум и максимум по фолдам."""
    out: dict = {"фолдов": len(folds)}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        name = key + (suffix or "")
        values = np.array([f[name] for f in folds if name in f], dtype=float)
        if values.size:
            out[f"{key}{suffix}_среднее"] = float(values.mean())
            out[f"{key}{suffix}_отклонение"] = float(values.std(ddof=0))
            out[f"{key}{suffix}_минимум"] = float(values.min())
            out[f"{key}{suffix}_максимум"] = float(values.max())
    f1 = np.array([f["f1" + suffix] for f in folds], dtype=float)
    out["провалов_ниже_0.5"] = int((f1 < 0.5).sum())
    out["полных_попаданий"] = int((f1 >= 0.999).sum())
    out["отказов_всего"] = int(sum(f["верно_отказ"] + f["пропуск_отказа"] for f in folds))
    return out


def run_loeo(
    cfg: Config,
    x_all: np.ndarray,
    y_all: np.ndarray,
    meta_all: pd.DataFrame,
    epochs: int = 120,
    logger=None,
) -> dict:
    """Проводит оценку по одному эксперименту и собирает честные метрики."""
    log = logger.info if logger is not None else (lambda *a, **k: None)
    features = build_features(x_all, cfg.blocks)
    splits = loeo_splits(meta_all)
    folds: list[dict] = []
    pooled_prob = np.full(len(y_all), np.nan, dtype=float)

    for item in splits:
        tr = item["индексы_обучения"]
        te = item["индексы_контроля"]
        inner = meta_all.iloc[tr].reset_index(drop=True)
        idx_fit, idx_es, es_experiments = holdout_by_experiment(
            inner, 1, seed=item["зерно"]
        )
        fit_idx = tr[idx_fit]
        es_idx = tr[idx_es]

        scaler = StandardScaler().fit(features[fit_idx])
        model, history = fit_with_early_stopping(
            cfg,
            scaler.transform(features[fit_idx]),
            y_all[fit_idx],
            scaler.transform(features[es_idx]),
            y_all[es_idx],
            epochs,
            seed=item["зерно"],
        )
        prob_es = model.predict(scaler.transform(features[es_idx]), verbose=0).ravel()
        picks = best_thresholds(threshold_curve(y_all[es_idx], prob_es))
        threshold = float(picks["f1"]["порог"])
        epochs_used = len(history.history["loss"])

        scaler_full = StandardScaler().fit(features[tr])
        final = build_model(cfg, features.shape[1], item["зерно"])
        final.fit(
            scaler_full.transform(features[tr]),
            y_all[tr],
            epochs=epochs_used,
            batch_size=32,
            verbose=0,
            class_weight=class_weights(y_all[tr], cfg.class_weight),
        )
        prob = final.predict(scaler_full.transform(features[te]), verbose=0).ravel()
        pooled_prob[te] = prob

        chosen = compute_metrics(y_all[te], prob, threshold)
        at_half = compute_metrics(y_all[te], prob, 0.5)
        row = {
            "фолд": item["фолд"],
            "эксперимент": item["эксперимент"],
            "зерно": item["зерно"],
            "объектов": chosen["объектов"],
            "порог": threshold,
            "контроль_порога": es_experiments,
            "эпох": epochs_used,
            "f1_в_пороге_0.5": at_half["f1"],
        }
        row.update({k: chosen[k] for k in
                    ("accuracy", "precision", "recall", "f1", "roc_auc",
                     "верно_норма", "ложная_тревога", "пропуск_отказа", "верно_отказ")
                    if k in chosen})
        row.update({f"{k}_0.5": at_half[k] for k in
                    ("accuracy", "precision", "recall", "f1")
                    if k in at_half})
        folds.append(row)
        log(
            "   фолд %2d: эксперимент %-4s окон %2d | F1 %.3f (порог %.3f) | "
            "F1 при 0.5 %.3f | полнота %.3f",
            row["фолд"], row["эксперимент"], row["объектов"], row["f1"],
            row["порог"], row["f1_в_пороге_0.5"], row["recall"],
        )

    y_all = np.asarray(y_all).astype(int).ravel()
    pooled = compute_metrics(y_all, pooled_prob, 0.5)
    # объединённая оценка с учётом индивидуальных порогов: считается по
    # решениям, чтобы ни одно окно не «перевешивало» из-за размера записи
    decisions = np.zeros(len(y_all), dtype=int)
    for row in folds:
        mask = (meta_all[EXP_COL] == row["эксперимент"]).to_numpy()
        decisions[mask] = (pooled_prob[mask] >= row["порог"]).astype(int)
    pooled_chosen = compute_metrics(y_all, decisions.astype(float), 0.5)
    # площадь под кривой ошибок по решениям не считается: знак «<» или «>=»
    # не задаёт вероятности, и такое число вводило бы в заблуждение
    pooled_chosen.pop("roc_auc", None)
    pooled_chosen.pop("порог", None)
    pooled_chosen["порог_средний"] = float(np.mean([f["порог"] for f in folds]))

    summary = summarize(folds)
    log(
        "   сводка по 16 фолдам: F1 %.3f ± %.3f (от %.3f до %.3f), полнота %.3f, "
        "точность %.3f",
        summary["f1_среднее"], summary["f1_отклонение"],
        summary["f1_минимум"], summary["f1_максимум"],
        summary["recall_среднее"], summary["precision_среднее"],
    )
    return {
        "конфигурация": cfg.name,
        "архитектура": architecture_text(cfg),
        "фолдов": len(folds),
        "правило_порога": THRESHOLD_RULE,
        "предупреждение": WARNING,
        "фолды": folds,
        "сводка": summary,
        "сводка_0.5": summarize(folds, "_0.5"),
        "объединённо_порог_0.5": pooled,
        "объединённо_подобранный_порог": pooled_chosen,
    }
