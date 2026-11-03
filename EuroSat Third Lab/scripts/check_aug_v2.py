# -*- coding: utf-8 -*-
"""Проверка артефактов aug v2 без обучения и без изменения файлов."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT / "reports"
AUG = REPORTS / "control_patience_max_aug"

sys.path.insert(0, str(PROJECT))

from src.augment import build_augmentation  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)
    print("OK:", message)


def main() -> int:
    v1 = load(REPORTS / "control_patience_max" / "metrics_eurosat_patience_max.json")
    v2 = load(AUG / "metrics_eurosat_patience_max_aug.json")
    summary = load(REPORTS / "summary_baselines.json")

    check(abs(v1["test"]["accuracy"] - 0.9079012345679013) < 1e-12, "aug v1 accuracy не изменена")
    check(abs(v1["test"]["macro_f1"] - 0.9054564304020287) < 1e-12, "aug v1 macro-F1 не изменён")
    check(v2["params"] == 549626, "aug v2 params = 549626")
    check(v2["config"]["patience"] == 50, "patience = 50")
    check(v2["config"]["epochs"] == 50, "epochs = 50")
    check(v2["config"]["batch_size"] == 64, "batch_size = 64")
    check(abs(v2["config"]["lr"] - 1e-3) < 1e-12, "lr = 1e-3")
    check(v2["config"]["seed"] == 42, "seed = 42")
    check("brightness" in v2["config"]["augmentation"].lower(), "augmentation содержит brightness")
    check("contrast" in v2["config"]["augmentation"].lower(), "augmentation содержит contrast")
    check(v2.get("augmentation_version") == "aug_v2", "augmentation_version = aug_v2")

    test = v2["test"]
    check(test["n_samples"] == 4050, "n_test = 4050")
    check(sum(test["confusion_matrix"][i][j] for i in range(10) for j in range(10)) == 4050,
          "confusion matrix sum = 4050")
    check(sum(r["support"] for r in test["per_class"]) == 4050, "support sum = 4050")
    check(len(test["per_class"]) == 10, "10 классов в per_class")
    check(len(v2["history"]["epoch"]) == v2["total_epochs_run"], "history length = total_epochs_run")
    check(1 <= v2["best_epoch"] <= v2["total_epochs_run"], "best_epoch в диапазоне")
    check(v2["training_time_sec"] > 0, "training_time_sec > 0")

    layers = [layer.__class__.__name__ for layer in build_augmentation(42).layers]
    expected = ["RandomFlip", "RandomRotation", "RandomZoom", "RandomBrightness", "RandomContrast"]
    check(layers == expected, f"порядок аугментации: {layers}")

    check(abs(summary["cnn"]["accuracy"] - 0.6898765432098766) < 1e-12,
          "summary_baselines.cnn.accuracy остался 0.6899")
    check(abs(summary["cnn"]["macro_f1"] - 0.6702692721270571) < 1e-12,
          "summary_baselines.cnn.macro_f1 остался 0.6703")

    figures = [
        "confusion_matrix_aug_v2.png", "training_curves_aug_v2.png",
        "per_class_f1_aug_v2.png", "baseline_comparison_aug_v2.png",
        "aug_v1_vs_v2_per_class.png", "aug_v1_vs_v2_metrics.png",
    ]
    for name in figures:
        path = AUG / "figures" / name
        check(path.exists() and path.stat().st_size > 10_000, f"рисунок {name}")

    report = REPORTS / "lab3_report_aug_v2.html"
    check(report.exists() and report.stat().st_size > 100_000, "отчёт lab3_report_aug_v2.html")
    html = report.read_text(encoding="utf-8")
    for heading in ["1. Титул", "2. Данные", "3. Методология", "4. Результаты aug v2",
                    "5. Сравнение aug v1 vs aug v2", "6. Сравнение с baseline",
                    "7. Проверка протокола", "8. Анализ", "9. Ограничения",
                    "10. Выводы", "11. Приложение"]:
        check(heading in html, f"раздел: {heading}")
    check("ограничение протокола, а не данных" in html.lower(),
          "формулировка «ограничение протокола, а не данных»")
    check("test не полностью независим" not in html.lower(),
          "нет формулировки «test не полностью независим»")
    check(html.count("<img ") == 10, "10 встроенных изображений")
    check("data:image/png;base64," in html, "изображения base64")
    check(not re.search(r'<img[^>]+src="(?!data:)', html), "нет внешних img src")

    print()
    print("ИТОГ: aug v2 проверен; accuracy=%.6f macro_f1=%.6f best_epoch=%d" %
          (test["accuracy"], test["macro_f1"], v2["best_epoch"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
