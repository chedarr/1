# -*- coding: utf-8 -*-
"""Проверка состава проекта: файлы, артефакты, чёрный список терминов, офлайн.

Запуск:  .venv\\Scripts\\python.exe scripts\\check_project.py
Код возврата 1 — проект неполон или найден запрещённый термин.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "conftest.py",
    "src/config.py",
    "src/loader_adapt.py",
    "src/split_adapt.py",
    "src/model_adapt.py",
    "src/metrics_adapt.py",
    "src/validate_adapt.py",
    "src/plots_adapt.py",
    "src/export_adapt.py",
    "src/html_report.py",
    "src/report_text.py",
    "src/main_adapt.py",
    "src/likbez_adapt.py",
    "src/experiments_adapt.py",
    "src/loeo_adapt.py",
    "scripts/run_all.py",
    "scripts/check_leakage.py",
    "scripts/check_project.py",
    "scripts/step6_build_step5.py",
    "scripts/diagnose_loeo.py",
    "scripts/make_extra_figures.py",
    "scripts/rebuild_report.py",
    "tests/test_experiments.py",
    "tests/test_loeo.py",
    "tests/test_diagnostics.py",
    "docs/lab2_likbez.md",
    "data/processed/adapt_train_step5.parquet",
    "reports/lab2_report.html",
    "reports/model_adapt.keras",
    "reports/metrics_adapt.json",
    "reports/summary_baselines.json",
    "reports/improvement_adapt.json",
    "reports/loeo_adapt.json",
    "reports/loeo_diagnostics.json",
    "reports/lab2_likbez.html",
    "reports/style.css",
]

NETWORK_MARKERS = ("requests.", "urllib.request", "http.client", "socket.socket", "https://")


def main() -> int:
    ok = True

    for rel in REQUIRED_FILES:
        path = config.BASE_DIR / rel
        if not path.exists():
            ok = False
            print(f"[НЕТ] {rel}")
    print(f"[OK] проверено файлов: {len(REQUIRED_FILES)}")

    for fig in sorted(config.FIGURES_DIR.glob("*.png")):
        print(f"[OK] график {fig.name} ({fig.stat().st_size // 1024} КБ)")
    if not list(config.FIGURES_DIR.glob("*.png")):
        ok = False
        print("[НЕТ] графики в reports/figures")

    # чёрный список терминов — только в тексте отчёта и ликбеза
    for rel in ("reports/lab2_report.html", "reports/lab2_likbez.html", "docs/lab2_likbez.md"):
        path = config.BASE_DIR / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hits = [t for t in config.BANNED_IN_TEXT if t in text]
        if hits:
            ok = False
            print(f"[ТЕРМИН] {rel}: {hits}")
        else:
            print(f"[OK] {rel}: запрещённых терминов нет")

    # офлайн: в исходниках не должно быть сетевых вызовов.
    # Файл, в котором перечислены сами маркеры, из проверки исключается.
    for py in sorted(list((config.BASE_DIR / "src").glob("*.py"))
                     + list((config.BASE_DIR / "scripts").glob("*.py"))):
        if py.name == Path(__file__).name:
            continue
        text = py.read_text(encoding="utf-8")
        hits = [m for m in NETWORK_MARKERS if m in text]
        if hits:
            ok = False
            print(f"[СЕТЬ] {py.name}: {hits}")
    print("[OK] сетевых вызовов в коде нет")

    report = config.REPORT_HTML
    if report.exists():
        html = report.read_text(encoding="utf-8")
        external = re.findall(r'(?:src|href)="(https?:[^"]+)"', html)
        if external:
            ok = False
            print(f"[СЕТЬ] внешние ссылки в отчёте: {external[:3]}")
        else:
            print("[OK] отчёт самодостаточен: внешних ссылок нет")

    print("ИТОГ: проект в порядке" if ok else "ИТОГ: есть замечания")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
