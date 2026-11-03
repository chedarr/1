# -*- coding: utf-8 -*-
"""Один запуск: обучение, отчёт, ликбез, проверки.

Запуск:  .venv\\Scripts\\python.exe scripts\\run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable

STEPS = [
    ("датасет с частой сеткой окон", [PY, str(ROOT / "scripts" / "step6_build_step5.py")]),
    ("обучение и отчёт", [PY, "-m", "src.main_adapt"]),
    ("диагностика устойчивости", [PY, str(ROOT / "scripts" / "diagnose_loeo.py")]),
    ("схемы и график сравнения с baseline", [PY, str(ROOT / "scripts" / "make_extra_figures.py")]),
    ("пересборка отчёта из артефактов", [PY, str(ROOT / "scripts" / "rebuild_report.py")]),
    ("ликбез", [PY, "-m", "src.likbez_adapt"]),
    ("проверка утечек", [PY, str(ROOT / "scripts" / "check_leakage.py")]),
    ("проверка проекта", [PY, str(ROOT / "scripts" / "check_project.py")]),
    ("автотесты", [PY, "-m", "pytest", "tests", "-q"]),
]


def main() -> int:
    code = 0
    for name, cmd in STEPS:
        print(f"\n=== {name} ===")
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            code = proc.returncode
            print(f"!! шаг «{name}» завершился с кодом {proc.returncode}")
    print("\nвсе шаги выполнены" if code == 0 else "\nесть неуспешные шаги")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
