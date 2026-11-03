# -*- coding: utf-8 -*-
"""Полный прогон проекта по шагам (без обучения, если не указан --with-training)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

STEPS = [
    ("prepare_data", ["scripts/prepare_data.py"]),
    ("baselines", ["scripts/run_baselines.py"]),
    ("summarize", ["scripts/summarize.py"]),
    ("figures", ["scripts/make_figures.py"]),
    ("report", ["scripts/build_report.py"]),
    ("likbez", ["scripts/build_likbez.py"]),
    ("leakage", ["scripts/check_leakage.py"]),
]

TRAIN = [
    ["scripts/run_experiment.py", "--name", "A", "--save-model"],
    ["scripts/run_experiment.py", "--name", "B"],
    ["scripts/run_experiment.py", "--name", "C"],
    ["scripts/run_experiment.py", "--name", "D"],
    ["scripts/run_experiment.py", "--name", "E"],
    ["scripts/run_experiment.py", "--name", "F"],
    ["scripts/run_experiment.py", "--name", "G"],
    ["scripts/run_experiment.py", "--name", "E0"],
    ["scripts/run_experiment.py", "--name", "lb72"],
    ["scripts/run_experiment.py", "--name", "lb288"],
    ["scripts/run_experiment.py", "--name", "hz72"],
    ["scripts/run_experiment.py", "--name", "hz28"],
]


def run(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===")
    result = subprocess.run([PY, *args], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"шаг {name} завершился с кодом {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-training", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if args.with_training:
        for cmd in TRAIN:
            run(f"обучение {cmd[2]}", cmd)

    for name, cmd in STEPS:
        if args.only and name not in args.only:
            continue
        run(name, cmd)
    print("\nИТОГ: конвейер завершён")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
