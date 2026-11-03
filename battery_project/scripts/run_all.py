"""Запуск всего пайплайна лабораторной работы №1."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import run_pipeline

if __name__ == "__main__":
    run_pipeline()
    print("Готово.")