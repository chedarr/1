# -*- coding: utf-8 -*-
"""Собирает HTML-отчёт по лабораторной работе №4."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.html_report import build


def main() -> int:
    path = build()
    print(f"ИТОГ: отчёт собран: {path} ({path.stat().st_size} Б)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
