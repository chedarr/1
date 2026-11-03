"""Собрать HTML-отчёт из Markdown-исходника (Lab. №1).

Пример:
    python -X utf8 scripts/build_html_report.py

Читает:  reports/lab1_report.md
Стили:   reports/style.css
Результат: reports/lab1_report.html
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LAB1_REPORT_HTML_PATH, LAB1_REPORT_PATH, STYLE_CSS_PATH
from src.html_report import build_html


if __name__ == "__main__":
    build_html(LAB1_REPORT_PATH, STYLE_CSS_PATH, LAB1_REPORT_HTML_PATH)
    print("Готово: reports/lab1_report.html")