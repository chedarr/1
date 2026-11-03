# -*- coding: utf-8 -*-
"""Конвертирует docs/lab4_likbez.md в самодостаточный HTML."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import markdown

from src import config
from src.html_report import CSS


def main() -> int:
    source = config.DOCS / "lab4_likbez.md"
    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"]
    )
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лабораторная работа №4 — ликбез</title>
<style>{CSS}</style>
</head>
<body>
<main>
{body}
<footer>Лабораторная работа №4 — многомерный прогноз временного ряда (GOCE, LSTM).</footer>
</main>
</body>
</html>
"""
    out = config.REPORTS / "lab4_likbez.html"
    out.write_text(html, encoding="utf-8")
    print(f"ликбез: {out} ({out.stat().st_size} Б)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
