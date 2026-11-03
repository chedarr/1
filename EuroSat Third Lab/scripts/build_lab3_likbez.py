# -*- coding: utf-8 -*-
"""Сборка HTML-версии ликбеза из docs/lab3_likbez.md (стиль как у отчёта)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import markdown

from build_lab3_report import CSS
from src import config


def main() -> int:
    md_path = config.DOCS / "lab3_likbez.md"
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ликбез — лабораторная №3 (EuroSAT, CNN)</title>
<style>{CSS}</style>
</head>
<body>
<main>
{body}
<footer>Ликбез собран из <code>docs/lab3_likbez.md</code>; числа взяты из JSON-артефактов проекта.</footer>
</main>
</body>
</html>
"""
    out = config.REPORTS / "lab3_likbez.html"
    out.write_text(html, encoding="utf-8")
    print(f"ликбез: {out} ({out.stat().st_size:,} Б)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
