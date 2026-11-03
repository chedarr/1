# -*- coding: utf-8 -*-
"""Собрать HTML-версию документа из markdown: docs/<name>.md → docs/<name>.html

    docs/lab1_likbez.md   --(pandoc)--> docs/lab1_likbez.html
    docs/lab1_defense.md  --(pandoc)--> docs/lab1_defense.html

Текст берётся из markdown без правок, статичная обвязка (шапка, оглавление,
подвал) добавляется скриптом. Стили — локальные docs/style_likbez.css,
внешних ресурсов нет.

Использование:
    py -X utf8 scripts/build_doc_html.py                      # оба документа
    py -X utf8 scripts/build_doc_html.py lab1_defense         # только один
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / "docs"
STYLE = "style_likbez.css"

# подзаголовок и подвал зависят от документа
META = {
    "lab1_likbez": {
        "subtitle": "Документ-разбор лабораторной №1: предмет, данные, пайплайн, "
                    "модели и базовые предсказатели, честные результаты и открытые вопросы",
        "footer": """  <p>Все числа — из <code>data/processed/</code> и <code>data/processed/metrics_*.json</code>,
     округление только при переносе в текст. Стиль документа: <code>style_likbez.css</code>.</p>
  <p>Лабораторная работа №1 — регрессия нейронными сетями на табличных данных,
     NASA PCoE Battery Dataset (34 батареи).</p>""",
    },
    "lab1_defense": {
        "subtitle": "Короткая версия для устной защиты: рассказ по шагам, ключевые "
                    "числа, три ответа наизусть и словарик всех терминов",
        "footer": """  <p>Все числа — из <code>data/processed/</code> и <code>data/processed/metrics_*.json</code>.
     Полная версия работы: <code>docs/lab1_likbez.md</code>, <code>reports/lab1_report.md</code>.</p>
  <p>Лабораторная работа №1 — регрессия нейронными сетями на табличных данных,
     NASA PCoE Battery Dataset (34 батареи).</p>""",
    },
}

DEFAULT_META = {"subtitle": "", "footer": ""}


def pandoc_fragment(md_text: str) -> str:
    """markdown → HTML-фрагмент (pandoc, без внешних ресурсов)."""
    proc = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html5"],
        input=md_text, capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(f"pandoc завершился с ошибкой: {proc.stderr}")
    return proc.stdout


def split_document(fragment: str) -> tuple[str, str, list[tuple[str, str]]]:
    """Отделить h1 и вводный абзац, разбить текст на секции по <h2>.

    Возвращает (title, lead_html, [(заголовок, html_секции), ...]).
    """
    head = re.search(r"<h1[^>]*>(.*?)</h1>\s*(<p>.*?</p>)?", fragment, re.S)
    if not head:
        raise SystemExit("в markdown нет h1 — проверьте начало файла")
    title = re.sub(r"<[^>]+>", "", head.group(1))
    title = re.sub(r"\s+", " ", title).strip()
    lead = (head.group(2) or "").strip()
    body = fragment[head.end():]

    # горизонтальные линейки-разделители в HTML не нужны
    body = re.sub(r"<hr\s*/?>\s*", "", body)

    parts = re.split(r"(?=<h2)", body)
    preamble = parts[0].strip()
    if preamble:
        raise SystemExit(f"неожиданный текст до первого <h2>: {preamble[:120]!r}")

    sections: list[tuple[str, str]] = []
    for part in parts[1:]:
        part = part.strip()
        head2 = re.search(r"<h2[^>]*>(.*?)</h2>", part, re.S).group(1)
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", head2)).strip()
        label = re.sub(r"^\d+\.\s*", "", text)  # "4. Модели" → "Модели"
        html = re.sub(r"<h2[^>]*>", "<h2>", part, count=1)
        # обёртка таблиц для горизонтальной прокрутки
        html = html.replace("<table>", '<div class="table-wrap"><table class="data">')
        html = html.replace("</table>", "</table></div>")
        sections.append((label, f'<section id="s{len(sections) + 1}">{html}</section>'))
    return title, lead, sections


def build_html(title: str, lead: str, meta: dict, sections: list[tuple[str, str]]) -> str:
    nav = "\n".join(
        f'    <li><a href="#s{i}">{label}</a></li>'
        for i, (label, _) in enumerate(sections, start=1)
    )
    body = "\n".join(html for _, html in sections)
    subtitle = f'  <p class="subtitle">{meta["subtitle"]}</p>\n' if meta["subtitle"] else ""
    footer = f"\n<footer class=\"doc-footer\">\n{meta['footer']}\n</footer>" if meta["footer"] else ""
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{STYLE}">
</head>
<body>
<article class="content">

<header class="doc-header">
  <h1>{title}</h1>
{subtitle}  <p class="lead">{lead}</p>
</header>

<nav class="toc" aria-label="Оглавление">
  <p class="toc-title">Содержание</p>
  <ol>
{nav}
  </ol>
</nav>

<main>{body}</main>{footer}

</article>
</body>
</html>
"""


def build_one(name: str) -> str:
    md_path = DOCS / f"{name}.md"
    if not md_path.exists():
        raise SystemExit(f"нет файла {md_path}")
    meta = META.get(name, DEFAULT_META)
    title, lead, sections = split_document(pandoc_fragment(md_path.read_text(encoding="utf-8")))
    html = build_html(title, lead, meta, sections)
    out = DOCS / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"OK: {out} ({len(html)} символов, секций: {len(sections)})")
    return out.name


def main(argv: list[str]) -> int:
    names = argv or list(META)
    for name in names:
        build_one(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
