# -*- coding: utf-8 -*-
"""Ликбез: преобразование разметки и готовые файлы."""
from __future__ import annotations

import re

from src import config
from src.likbez_adapt import build_html, md_to_html


def test_headings_and_paragraphs():
    html = md_to_html("# Заголовок\n\nОбычный текст.")
    assert "<h1>Заголовок</h1>" in html
    assert "<p>Обычный текст.</p>" in html


def test_bullet_list():
    html = md_to_html("- первый\n- второй\n\nпосле")
    assert html.count("<li>") == 2
    assert "</ul>" in html
    assert "<p>после</p>" in html


def test_ordered_list():
    html = md_to_html("1. один\n2. два")
    assert "<ol>" in html and html.count("<li>") == 2


def test_bold_and_code():
    html = md_to_html("Это **важно** и `код`.")
    assert "<strong>важно</strong>" in html
    assert "<code>код</code>" in html


def test_bold_may_span_line_break():
    html = md_to_html("работа **на тех записях,\nкоторые видит** — и всё.")
    assert "<strong>на тех записях, которые видит</strong>" in html
    assert "**" not in html


def test_table():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    html = md_to_html(md)
    assert '<div class="table-wrap">' in html
    assert '<table class="data">' in html
    assert "<th>A</th>" in html
    assert '<td class="num">1</td>' in html


def test_numeric_cells_are_marked():
    md = "| Метрика | Значение |\n|---|---|\n| accuracy | 0.903 |\n| комментарий | текст |"
    html = md_to_html(md)
    assert '<td class="num">0.903</td>' in html
    assert "<td>текст</td>" in html


def test_build_html_is_offline_page():
    html = build_html("# Тема\n\nТекст.", "Тема")
    assert html.startswith("<!DOCTYPE html>")
    assert 'href="style_likbez.css"' in html
    assert '<article class="content">' in html
    assert '<header class="doc-header">' in html
    assert '<nav class="toc"' in html
    assert "<main>" in html
    assert '<footer class="doc-footer">' in html
    assert "http://" not in html and "https://" not in html


def test_real_likbez_source_converts_cleanly():
    md = config.LIKBEZ_MD.read_text(encoding="utf-8")
    html = build_html(md, "Ликбез")
    assert "<h1>" in html
    assert html.count('<section id="s') == 9
    for number in range(1, 10):
        assert f'id="s{number}"' in html
        assert f'href="#s{number}"' in html
    assert '<table class="data">' in html
    assert "figures/12_loeo_folds.png" in html
    # разметка не должна остаться «сырой»
    assert not re.search(r"^\s*##\s", html, flags=re.MULTILINE)
    assert "**" not in html
    for term in config.BANNED_IN_TEXT:
        assert term not in html
