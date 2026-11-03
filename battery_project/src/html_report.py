"""Markdown → HTML-отчёт (Лаб. №1): конвертация через pandoc.

Markdown конвертируется в HTML5 командой pandoc (--from=markdown --to=html5,
как в ТЗ). Лёгкая обвязка собирает страницу:
    header (титул, автор, дата — из h1 и первой таблицы md)
    nav    (оглавление по разделам h2, id из pandoc-вывода)
    main   (контент от pandoc)
    footer (источник данных, ссылка на код)
Все стили — локальные reports/style.css (никаких CDN).
Кодировка — UTF-8 на всех этапах (чтение markdown, подпроцесс pandoc, запись HTML).
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

# локальная установка pandoc вне PATH (C:\Program Files\Pandoc)
PANDOC = shutil.which("pandoc") or r"C:\Program Files\Pandoc\pandoc.exe"


def _pandoc_fragment(md_text: str) -> str:
    """Markdown → HTML5-фрагмент (обычный вывод pandoc, без --standalone)."""
    if not Path(PANDOC).exists():
        raise RuntimeError(f"pandoc не найден: {PANDOC}")
    proc = subprocess.run(
        [PANDOC, "-f", "markdown", "-t", "html5"],
        input=md_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pandoc завершился с кодом {proc.returncode}:\n{proc.stderr}"
        )
    return proc.stdout


def _collapse(text: str) -> str:
    """Один пробел вместо переносов строк (pandoc переносит длинные строки)."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_title_meta(soup: BeautifulSoup) -> tuple[str, dict[str, str]]:
    """Извлечь (title, meta) из h1 и первой (титульной) таблицы фрагмента."""
    title = ""
    h1 = soup.find("h1")
    if h1 is not None:
        title = _collapse(h1.get_text(" ", strip=True))
        h1.decompose()  # h1 выносим в header

    meta: dict[str, str] = {}
    first_tbl = soup.find("table")
    if first_tbl is not None:
        # первая таблица — «титульная» (Дисциплина/Автор/Дата/Инструменты)
        for tr in first_tbl.find_all("tr"):
            cells = [_collapse(td.get_text(" ", strip=True))
                     for td in tr.find_all(["th", "td"])]
            if len(cells) >= 2:
                meta[cells[0].lower()] = cells[1]
        first_tbl.decompose()  # в main не выводим (уходит в header)
    return title, meta


def build_html(md_path, css_path: Optional[Path] = None,
               html_path: Optional[Path] = None) -> Path:
    md_path = Path(md_path)
    css_path = Path(css_path) if css_path else md_path.parent / "style.css"
    html_path = Path(html_path) if html_path else md_path.with_suffix(".html")

    md_text = md_path.read_text(encoding="utf-8")
    fragment = _pandoc_fragment(md_text)

    soup = BeautifulSoup(fragment, "html.parser")
    title, meta = _parse_title_meta(soup)
    if not title:
        title = md_path.stem

    author = meta.get("автор", "студент")
    date = meta.get("дата", "")

    # nav по разделам (h2; id и заголовок — из pandoc-вывода)
    nav_items = ['<li><a href="#top">1. Титул</a></li>']
    for h2 in soup.find_all("h2"):
        hid = h2.get("id") or ""
        if not hid:
            hid = f"sec-{len(nav_items) + 1:02d}"
            h2["id"] = hid
        label = _collapse(h2.get_text(" ", strip=True))
        nav_items.append(f'<li><a href="#{hid}">{html.escape(label)}</a></li>')

    body = str(soup)

    doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{html.escape(css_path.name)}">
</head>
<body id="top">
<header>
  <h1>{html.escape(title)}</h1>
  <p class="meta">Автор: {html.escape(author)} · Дата: {html.escape(date)} · NASA PCoE Battery</p>
</header>
<nav aria-label="Оглавление">
  <ul>{''.join(nav_items)}</ul>
</nav>
<main>
{body}
</main>
<footer>
  <p>Источник данных: NASA PCoE Battery Dataset (34 батареи, data/raw/*.mat).</p>
  <p>Код: <a href="../README.md">README.md</a> · <code>src/</code> · <code>scripts/</code>
     (воспроизводимость: <code>python scripts/run_all.py</code>,
     <code>python -m pytest tests -q</code>)</p>
</footer>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(doc, encoding="utf-8")
    print(f"HTML-отчёт (pandoc): {html_path}")
    return html_path


if __name__ == "__main__":
    from .config import LAB1_REPORT_HTML_PATH, LAB1_REPORT_PATH, STYLE_CSS_PATH
    build_html(LAB1_REPORT_PATH, STYLE_CSS_PATH, LAB1_REPORT_HTML_PATH)