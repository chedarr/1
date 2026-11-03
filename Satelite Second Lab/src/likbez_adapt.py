# -*- coding: utf-8 -*-
"""Ликбез: сборка HTML-версии из markdown-источника.

Поддерживается ограниченный набор разметки: заголовки 1–3 уровня, списки,
таблицы, выделение полужирным, код в обратных кавычках, врезки через «>»
и рисунки вида ``![подпись](путь)``. Этого достаточно для учебного текста,
внешние библиотеки не требуются.

Запуск:  .venv\\Scripts\\python.exe -m src.likbez_adapt
"""
from __future__ import annotations

import html as _html
import re
from pathlib import Path

from src import config

_CSS = """
:root {
  --bg: #f2f5f8;
  --card: #fdfdfd;
  --text: #2c3e50;
  --muted: #5a6c7d;
  --accent: #3498db;
  --accent-2: #1f6fb2;
  --heading: #2c3e50;
  --border: #d0d7de;
  --code-bg: #f4f6f8;
  --code-text: #1b3a5c;
  --zebra: #f6f9fc;
  --radius: 6px;
  --maxw: 940px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  padding: 2rem 1rem 4rem;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", Arial, sans-serif;
  font-size: 16.5px;
  line-height: 1.68;
  -webkit-text-size-adjust: 100%;
}
.content {
  max-width: var(--maxw);
  margin: 0 auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 2.25rem 2.5rem 2.5rem;
  box-shadow: 0 1px 4px rgba(44, 62, 80, 0.08);
}
.doc-header {
  border-bottom: 3px solid var(--accent);
  padding-bottom: 0.9rem;
  margin-bottom: 1.1rem;
}
.doc-header h1 {
  margin: 0 0 0.4rem;
  font-size: 1.9rem;
  line-height: 1.22;
  font-weight: 600;
  color: var(--heading);
}
.subtitle {
  margin: 0 0 0.6rem;
  color: var(--accent-2);
  font-size: 1.02rem;
}
.lead {
  margin: 0.7rem 0 0;
  padding: 0.6rem 0.85rem;
  background: var(--code-bg);
  border-left: 3px solid var(--accent);
  border-radius: 0 4px 4px 0;
  color: var(--muted);
  font-size: 0.95rem;
}
.toc {
  position: sticky;
  top: 0;
  z-index: 5;
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 0.7rem 1.1rem;
  margin: 0 0 1.8rem;
  box-shadow: 0 1px 3px rgba(44, 62, 80, 0.06);
}
.toc-title {
  margin: 0 0 0.35rem;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.toc ol {
  margin: 0;
  padding-left: 1.15rem;
  columns: 2;
  column-gap: 1.4rem;
}
.toc li {
  margin: 0.1rem 0;
  break-inside: avoid;
  font-size: 0.95rem;
  line-height: 1.45;
}
.toc a {
  color: var(--accent-2);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: color 0.12s ease, background-color 0.12s ease;
}
.toc a:hover,
.toc a:focus {
  color: var(--accent);
  background: var(--code-bg);
  border-bottom-color: var(--accent);
}
h2 {
  font-size: 1.42rem;
  font-weight: 600;
  color: var(--heading);
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.3rem;
  margin: 2.2rem 0 0.9rem;
}
h3 {
  font-size: 1.14rem;
  font-weight: 600;
  color: var(--accent-2);
  margin: 1.6rem 0 0.55rem;
}
section > h2:first-child { margin-top: 0.3rem; }
section { scroll-margin-top: 5.5rem; }
p { margin: 0.6rem 0; }
a { color: var(--accent-2); }
a:hover { color: var(--accent); }
strong { color: #22384c; }
ul, ol { margin: 0.6rem 0; padding-left: 1.5rem; }
li { margin: 0.28rem 0; }
li > ul, li > ol { margin: 0.2rem 0; }
code {
  font-family: Consolas, "Cascadia Mono", Menlo, monospace;
  font-size: 0.9em;
  background: var(--code-bg);
  color: var(--code-text);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.08em 0.35em;
  white-space: nowrap;
}
.note {
  background: var(--code-bg);
  border-left: 3px solid var(--accent);
  border-radius: 0 4px 4px 0;
  padding: 0.6rem 0.9rem;
  margin: 0.9rem 0;
}
.table-wrap {
  overflow-x: auto;
  margin: 0.9rem 0 1.3rem;
  max-width: 100%;
}
table {
  width: 100%;
  max-width: 100%;
  border-collapse: collapse;
  font-size: 0.93rem;
  line-height: 1.45;
}
th, td {
  border: 1px solid var(--border);
  padding: 0.42rem 0.65rem;
  text-align: left;
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
}
thead th {
  background: var(--accent);
  color: #ffffff;
  font-weight: 600;
  text-align: left;
  white-space: nowrap;
  border-color: #2d87c4;
}
tbody tr:nth-child(even) { background: var(--zebra); }
tbody tr:hover { background: #eaf4fc; }
td strong { color: #1b3a5c; }
table.data td.num,
table.data th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
figure {
  margin: 1.2rem 0 1.4rem;
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #fff;
}
figcaption {
  margin-top: 0.45rem;
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.45;
}
.doc-footer {
  margin-top: 2.6rem;
  padding-top: 0.9rem;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.9rem;
}
.doc-footer p { margin: 0.3rem 0; }
@media print {
  body { background: #fff; padding: 0; font-size: 11pt; }
  .content { max-width: 100%; border: none; box-shadow: none; padding: 0; }
  .toc { position: static; break-inside: avoid; box-shadow: none; }
  h2, h3 { break-after: avoid; }
  table { break-inside: auto; }
  thead th {
    background: var(--accent) !important;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  code { background: #f4f6f8 !important; }
}
@media (max-width: 700px) {
  body { padding: 0.6rem 0.4rem 2rem; font-size: 15.5px; }
  .content { padding: 1.2rem 1rem 1.6rem; }
  .doc-header h1 { font-size: 1.5rem; }
  .toc { position: static; }
  .toc ol { columns: 1; }
  table { font-size: 0.86rem; }
  th, td { padding: 0.32rem 0.45rem; }
}
"""

_NUM_RE = re.compile(
    r"^[−+\-–]?\d[\d\s.,]*(?:[eE][−+\-]?\d+)?(?:\s*±\s*\d[\d\s.,]*)?\s*%?$"
)


def _escape(text: str) -> str:
    """Экранирует текст для HTML, не трогая разметку markdown."""
    return _html.escape(text, quote=False)


def _attr(text: str) -> str:
    """Экранирует значение атрибута."""
    return _html.escape(text, quote=True)


def _inline(text: str) -> str:
    """Полужирный текст и код внутри строки."""
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _is_separator(line: str) -> bool:
    """Строка-разделитель таблицы вида |---|---|."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|-: ") and "-" in stripped


def _is_numeric(cell: str) -> bool:
    """Числовая ячейка: её нужно выравнивать вправо."""
    plain = re.sub(r"[*`]", "", cell).strip()
    return bool(_NUM_RE.match(plain))


def md_to_html(md: str) -> str:
    """Переводит ограниченный markdown в html-фрагмент."""
    md = md.replace("\r\n", "\n")
    # выделение полужирным может разрываться переносом строки внутри абзаца:
    # стягиваем такие разрывы, чтобы разметка разбиралась построчно
    md = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: "**" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "**",
        md,
        flags=re.S,
    )
    lines = md.split("\n")
    out: list[str] = []
    open_list: str | None = None
    paragraph: list[str] = []

    def close_list() -> None:
        nonlocal open_list
        if open_list:
            out.append(f"</{open_list}>")
            open_list = None

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(paragraph).strip()
            if text:
                out.append(f"<p>{_inline(text)}</p>")
            paragraph.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if not s:
            flush_paragraph()
            close_list()
            i += 1
            continue

        image = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
        if image:
            flush_paragraph()
            close_list()
            alt = image.group(1).strip()
            src = image.group(2).strip()
            out.append(
                f'<figure><img src="{_attr(src)}" alt="{_attr(alt)}">'
                f"<figcaption>{_escape(alt)}</figcaption></figure>"
            )
            i += 1
            continue

        # таблица: строка с «|», следом разделитель
        if s.startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            flush_paragraph()
            close_list()
            header = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head = "".join(f"<th>{_inline(c)}</th>" for c in header)
            body_rows: list[str] = []
            for row in rows:
                cells: list[str] = []
                for cell in row:
                    cls = ' class="num"' if _is_numeric(cell) else ""
                    cells.append(f"<td{cls}>{_inline(cell)}</td>")
                body_rows.append("<tr>" + "".join(cells) + "</tr>")
            out.append(
                '<div class="table-wrap"><table class="data"><thead><tr>'
                + head
                + "</tr></thead><tbody>"
                + "".join(body_rows)
                + "</tbody></table></div>"
            )
            continue

        if s.startswith("### "):
            flush_paragraph()
            close_list()
            out.append(f"<h3>{_inline(s[4:])}</h3>")
        elif s.startswith("## "):
            flush_paragraph()
            close_list()
            out.append(f"<h2>{_inline(s[3:])}</h2>")
        elif s.startswith("# "):
            flush_paragraph()
            close_list()
            out.append(f"<h1>{_inline(s[2:])}</h1>")
        elif s.startswith("> "):
            flush_paragraph()
            close_list()
            out.append(f'<div class="note">{_inline(s[2:])}</div>')
        elif re.match(r"^[-*] ", s):
            flush_paragraph()
            if open_list != "ul":
                close_list()
                out.append("<ul>")
                open_list = "ul"
            out.append(f"<li>{_inline(s[2:])}</li>")
        elif re.match(r"^\d+\. ", s):
            flush_paragraph()
            if open_list != "ol":
                close_list()
                out.append("<ol>")
                open_list = "ol"
            out.append(f"<li>{_inline(re.sub(r'^\d+[.] ', '', s))}</li>")
        else:
            close_list()
            paragraph.append(s)
        i += 1

    flush_paragraph()
    close_list()
    return "\n".join(out)


def _split_sections(md_text: str) -> tuple[str, str, list[tuple[str, str, str, str]]]:
    """Возвращает preamble, lead и список (номер, полный заголовок, короткий, тело)."""
    lines = md_text.replace("\r\n", "\n").split("\n")
    h1_index = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if h1_index is not None:
        del lines[h1_index]

    first_section = next(
        (i for i, line in enumerate(lines) if line.startswith("## ")), len(lines)
    )
    lead = ""
    for i in range(first_section):
        if lines[i].startswith("> "):
            lead = lines[i][2:].strip()
            del lines[i]
            break

    parts = re.split(r"(?m)^## ", "\n".join(lines))
    preamble = parts[0]
    sections: list[tuple[str, str, str, str]] = []
    for part in parts[1:]:
        title_line, _, body = part.partition("\n")
        title_line = title_line.strip()
        if not title_line:
            continue
        match = re.match(r"^(\d+)\.\s*(.*)$", title_line)
        if match:
            number = match.group(1)
            short = match.group(2).strip()
        else:
            number = str(len(sections) + 1)
            short = title_line
        sections.append((number, title_line, short, body))
    return preamble, lead, sections


def build_html(md_text: str, title: str) -> str:
    """Собирает страницу с шапкой, оглавлением, разделами и подвалом."""
    preamble, lead, sections = _split_sections(md_text)

    toc_items = "\n".join(
        f'<li><a href="#s{number}">{_inline(short)}</a></li>'
        for number, _, short, _ in sections
    )

    main_parts: list[str] = []
    preamble_html = md_to_html(preamble).strip()
    if preamble_html:
        main_parts.append(preamble_html)
    for number, full_title, _, body in sections:
        main_parts.append(
            f'<section id="s{number}">\n'
            f"<h2>{_inline(full_title)}</h2>\n"
            f"{md_to_html(body)}\n"
            "</section>"
        )

    lead_html = f'<p class="lead">{_inline(lead)}</p>\n' if lead else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="ru">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_escape(title)}</title>\n"
        '<link rel="stylesheet" href="style_likbez.css">\n'
        "</head>\n"
        "<body>\n"
        '<article class="content">\n'
        '<header class="doc-header">\n'
        f"<h1>{_inline(title)}</h1>\n"
        '<p class="subtitle">Лабораторная №2 · ADAPT · бинарная классификация</p>\n'
        f"{lead_html}"
        "</header>\n"
        '<nav class="toc" aria-label="Оглавление">\n'
        '<p class="toc-title">Содержание</p>\n'
        f"<ol>\n{toc_items}\n</ol>\n"
        "</nav>\n"
        "<main>\n"
        + "\n".join(main_parts)
        + "\n</main>\n"
        '<footer class="doc-footer">\n'
        "<p>Основные числа — из <code>metrics_adapt.json</code>, <code>loeo_adapt.json</code>, "
        "<code>summary_baselines.json</code>, <code>improvement_adapt.json</code>, "
        "<code>loeo_diagnostics.json</code> и <code>split_info.json</code>. "
        "Значения, которых нет в этих файлах, отмечены в тексте как TODO или проверены "
        "по <code>data/processed/*.parquet</code>. "
        "Стиль документа: <code>style_likbez.css</code>.</p>\n"
        "<p>Документ не содержит внешних ссылок и открывается офлайн.</p>\n"
        "</footer>\n"
        "</article>\n"
        "</body>\n"
        "</html>\n"
    )


def write_style(path: Path | str = config.STYLE_CSS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_CSS.strip() + "\n", encoding="utf-8")
    return path


def main() -> int:
    md_path = config.LIKBEZ_MD
    if not md_path.exists():
        raise FileNotFoundError(f"нет источника ликбеза: {md_path}")
    md_text = md_path.read_text(encoding="utf-8")
    title = next(
        (line[2:].strip() for line in md_text.splitlines() if line.startswith("# ")),
        "Ликбез",
    )
    write_style(config.STYLE_CSS)
    html = build_html(md_text, title)
    config.LIKBEZ_HTML.parent.mkdir(parents=True, exist_ok=True)
    config.LIKBEZ_HTML.write_text(html, encoding="utf-8")
    print(f"ликбез: {config.LIKBEZ_HTML}")
    print(f"оформление: {config.STYLE_CSS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
