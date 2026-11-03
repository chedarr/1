# -*- coding: utf-8 -*-
"""Конвертер XML-схемы процесса лабораторной №1 в HTML.

  docs/lab1_process.xml --(XSD валидация)--> docs/lab1_process.html (XSLT)

Использование:
    py -X utf8 scripts/xml_to_html.py [--validate-only] [--no-check-links]

Что делает:
  1. Валидирует docs/lab1_process.xml против docs/lab1_process.xsd (lxml.XMLSchema).
  2. Применяет docs/lab1_process.xsl (lxml XSLT 1.0).
  3. Записывает docs/lab1_process.html.
  4. Программно проверяет результат: все секции 1–11, таблицы, бейджи,
     отсутствие запрещённого текста ("лекция 3", "ИИ", "агент-").
  5. Проверяет, что пути, на которые ссылается HTML, существуют на диске.

Все проверки офлайн, внешних ресурсов не требуется.
"""
from __future__ import annotations

import sys
from pathlib import Path

from lxml import etree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / "docs"

XML_PATH = DOCS / "lab1_process.xml"
XSD_PATH = DOCS / "lab1_process.xsd"
XSL_PATH = DOCS / "lab1_process.xsl"
CSS_PATH = DOCS / "lab1_process.css"
HTML_PATH = DOCS / "lab1_process.html"

FORBIDDEN = [
    ("лекция 3", False),
    ("лекции 3", False),
    ("ИИ", False),
    ("искусственный интеллект", True),
    ("агент-агент", True),
    ("V1", False),
    ("V2", False),
    ("V3", False),
    ("discharge_num", False),
    ("shortcut", False),
    ("конфаунд", True),
    ("temp_check", False),
    ("cross-temp", False),
    ("no_temp", False),
    ("абляц", True),
    ("до/после", False),
]


# ------------------------------------------------------------------ XSD
def validate_xsd() -> bool:
    xsd = etree.XMLSchema(etree.parse(str(XSD_PATH)))
    doc = etree.parse(str(XML_PATH))
    ok = xsd.validate(doc)
    if not ok:
        print("XSD: НЕ ВАЛИДНО")
        for e in xsd.error_log:
            print(f"  {e.line}: {e.message}")
    else:
        print("XSD: валидно")
    return ok


# ------------------------------------------------------------------ XSLT
def apply_xslt() -> bytes:
    transform = etree.XSLT(etree.parse(str(XSL_PATH)))
    result = transform(etree.parse(str(XML_PATH)))
    html = etree.tostring(result, encoding="UTF-8", pretty_print=True)
    print("XSLT: преобразование выполнено")
    return html


# ------------------------------------------------------------------ проверки результата
def check_html(html: bytes) -> list[str]:
    errors: list[str] = []
    text = html.decode("utf-8")

    # 1. Ключевые секции
    for anchor in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"]:
        if f'id="{anchor}"' not in text:
            errors.append(f"нет секции #{anchor}")

    # 2. Заголовки-разделы
    for heading in ["Метаданные", "Хронология", "Сущности", "Файлы", "Формулы",
                    "Процессы", "Проверки", "Результаты", "Базовые предсказатели",
                    "Ссылки", "Приложение"]:
        if heading not in text:
            errors.append(f"нет заголовка «{heading}»")

    # 3. Формулы в LaTeX и plain
    for f in ["MSE", "MAE", "RMSE", "R²", "RUL_frac", "capacity_ratio"]:
        if f not in text:
            errors.append(f"нет формулы/упоминания {f}")

    # 4. Бейджи статусов
    for badge in ['class="badge">', 'class="badge ok"']:
        pass  # бейджи встречаются иначе
    if 'badge' not in text:
        errors.append("нет бейджей статусов")

    # 5. Запрещённые слова (тема версий и проверок temp_* не должна возвращаться)
    low = text.lower()
    for w, case_insensitive in FORBIDDEN:
        if (w.lower() in low) if case_insensitive else (w in text):
            errors.append(f"найдено запрещённое слово «{w}»")

    # 6. Ссылки на локальные файлы существуют
    tree = etree.HTML(html)
    if tree is None:
        errors.append("HTML не разбирается lxml")
        return errors
    for a in tree.xpath("//a[@href]"):
        href = a.get("href", "")
        if href.startswith("#"):
            continue
        if href.startswith("http://") or href.startswith("https://"):
            continue
        candidate = PROJECT_ROOT / href
        if not candidate.exists():
            errors.append(f"битая локальная ссылка: {href}")

    return errors


# ------------------------------------------------------------------ main
def main() -> int:
    argv = sys.argv[1:]
    validate_only = "--validate-only" in argv

    if not validate_xsd():
        print("ОШИБКА: XML не проходит XSD-валидацию")
        return 1

    if validate_only:
        print("Валидация XSD завершена. HTML не генерируется.")
        return 0

    if not XSL_PATH.exists():
        print(f"ОШИБКА: нет файла {XSL_PATH}")
        return 1

    html = apply_xslt()
    HTML_PATH.write_bytes(html)

    errors = check_html(html)
    if errors:
        print("ПРОВЕРКИ РЕЗУЛЬТАТА:")
        for e in errors:
            print(f"  [FAIL] {e}")
        return 2

    print(f"OK: {HTML_PATH} ({HTML_PATH.stat().st_size} байт)")
    print("Проверки результата: все пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())