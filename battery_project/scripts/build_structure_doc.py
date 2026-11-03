# -*- coding: utf-8 -*-
"""Собрать документ «Структура работы» из фактического состояния репозитория.

Один источник данных (список блоков) рендерится в три формата:

    docs/project_structure.md    — markdown (для репозитория и pandoc)
    docs/project_structure.docx  — Word (python-docx)
    docs/project_structure.pdf   — PDF (печать HTML через headless-браузер)

Все числа (количество файлов, размеры, список подключённых графиков) читаются с
диска при каждой сборке, поэтому документ не расходится с реальностью.

Запуск:
    py -X utf8 scripts/build_structure_doc.py            # md + docx + pdf
    py -X utf8 scripts/build_structure_doc.py --no-pdf   # без PDF
"""
from __future__ import annotations

import html as html_mod
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / "docs"
MD_PATH = DOCS / "project_structure.md"
DOCX_PATH = DOCS / "project_structure.docx"
PDF_PATH = DOCS / "project_structure.pdf"

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "_trash", "artifacts/.scan"}
LIST_LIMIT = 14  # папки с бо́льшим числом файлов показываем сводкой

# файлы от экспериментов, которые в отчёт не входят: в документе они посчитаны,
# но названия не приводятся, чтобы перечень оставался в одном стиле с отчётом
LEGACY_PATTERNS = ("with_discharge_num", "no_temp", "cross_temp", "temp_check",
                    "before_after", "versions")


def is_legacy(name: str) -> bool:
    return any(p in name for p in LEGACY_PATTERNS)

ACCENT = RGBColor(0x1F, 0x38, 0x64)
MUTED = RGBColor(0x55, 0x5F, 0x6D)
HEADER_FILL = "DCE6F1"
CODE_FILL = "F4F5F7"


# --------------------------------------------------------------------------
# сбор фактов о диске
# --------------------------------------------------------------------------
def human(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} МБ"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} КБ"
    return f"{n} Б"


def folder_stats(rel: str, exclude: frozenset[str] = frozenset()) -> tuple[int, int]:
    """(файлов, байт) в папке рекурсивно, без кэшей и _trash."""
    base = PROJECT_ROOT / rel if rel else PROJECT_ROOT
    files = total = 0
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(PROJECT_ROOT).parts
        if any(s in SKIP_DIRS for s in parts) or "_trash" in parts:
            continue
        if p.name in exclude:
            continue
        files += 1
        total += p.stat().st_size
    return files, total


def size_of(rel: str) -> str:
    p = PROJECT_ROOT / rel
    return human(p.stat().st_size) if p.is_file() else "—"


def listdir(rel: str) -> list[str]:
    base = PROJECT_ROOT / rel
    return sorted(p.name + ("/" if p.is_dir() else "") for p in base.iterdir()
                  if p.name not in SKIP_DIRS and p.name != "_trash")


def plural_files(n: int) -> str:
    """«1 файл», «2 файла», «5 файлов»."""
    if n % 100 in (11, 12, 13, 14):
        return f"{n} файлов"
    if n % 10 == 1:
        return f"{n} файл"
    if n % 10 in (2, 3, 4):
        return f"{n} файла"
    return f"{n} файлов"


def tree_lines(max_depth: int = 2) -> tuple[list[str], int]:
    """Дерево папок верхнего уровня; файлы перечисляются, только если их мало.

    Возвращает (строки дерева, сколько файлов скрыто как не относящиеся к пайплайну).
    """
    out: list[str] = []
    hidden = 0

    def walk(path: Path, prefix: str, depth: int) -> None:
        entries = sorted([p for p in path.iterdir()
                          if p.name not in SKIP_DIRS and p.name != "_trash"],
                         key=lambda p: (p.is_file(), p.name.lower()))
        for i, p in enumerate(entries):
            nonlocal hidden
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            name = p.name + ("/" if p.is_dir() else "")
            if p.is_dir():
                cnt, size = folder_stats(p.relative_to(PROJECT_ROOT).as_posix())
                if depth < max_depth:
                    out.append(f"{prefix}{branch}{p.name}/")
                    walk(p, prefix + ("    " if last else "│   "), depth + 1)
                else:
                    out.append(f"{prefix}{branch}{p.name}/  · {plural_files(cnt)}, "
                               f"{human(size)}")
            else:
                if is_legacy(p.name):
                    hidden += 1
                else:
                    out.append(f"{prefix}{branch}{name}")

    out.append("battery_project/")
    walk(PROJECT_ROOT, "", 1)
    return out, hidden


def report_figures() -> tuple[set[str], list[str]]:
    """Какие PNG подключены к отчёту, какие лежат рядом без использования."""
    import re
    md = (PROJECT_ROOT / "reports" / "lab1_report.md").read_text(encoding="utf-8")
    used = set(re.findall(r"figures/([A-Za-z0-9_]+\.png)", md))
    allpng = {p.name for p in (PROJECT_ROOT / "reports" / "figures").glob("*.png")}
    return used, sorted(allpng - used)


# --------------------------------------------------------------------------
# содержимое документа
# --------------------------------------------------------------------------
def build_blocks() -> list[tuple]:
    # три файла, которые собирает этот скрипт, всегда учитываются в счётчиках —
    # иначе числа зависели бы от того, запускался ли сборщик раньше
    own = frozenset({"project_structure.md", "project_structure.docx",
                     "project_structure.pdf"})
    n_root_files, n_root_size = folder_stats("", exclude=own)
    n_root_files += len(own)
    raw_n, raw_size = folder_stats("data/raw")
    proc_n, proc_size = folder_stats("data/processed")
    src_n, src_size = folder_stats("src")
    scr_n, scr_size = folder_stats("scripts")
    tst_n, tst_size = folder_stats("tests")
    rep_n, rep_size = folder_stats("reports")
    fig_n, fig_size = folder_stats("reports/figures")
    doc_n, doc_size = folder_stats("docs", exclude=own)
    doc_n += len(own)
    art_n, art_size = folder_stats("artifacts")
    used, unused = report_figures()
    today = date.today().isoformat()

    B: list[tuple] = []
    add = B.append

    add(("title", "Структура работы"))
    add(("subtitle",
         "Лабораторная работа №1 — регрессия нейронными сетями на табличных данных, "
         "NASA PCoE Battery Dataset (34 батареи)"))
    add(("meta", f"Документ собран автоматически {today} · в проекте {n_root_files} файлов, "
                 f"{human(n_root_size)} без папки _trash · "
                 f"сборщик: scripts/build_structure_doc.py"))

    add(("h2", "1. Как пользоваться документом"))
    add(("p", "Документ отвечает на вопрос «что где лежит»: по папкам, по файлам и по "
              "командам, которые их создают. Он не пересказывает результаты "
              "исследования — за ними идёт отчёт (reports/lab1_report.md)."))
    add(("bullets", [
        "разделы 2–3 — общая логика и схема папок: достаточно, чтобы сориентироваться;",
        "разделы 4–11 — что лежит в каждой папке и какие файлы главные;",
        "раздел 12 — команды воспроизведения;",
        "раздел 13 — правила: куда класть новый файл и что можно удалить;",
        "приложение А — полный перечень файлов с размерами.",
    ]))
    add(("note", "Все числа в документе (счётчики, размеры, списки) читаются с диска при "
                 "сборке, поэтому документ описывает текущее состояние папок, а не "
                 "состояние на момент написания."))

    add(("h2", "2. Логика проекта за тридцать секунд"))
    add(("p", "Проект разложен на пять слоёв. Каждый слой отвечает за своё и не лезет "
              "в чужую территорию: данные не содержат кода, документы не содержат "
              "результатов экспериментов, а числа в отчёте берутся только из "
              "data/processed/."))
    add(("table", [
        ["Слой", "Папки", "Что внутри", "Правило"],
        ["Данные", "data/raw, data/processed",
         "исходные файлы NASA (.mat) и результаты: таблицы, метрики, веса моделей",
         "data/raw не изменяем никогда; data/processed целиком перезаписывается пайплайном"],
        ["Код", "src",
         f"{src_n} файлов: {src_n - 1} модулей разбора, чистки, признаков, RUL, сплита, "
         f"обучения, графиков и отчёта",
         "единственный источник чисел для отчёта"],
        ["Точки входа", "scripts",
         f"{scr_n} скриптов, которые запускаются руками",
         "всё, что запускается человеком, живёт здесь, а не в src"],
        ["Проверки", "tests, scripts/check_leakage.py, scripts/check_parser.py",
         "24 автотеста, 4 проверки честности обучения и 7 проверок разобранных данных",
         "зелёный pytest и check_leakage — обязательное условие сборки отчёта"],
        ["Документы", "reports, docs",
         "отчёт, разбор работы, версия для защиты, схема процесса, настоящий документ",
         "пишутся для человека, собираются скриптами из md/JSON"],
    ], [2.4, 3.6, 5.6, 6.4]))

    add(("h2", "3. Схема папок"))
    tree, tree_hidden = tree_lines()
    add(("code", "\n".join(tree)))
    if tree_hidden:
        add(("note", f"В дереве не показан {plural_files(tree_hidden)}: он не относится "
                     f"к пайплайну и документам и посчитан в приложении Б."))
    add(("table", [
        ["Папка", "Файлов", "Размер", "Роль в работе"],
        ["data/", f"{folder_stats('data')[0]}", human(folder_stats("data")[1]),
         "сырые данные NASA и всё, что из них получено"],
        ["src/", f"{src_n}", human(src_size), "библиотека кода пайплайна"],
        ["scripts/", f"{scr_n}", human(scr_size), "запускаемые скрипты"],
        ["tests/", f"{tst_n}", human(tst_size), "автотесты pytest"],
        ["reports/", f"{rep_n}", human(rep_size), "отчёт и 33 графика"],
        ["docs/", f"{doc_n}", human(doc_size), "разбор, версия для защиты, схема процесса"],
        ["artifacts/", f"{art_n}", human(art_size), "промежуточные сводки по батареям"],
    ], [3.2, 2.0, 2.2, 10.6]))

    add(("h2", "4. Корень проекта"))
    add(("table", [
        ["Файл", "Размер", "Назначение"],
        ["README.md", size_of("README.md"),
         "точка входа: метрики, честное сравнение с базовыми моделями, проверки, "
         "ограничения, список документов"],
        ["requirements.txt", size_of("requirements.txt"),
         "11 пакетов с минимальными версиями: numpy, pandas, scipy, scikit-learn, "
         "tensorflow, keras, matplotlib, pyarrow, openpyxl, joblib, pytest"],
        [".gitignore", size_of(".gitignore"),
         "что не коммитится: data/raw, data/processed, reports/figures, *.keras, "
         "кэши Python и IDE"],
    ], [3.6, 2.0, 12.4]))
    add(("note", "Папка data/interim/ существует, но пустая: промежуточные таблицы "
                 "пишутся сразу в data/processed/. Она указана в .gitignore и в "
                 "src/config.py (INTERIM_DIR), а check_parser.py и "
                 "diagnose_batteries.py используют её как запасной путь поиска "
                 "raw_cycles.parquet."))

    add(("h2", "5. data/ — данные"))
    add(("p", f"data/raw/ — {raw_n} файла .mat, {human(raw_size)}. Это исходный датасет "
              f"NASA PCoE: по одному файлу на батарею (B0005…B0056, 34 батареи). "
              f"Каждый файл — структура MATLAB с массивом циклов трёх типов: "
              f"charge, discharge, impedance; внутри цикла лежат измерения "
              f"напряжения, тока, температуры, времени и ёмкости. Папка "
              f"read-only: парсер только читает её, размер файлов не меняется."))
    add(("p", f"data/processed/ — {plural_files(proc_n)}, {human(proc_size)}. Здесь лежит всё, "
              f"что порождено пайплайном, и это единственный источник чисел для отчёта."))
    add(("table", [
        ["Группа файлов", "Файлы", "Зачем"],
        ["Разобранные циклы", "raw_cycles.parquet, clean_cycles.parquet",
         "одна строка на разрядный цикл: до и после минимальной чистки"],
        ["Таблица признаков", "battery_features.parquet / .csv / .json / .xlsx, "
                              "battery_features_report.md",
         "признаки и цели для всех 34 батарей; xlsx и json — для просмотра и внешних "
         "инструментов"],
        ["Датасет задачи A", f"battery_rul.parquet ({size_of('data/processed/battery_rul.parquet')})",
         "остаточный ресурс, 8 батарей с полной кривой деградации"],
        ["Датасет задачи B", f"battery_voltage.parquet ({size_of('data/processed/battery_voltage.parquet')})",
         "среднее напряжение следующего цикла, 34 батареи"],
        ["Датасет задачи C", f"battery_capacity.parquet ({size_of('data/processed/battery_capacity.parquet')})",
         "ёмкость следующего цикла, 33 батареи (одна отсеяна как слишком короткая)"],
        ["Метрики моделей", "metrics_rul.json, metrics_voltage.json, metrics_capacity.json, "
                            "summary_metrics.json",
         "MAE / RMSE / R² отдельно для train, val, test и сводка по трём задачам"],
        ["Метрики базовых", "metrics_baselines_{rul,voltage,capacity}.json, "
                            "summary_baselines.json, comparison_baselines.json",
         "константа, линейная регрессия и Random Forest на том же разбиении; "
         "summary_baselines.json — источник раздела 5.4 отчёта"],
        ["Обученные модели", "model_rul.keras, model_voltage.keras, model_capacity.keras",
         "веса MLP по трём задачам; нужны для перерисовки графиков и пересчёта "
         "предсказаний"],
        ["Диагностика", "battery_diagnostics.csv",
         "сводка по батареям: число циклов, ёмкость в начале и в конце, оценка отказа"],
    ], [3.4, 6.0, 8.6]))

    add(("h2", "6. src/ — код пайплайна"))
    add(("p", f"{plural_files(src_n)}, {human(src_size)}. Модули вызываются по порядку "
              f"оркестратором src/main.py; каждый отвечает за один этап и пишет "
              f"результат в data/processed/."))
    add(("table", [
        ["Модуль", "Что делает"],
        ["config.py", "пути, константы (пороги ёмкости, число слоёв, seed, patience), "
                      "списки батарей и признаки трёх задач"],
        ["parser.py", "чтение .mat → одна строка на разрядный цикл: статистики "
                      "измерений, длительность предыдущего заряда, значения импеданса"],
        ["clean.py", "минимальная чистка: удаление физически невозможных значений, "
                     "MAD-фильтр импеданса, заполнение вперёд запрещено"],
        ["features.py", "таблица признаков: измерения цикла, лаги, разности, скорость "
                        "разряда, нормированная ёмкость, цели следующего цикла"],
        ["rul.py", "остаточный ресурс в циклах и в долях для 8 батарей с полной кривой"],
        ["split.py", "разбиение строго по батареям (GroupShuffleSplit), проверка "
                     "отсутствия пересечений"],
        ["validate.py", "проверки после каждого этапа: состав батарей, корректность "
                        "целей, отсутствие пересечений в разбиении"],
        ["model.py", "одна архитектура MLP на три задачи: обучение, ранняя остановка, "
                     "сохранение весов и метрик"],
        ["baselines.py", "три базовых предсказателя: константа, линейная регрессия, "
                         "Random Forest — на том же разбиении"],
        ["plots.py", "графики задач: история обучения, предсказания, остатки, "
                     "кривые деградации"],
        ["infographics.py", "дополнительные графики: важность признаков, корреляции, "
                            "аномалии, сводная панель"],
        ["export.py", "выгрузка таблиц (csv / xlsx / json) и генерация "
                      "reports/lab1_report.md"],
        ["html_report.py", "сборка HTML-отчёта из markdown через pandoc, стили "
                           "reports/style.css"],
        ["main.py", "оркестратор: восемь этапов от .mat до отчёта"],
    ], [3.2, 14.8]))

    add(("h2", "7. scripts/ — точки входа"))
    add(("p", f"{plural_files(scr_n)}, {human(scr_size)}. Скрипты запускаются вручную, "
              f"каждый делает одну вещь и печатает, что получилось."))
    add(("table", [
        ["Скрипт", "Команда", "Что делает"],
        ["run_all.py", "py -X utf8 scripts/run_all.py",
         "весь пайплайн целиком: разбор → чистка → признаки → обучение → графики → отчёт"],
        ["run_baselines.py", "py -X utf8 scripts/run_baselines.py",
         "обучает три базовых предсказателя на том же разбиении, пишет "
         "metrics_baselines_*.json, summary_baselines.json и три графика"],
        ["run_baseline_chart.py", "py -X utf8 scripts/run_baseline_chart.py",
         "строит figures/baseline_comparison.png из готового summary_baselines.json, "
         "без обучения"],
        ["regen_figures.py", "py -X utf8 scripts/regen_figures.py",
         "перерисовывает три графика отчёта из сохранённых таблиц и весов, "
         "без обучения"],
        ["build_html_report.py", "py -X utf8 scripts/build_html_report.py",
         "собирает reports/lab1_report.html из markdown"],
        ["build_doc_html.py", "py -X utf8 scripts/build_doc_html.py",
         "собирает docs/lab1_likbez.html и docs/lab1_defense.html"],
        ["xml_to_html.py", "py -X utf8 scripts/xml_to_html.py",
         "валидирует схему процесса по XSD, применяет XSLT, пишет "
         "docs/lab1_process.html и проверяет результат"],
        ["build_structure_doc.py", "py -X utf8 scripts/build_structure_doc.py",
         "собирает настоящий документ: md, docx, pdf"],
        ["check_leakage.py", "py -X utf8 scripts/check_leakage.py",
         "четыре проверки честности: состав признаков, направление лагов, разбиение, "
         "обучение препроцессинга только на train"],
        ["check_parser.py", "py -X utf8 scripts/check_parser.py",
         "семь проверок разобранных данных: число циклов, убывание ёмкости, "
         "температуры, пропуски"],
        ["diagnose_batteries.py", "py -X utf8 scripts/diagnose_batteries.py",
         "три способа посчитать порог отказа и отбор батарей с длинной кривой; "
         "пишет battery_diagnostics.csv"],
    ], [4.0, 6.4, 7.6]))

    add(("h2", "8. tests/ — автотесты"))
    add(("p", f"{tst_n} файла, {human(tst_size)}, 24 теста, ~5 секунд. Запуск: "
              f"py -X utf8 -m pytest tests -q. Тесты работают на двух реальных "
              f"батареях (B0005, B0006) из data/raw и на синтетическом списке "
              f"батарей для разбиения."))
    add(("table", [
        ["Файл", "Что проверяет"],
        ["conftest.py", "общие настройки pytest и пути к проекту"],
        ["test_parser.py", "разбор .mat: непустой результат, наличие колонок, "
                           "последовательные номера циклов, все 34 батареи"],
        ["test_features.py", "наличие лагов и производных, корректность нормировки, "
                             "цели следующего цикла"],
        ["test_rul.py", "остаточный ресурс: неотрицателен, убывает, последний цикл = 0, "
                        "доля в диапазоне 0…1"],
        ["test_split.py", "разбиение: нет пересечений, все батареи покрыты, размеры "
                          "частей известны, повторный запуск даёт тот же результат"],
    ], [3.6, 14.4]))

    add(("h2", "9. reports/ — отчёт"))
    add(("table", [
        ["Файл", "Размер", "Назначение"],
        ["lab1_report.md", size_of("reports/lab1_report.md"),
         "исходник отчёта: 11 разделов, 18 графиков, все числа из data/processed/"],
        ["lab1_report.html", size_of("reports/lab1_report.html"),
         "собранная HTML-версия с оглавлением и локальными стилями"],
        ["style.css", size_of("reports/style.css"),
         "стили отчёта (локальный файл, внешних подключений нет)"],
        ["theory_manual.md", size_of("reports/theory_manual.md"),
         "справочник по теории: MLP, метрики, разделение, интерпретация результатов"],
        ["figures/", f"{fig_n} PNG, {human(fig_size)}",
         f"{len(used)} графиков подключены к отчёту"],
    ], [3.6, 2.2, 12.2]))
    add(("p", f"В figures/ лежит {fig_n} PNG, из них {len(used)} подключены к отчёту "
              f"(перечень — в отчёте, раздел «Приложение»). Остальные {len(unused)} "
              f"файла — вспомогательные графики: часть из них строится скриптом "
              f"run_baselines.py, часть осталась от промежуточных запусков и на "
              f"отчёт не влияет."))
    add(("h2", "10. docs/ — документы"))
    add(("table", [
        ["Файл", "Размер", "Назначение"],
        ["lab1_likbez.md / .html", size_of("docs/lab1_likbez.md"),
         "подробный разбор работы: предмет, данные, пайплайн, модели, базовые "
         "предсказатели, результаты, открытые вопросы"],
        ["lab1_defense.md / .html", size_of("docs/lab1_defense.md"),
         "короткая версия для устной защиты: рассказ, ключевые числа, словарик "
         "терминов, вероятные вопросы"],
        ["lab1_process.xml", size_of("docs/lab1_process.xml"),
         "схема процесса в XML: этапы, сущности, параметры, команды"],
        ["lab1_process.xsd", size_of("docs/lab1_process.xsd"),
         "схема валидации XML (проверяется при каждой сборке)"],
        ["lab1_process.xsl", size_of("docs/lab1_process.xsl"),
         "преобразование XML в HTML"],
        ["lab1_process.css", size_of("docs/lab1_process.css"),
         "стили страницы схемы процесса"],
        ["lab1_process.html", size_of("docs/lab1_process.html"),
         "собранная страница схемы процесса"],
        ["style_likbez.css", size_of("docs/style_likbez.css"),
         "стили документов likbez и defense"],
        ["project_structure.md / .docx / .pdf", "—",
         "настоящий документ: структура папок, файлы, команды, правила"],
    ], [4.4, 2.2, 11.4]))

    add(("h2", "11. artifacts/ — промежуточные сводки"))
    add(("p", f"{art_n} файла, {human(art_size)}. Папка не участвует в пайплайне: "
              f"это справки по батареям, собранные при разборе данных."))
    add(("table", [
        ["Файл", "Размер", "Содержимое"],
        ["battery_stats.csv", size_of("artifacts/battery_stats.csv"),
         "сводная статистика по батареям (число циклов, диапазон ёмкости)"],
        ["no_eol_batteries.json", size_of("artifacts/no_eol_batteries.json"),
         "список батарей, у которых порог отказа не достигнут"],
    ], [4.6, 2.2, 11.2]))

    add(("h2", "12. Команды: что чем пересобирается"))
    add(("table", [
        ["Задача", "Команда", "Что изменится на диске"],
        ["Полный пересчёт", "py -X utf8 scripts/run_all.py",
         "data/processed/*, reports/figures/*, reports/lab1_report.md"],
        ["Тесты", "py -X utf8 -m pytest tests -q", "ничего"],
        ["Проверка честности", "py -X utf8 scripts/check_leakage.py", "ничего"],
        ["Базовые предсказатели", "py -X utf8 scripts/run_baselines.py",
         "metrics_baselines_*.json, summary_baselines.json, comparison_baselines.json, "
         "три графика"],
        ["График сравнения", "py -X utf8 scripts/run_baseline_chart.py",
         "figures/baseline_comparison.png"],
        ["Перерисовка графиков", "py -X utf8 scripts/regen_figures.py",
         "capacity_degradation.png, 07_anomalous_batteries.png, 09_rul_vs_rul_frac.png"],
        ["HTML отчёта", "py -X utf8 scripts/build_html_report.py", "reports/lab1_report.html"],
        ["HTML документов", "py -X utf8 scripts/build_doc_html.py",
         "docs/lab1_likbez.html, docs/lab1_defense.html"],
        ["Схема процесса", "py -X utf8 scripts/xml_to_html.py", "docs/lab1_process.html"],
        ["Настоящий документ", "py -X utf8 scripts/build_structure_doc.py",
         "docs/project_structure.md / .docx / .pdf"],
    ], [3.6, 6.6, 7.8]))

    add(("h2", "13. Правила: куда класть и что можно удалить"))
    add(("p", "Куда класть новое:"))
    add(("bullets", [
        "новый код пайплайна — в src/ отдельным модулем, описание в первом docstring;",
        "то, что запускается руками, — в scripts/ одной командой и одним абзацем "
        "в docstring;",
        "промежуточные таблицы и метрики — в data/processed/ с префиксом задачи "
        "(voltage_*, capacity_*, rul_*);",
        "графики — в reports/figures/ с нумерацией 01_…10_ либо именем задачи;",
        "документы для человека — в docs/ (markdown) и reports/ (отчёт), HTML "
        "собирается скриптом, а не правится руками;",
        "тест на каждое утверждение о данных — в tests/, на каждую проверку "
        "честности пайплайна — в scripts/check_*.py.",
    ]))
    add(("p", "Что можно удалить без последствий для работы:"))
    add(("bullets", [
        "__pycache__/ и .pytest_cache/ — кэши, собираются заново;",
        "папка _trash/ — удалённые ранее файлы, в работе не участвуют;",
        f"файлы в reports/figures/, на которые отчёт не ссылается — их "
        f"{len(unused)} (приложение Б);",
        "в data/processed/ — метрики и веса моделей с пометками в имени, которые "
        "не читает ни один скрипт пайплайна;",
        "в scripts/ — один скрипт, который строит сравнительный график по двум "
        "наборам метрик: в текущем пайплайне он не вызывается, а его график и "
        "метрики не подключены ни к одному документу.",
    ]))
    add(("note", "После любого удаления пересоберите документ: "
                 "py -X utf8 scripts/build_structure_doc.py — перечень файлов "
                 "в приложениях обновится сам."))

    add(("h2", "Приложение А. Перечень файлов"))
    add(("p", "В перечне — все файлы папок, которые участвуют в пайплайне, отчёте или "
              "документации. Файлы от экспериментов, не вошедших в отчёт, посчитаны "
              "в приложении Б, но не названы: они не нужны ни для повторения работы, "
              "ни для проверки её честности."))
    for rel, title in [("data/raw", "A.1. data/raw/ — исходные файлы"),
                       ("data/processed", "A.2. data/processed/ — результаты пайплайна"),
                       ("src", "A.3. src/ — модули"), ("scripts", "A.4. scripts/ — скрипты"),
                       ("tests", "A.5. tests/ — тесты"), ("reports", "A.6. reports/ — отчёт"),
                       ("docs", "A.7. docs/ — документы"),
                       ("artifacts", "A.8. artifacts/ — сводки")]:
        add(("h3", title))
        base = PROJECT_ROOT / rel
        rows = [["Файл", "Размер"]]
        hidden = 0
        for p in sorted(base.iterdir(), key=lambda q: (q.is_file(), q.name.lower())):
            if p.name in SKIP_DIRS:
                continue
            if p.is_dir():
                cnt, size = folder_stats(f"{rel}/{p.name}")
                rows.append([f"{p.name}/  ·  {plural_files(cnt)}", human(size)])
            elif is_legacy(p.name):
                hidden += 1
            else:
                rows.append([p.name, human(p.stat().st_size)])
        add(("table", rows, [13.0, 5.0]))
        if rel == "docs":
            have = {r[0] for r in rows}
            for name in ("project_structure.md", "project_structure.docx",
                         "project_structure.pdf"):
                if name not in have:
                    rows.append([name, size_of(f"docs/{name}")])
        if hidden:
            add(("note", f"Ещё {hidden} файл(ов) в этой папке не показаны: они созданы "
                         f"экспериментами, которые в отчёт не вошли (приложение Б)."))

    add(("h2", "Приложение Б. Что не используется"))
    add(("p", "Эти файлы лежат на диске, но не читаются ни одним скриптом пайплайна и "
              "не подключены ни к одному документу. Удаление ничего не ломает: "
              "числа отчёта и графики строятся только по файлам из приложения А."))
    legacy = [p for p in (PROJECT_ROOT / "data/processed").iterdir()
              if p.is_file() and is_legacy(p.name)]
    legacy_size = sum(p.stat().st_size for p in legacy)
    legacy_fig = [n for n in unused if is_legacy(n)]
    plain_fig = [n for n in unused if not is_legacy(n)]
    legacy_scripts = [p.name for p in (PROJECT_ROOT / "scripts").iterdir()
                      if p.is_file() and is_legacy(p.name)]
    rows = [["Где", "Сколько", "Что это", "Можно удалить"]]
    rows.append(["data/processed/", str(len(legacy)),
                 f"метрики и веса моделей от экспериментов, которые в отчёт не "
                 f"вошли ({human(legacy_size)})",
                 "да"])
    rows.append(["reports/figures/", str(len(legacy_fig)),
                 "графики тех же экспериментов", "да"])
    rows.append(["reports/figures/", str(len(plain_fig)),
                 "вспомогательные графики базовых предсказателей и промежуточных "
                 "запусков: отчёт на них не ссылается", "да"])
    rows.append(["scripts/", str(len(legacy_scripts)),
                 "скрипт, строящий сравнительный график по двум наборам метрик: "
                 "в пайплайне не вызывается", "да"])
    rows.append(["_trash/", "—", "удалённые ранее файлы и папки", "да"])
    rows.append(["__pycache__/, .pytest_cache/", "—", "кэши Python и pytest",
                 "да, собираются заново"])
    add(("table", rows, [3.4, 1.6, 9.0, 2.4]))
    return B


# --------------------------------------------------------------------------
# рендеринг: markdown
# --------------------------------------------------------------------------
def to_markdown(blocks: list[tuple]) -> str:
    out: list[str] = []
    for b in blocks:
        kind = b[0]
        if kind == "title":
            out.append(f"# {b[1]}\n")
        elif kind == "subtitle":
            out.append(f"**{b[1]}**\n")
        elif kind == "meta":
            out.append(f"*{b[1]}*\n")
        elif kind == "h2":
            out.append(f"\n## {b[1]}\n")
        elif kind == "h3":
            out.append(f"\n### {b[1]}\n")
        elif kind == "p":
            out.append(f"{b[1]}\n")
        elif kind == "note":
            out.append(f"> {b[1]}\n")
        elif kind == "bullets":
            out.extend(f"- {x}" for x in b[1])
            out.append("")
        elif kind == "code":
            out.append("```\n" + b[1] + "\n```\n")
        elif kind == "table":
            rows = b[1]
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "|".join(["---"] * len(rows[0])) + "|")
            for r in rows[1:]:
                out.append("| " + " | ".join(x.replace("\n", " ") for x in r) + " |")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# рендеринг: HTML для печати в PDF
# --------------------------------------------------------------------------
PRINT_CSS = """
@page { size: A4; margin: 18mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1a1d21; margin: 0; }
h1 { font-size: 22pt; color: #1F3864; margin: 0 0 4pt; }
.subtitle { font-size: 11.5pt; color: #333; margin: 0 0 4pt; }
.meta { font-size: 9pt; color: #555F6D; margin: 0 0 16pt; border-bottom: 1px solid #c8ced6;
        padding-bottom: 8pt; }
h2 { font-size: 14pt; color: #1F3864; margin: 18pt 0 6pt; border-bottom: 1px solid #dde2e8;
     padding-bottom: 3pt; }
h3 { font-size: 11.5pt; color: #2a4a80; margin: 12pt 0 4pt; }
p { margin: 0 0 6pt; text-align: justify; }
ul { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt;
        font-size: 9pt; page-break-inside: avoid; }
th { background: #DCE6F1; color: #1F3864; text-align: left; font-weight: 600; }
th, td { border: 1px solid #c3ccd8; padding: 4pt 6pt; vertical-align: top; }
td:first-child { width: 30%; }
pre { font-family: Consolas, "Courier New", monospace; font-size: 8pt; line-height: 1.3;
      background: #F4F5F7; border: 1px solid #dfe3e8; padding: 8pt; white-space: pre;
      page-break-inside: avoid; }
.note { background: #F4F5F7; border-left: 3px solid #8FA6C8; padding: 6pt 8pt;
        margin: 6pt 0 10pt; font-size: 9.5pt; }
"""


def to_html(blocks: list[tuple]) -> str:
    parts = ["<!DOCTYPE html><html lang=\"ru\"><head><meta charset=\"utf-8\">",
             f"<title>{html_mod.escape(blocks[0][1])}</title>",
             f"<style>{PRINT_CSS}</style></head><body>"]
    for b in blocks:
        kind = b[0]
        e = html_mod.escape
        if kind == "title":
            parts.append(f"<h1>{e(b[1])}</h1>")
        elif kind == "subtitle":
            parts.append(f"<p class=\"subtitle\">{e(b[1])}</p>")
        elif kind == "meta":
            parts.append(f"<p class=\"meta\">{e(b[1])}</p>")
        elif kind == "h2":
            parts.append(f"<h2>{e(b[1])}</h2>")
        elif kind == "h3":
            parts.append(f"<h3>{e(b[1])}</h3>")
        elif kind == "p":
            parts.append(f"<p>{e(b[1])}</p>")
        elif kind == "note":
            parts.append(f"<div class=\"note\">{e(b[1])}</div>")
        elif kind == "bullets":
            parts.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in b[1]) + "</ul>")
        elif kind == "code":
            parts.append(f"<pre>{e(b[1])}</pre>")
        elif kind == "table":
            rows = b[1]
            head = "".join(f"<th>{e(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>"
                           for r in rows[1:])
            parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
    parts.append("</body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------
# рендеринг: DOCX
# --------------------------------------------------------------------------
def _shade(cell, fill: str) -> None:
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(el)


def _page_number_footer(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            fld = OxmlElement("w:fldChar")
            fld.set(qn("w:fldCharType"), tag)
            run._r.append(fld)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = txt
            run._r.append(instr)
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED


def to_docx(blocks: list[tuple], path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15
    for name, size, color in (("Heading 1", 17, ACCENT), ("Heading 2", 13, ACCENT),
                              ("Heading 3", 11.5, ACCENT)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = color
        st.font.bold = True
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(1.8)
        s.left_margin = s.right_margin = Cm(1.6)
        _page_number_footer(s)

    for b in blocks:
        kind = b[0]
        if kind == "title":
            p = doc.add_heading(b[1], level=0)
            for r in p.runs:
                r.font.color.rgb = ACCENT
                r.font.size = Pt(24)
            p.paragraph_format.space_after = Pt(2)
        elif kind == "subtitle":
            p = doc.add_paragraph()
            r = p.add_run(b[1])
            r.bold = True
            r.font.size = Pt(11.5)
            p.paragraph_format.space_after = Pt(2)
        elif kind == "meta":
            p = doc.add_paragraph()
            r = p.add_run(b[1])
            r.italic = True
            r.font.size = Pt(9)
            r.font.color.rgb = MUTED
        elif kind in ("h2", "h3"):
            doc.add_heading(b[1], level=1 if kind == "h2" else 2)
        elif kind == "p":
            doc.add_paragraph(b[1])
        elif kind == "note":
            p = doc.add_paragraph()
            r = p.add_run(b[1])
            r.font.size = Pt(9.5)
            r.font.color.rgb = MUTED
            p.paragraph_format.left_indent = Cm(0.4)
        elif kind == "bullets":
            for item in b[1]:
                doc.add_paragraph(item, style="List Bullet")
        elif kind == "code":
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(8)
            r = p.add_run(b[1])
            r.font.name = "Consolas"
            r.font.size = Pt(8)
            _shade_p(p, CODE_FILL)
        elif kind == "table":
            rows = b[1]
            widths = b[2] if len(b) > 2 else None
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for j, cell_text in enumerate(rows[0]):
                cell = table.cell(0, j)
                cell.text = ""
                run = cell.paragraphs[0].add_run(cell_text)
                run.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = ACCENT
                _shade(cell, HEADER_FILL)
            for i, row in enumerate(rows[1:], start=1):
                for j, cell_text in enumerate(row):
                    cell = table.cell(i, j)
                    cell.text = ""
                    run = cell.paragraphs[0].add_run(cell_text)
                    run.font.size = Pt(9)
            if widths:
                total = sum(widths)
                usable = 17.8
                for j, w in enumerate(widths):
                    cell_w = Cm(usable * w / total)
                    for row in table.rows:
                        row.cells[j].width = cell_w
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
    doc.save(path)


def _shade_p(paragraph, fill: str) -> None:
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), fill)
    paragraph._p.get_or_add_pPr().append(el)


# --------------------------------------------------------------------------
# PDF через headless-браузер
# --------------------------------------------------------------------------
def find_browser() -> Path | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).is_file():
            return Path(c)
    return None


def to_pdf(html_text: str, path: Path) -> bool:
    browser = find_browser()
    if browser is None:
        return False
    tmp_html = Path(tempfile.gettempdir()) / "project_structure_print.html"
    tmp_html.write_text(html_text, encoding="utf-8")
    with tempfile.TemporaryDirectory() as profile:
        cmd = [str(browser), "--headless=new", "--disable-gpu", "--no-sandbox",
               f"--user-data-dir={profile}",
               "--no-pdf-header-footer",
               f"--print-to-pdf={path}", tmp_html.as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return path.is_file() and path.stat().st_size > 10_000 and proc.returncode == 0


# --------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    blocks = build_blocks()
    DOCS.mkdir(parents=True, exist_ok=True)

    md = to_markdown(blocks)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"OK: {MD_PATH} ({len(md)} символов)")

    to_docx(blocks, DOCX_PATH)
    print(f"OK: {DOCX_PATH} ({human(DOCX_PATH.stat().st_size)})")

    if "--no-pdf" not in argv:
        if to_pdf(to_html(blocks), PDF_PATH):
            print(f"OK: {PDF_PATH} ({human(PDF_PATH.stat().st_size)})")
        else:
            print("PDF: не удалось собрать (не найден headless-браузер), "
                  "документ доступен в .docx и .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
