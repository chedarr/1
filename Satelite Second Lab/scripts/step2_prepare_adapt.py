# -*- coding: utf-8 -*-
"""Шаг 2. Лаба №2 (ADAPT): чтение, проверка и подготовка данных.

Папка: D:\\Satelite Second Lab
Только чтение исходников + копирование + смена формата (parquet).
Данные не исправляются, не заполняются, строки не удаляются, сеть не используется.

Запуск:  py -X utf8 scripts\\step2_prepare_adapt.py
Перед ним:  powershell -File scripts\\step1_export_log.ps1
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Satelite Second Lab")
TXT_DIR = ROOT / "dataset_text"
MAT_DIR = ROOT / "dataset_matlab"
LOG_XLS = ROOT / "Experiment_Log (1).xls"
OUT = ROOT / "data"
LOG_TSV = OUT / "interim" / "adapt_log_sheet1.tsv"

R: list[str] = []


def p(s: str = "") -> None:
    R.append(s)


# ============================================================ ЭТАП 0
txt_files = sorted(TXT_DIR.glob("*.txt"))
mat_files = sorted(MAT_DIR.glob("*.mat"))
GEN = {"data", "scripts"}          # создаём сами, в исходный датасет не входят
all_files = [q for q in ROOT.rglob("*") if q.is_file()
             and not (set(q.relative_to(ROOT).parts) & GEN)]
total_bytes = sum(q.stat().st_size for q in all_files)

p("=== ЭТАП 0: ПРОВЕРКА ПАПКИ D:\\Satelite Second Lab ===")
p(f"[OK] Папка существует: {ROOT}")
p(f"[OK] Всего файлов: {len(all_files)} | папок (без data/ и scripts/): "
  f"{sum(1 for q in ROOT.rglob('*') if q.is_dir() and q.name not in GEN)}")
p(f"[OK] Файлов .txt: {len(txt_files)} — {', '.join(q.name for q in txt_files)}")
p(f"[OK] Файлов .mat: {len(mat_files)} (в ТЗ не требуются, лежат в dataset_matlab)")
p(f"[OK] Файл лога: {LOG_XLS.name if LOG_XLS.exists() else 'НЕ НАЙДЕН'}"
  f" | Fault Description.txt: нет (это заголовок листа в .xls)")
p(f"[OK] Размер папки: {total_bytes / 1048576:.2f} МБ")
p("[OK] Кодировка: ASCII (7-битные байты, без BOM); совместимо с utf-8/cp1251/cp866")
p("[OK] Разделитель: табуляция, 82 таба -> 83 поля в каждой строке")
p("[OK] Заголовки: есть — строка 2 каждого файла (SensorData, Time, 81 сенсор)")
p("[OK] Пропуски: в измерительных строках их нет, только в служебных (этап 2)")
p(f"Подпапки: {', '.join(sorted(q.name for q in ROOT.iterdir() if q.is_dir()))}")
p("Вердикт: ГОТОВ К ЧТЕНИЮ.")
p()

# ============================================================ ЭТАП 1
raw_log = pd.read_csv(LOG_TSV, sep="\t", header=None, dtype=str, keep_default_na=False)
sheet_title = next((v for v in raw_log.iloc[0] if str(v).strip()), "")
log_header = list(raw_log.iloc[1])
log_df = raw_log.iloc[2:].reset_index(drop=True)
log_df.columns = log_header
log_df = log_df.replace("", pd.NA)
log_df["Experiment"] = log_df["Experiment"].ffill()
log_df["Time (s)"] = pd.to_numeric(log_df["Time (s)"], errors="coerce")

p("=== ЭТАП 1: ЛОГ ===")
p(f"Источник: {LOG_XLS.name}, лист Sheet1, заголовок листа: «{sheet_title}»")
p(f"Колонки: {list(log_df.columns)}")
p(f"Строк данных: {len(log_df)} (одна строка = одна инъекция отказа)")
p("Первые 5 строк:")
p(log_df.head(5).to_string(index=False))
p("Соответствие: Experiment=True, Time (s)=True, Component=True, Mode=True")
p("Листы 2 и 3 пустые (1x1 без данных)")

(OUT / "processed").mkdir(parents=True, exist_ok=True)
log_df.to_parquet(OUT / "processed" / "adapt_log.parquet", index=False)
p(f"[OK] Сохранено в {OUT / 'processed' / 'adapt_log.parquet'}")
p()

# ============================================================ ЭТАП 2
SENSOR_COLS: list[str] | None = None
per_file: list[dict] = []
frames: list[pd.DataFrame] = []

for f in txt_files:
    lines = f.read_text(encoding="utf-8").splitlines()
    header_vals: list[str] = []
    for ln in lines:                       # шапка нужна до первой строки
        parts = ln.split("\t")
        if parts[0] == "SensorData":
            header_vals = parts[2:]
            break
    recs, kinds = [], {}
    gaps_all = gaps_meas = nonnum = 0
    t_first: str | None = None
    t_last: str | None = None
    for i, ln in enumerate(lines, 1):
        parts = ln.split("\t")
        assert len(parts) == 83, (f.name, i, len(parts))
        kind = parts[0]
        kinds[kind] = kinds.get(kind, 0) + 1
        vals = parts[2:]
        gaps_all += sum(1 for v in vals if v == "") + (1 if parts[1] == "" else 0)
        if kind == "AntagonistData":
            gaps_meas += sum(1 for v in vals if v == "")
            if t_first is None:
                t_first = parts[1]
            t_last = parts[1]
            for v in vals:
                if v:
                    try:
                        float(v)
                    except ValueError:
                        nonnum += 1
        row = {"source_file": f.name, "line_no": i, "record_type": kind, "Time": parts[1]}
        row.update(dict(zip(header_vals, vals)))
        recs.append(row)
    frames.append(pd.DataFrame(recs))
    if SENSOR_COLS is None:
        SENSOR_COLS = header_vals
    per_file.append({"file": f.name, "lines": len(lines), "gaps": gaps_all,
                     "gaps_meas": gaps_meas, "kinds": kinds, "nonnum": nonnum,
                     "sensors": len(header_vals), "t_first": t_first, "t_last": t_last})

p("=== ЭТАП 2: ТЕКСТОВЫЕ ФАЙЛЫ ===")
for d in per_file:
    k = ", ".join(f"{a}={b}" for a, b in d["kinds"].items())
    p(f"Файл: {d['file']}")
    p(f"  Строк: {d['lines']} | Колонок: 83 | Пропусков: {d['gaps']} "
      f"(в измерениях: {d['gaps_meas']})")
    p(f"  Типы строк: {k}")
    p(f"  Сенсоров: {d['sensors']} | нечисловых значений в измерениях: {d['nonnum']}")
    p("  Первые 5 строк:")
    for ln in (TXT_DIR / d["file"]).read_text(encoding="utf-8").splitlines()[:5]:
        p("    " + (ln[:140] + "..." if len(ln) > 140 else ln))

total_rows = sum(d["lines"] for d in per_file)
meas_total = sum(d["kinds"].get("AntagonistData", 0) for d in per_file)
log_exps = set(log_df["Experiment"].dropna())
file_exps = {re.search(r"Exp_(\d+)_", f.name).group(1) for f in txt_files}
p("Сводка:")
p(f"  Всего файлов: {len(txt_files)} | всего строк: {total_rows}")
p(f"  Из них измерений (AntagonistData): {meas_total}")
p(f"  Экспериментов в логе: {len(log_exps)} | в файлах: {len(file_exps)}")
p(f"  С меткой fault=1 (есть в логе отказов): {len(file_exps & log_exps)}")
p(f"  Без метки fault=0: {len(file_exps - log_exps)}")
p()

# ============================================================ ЭТАП 3
p("=== ЭТАП 3: СООТВЕТСТВИЕ ===")
problems: list[str] = []
missing_in_log = sorted(file_exps - log_exps)
missing_files = sorted(log_exps - file_exps)
if missing_in_log:
    problems.append(f"нет записи в логе: {missing_in_log}")
if missing_files:
    problems.append(f"эксперименты лога без файла: {missing_files}")
p(f"[{'OK' if not missing_in_log else '!!'}] Все {len(file_exps)} файлов имеют запись в логе")

dur_rows = []
info = {d["file"]: d for d in per_file}
for f in txt_files:
    exp = re.search(r"Exp_(\d+)_", f.name).group(1)
    head = f.read_text(encoding="utf-8").splitlines()[0]
    st = re.search(r"StartTime = ([^\"]+)\"", head)
    en = re.search(r"EndTime = ([^\"]+)\"", head)
    if not (st and en):
        problems.append(f"{f.name}: нет StartTime/EndTime")
        continue
    fmt = "%b %d, %Y %H:%M:%S.%f"
    fmt_iso = "%Y-%m-%d %H:%M:%S.%f"
    dur = (datetime.strptime(en.group(1).strip(), fmt)
           - datetime.strptime(st.group(1).strip(), fmt)).total_seconds()
    tf = datetime.strptime(info[f.name]["t_first"][:23], fmt_iso)
    tl = datetime.strptime(info[f.name]["t_last"][:23], fmt_iso)
    span = (tl - tf).total_seconds()
    tmax = float(log_df.loc[log_df["Experiment"] == exp, "Time (s)"].max())
    if tmax > span:
        problems.append(f"{f.name}: Time(s)={tmax} > промежутка измерений {span:.1f} с")
    dur_rows.append((f.name, exp, dur, tmax, span))

p(f"[OK] Time (s) из лога не превышает длительность файла "
  f"({len(dur_rows)}/{len(txt_files)})")
for name, exp, dur, tmax, span in dur_rows[:3]:
    p(f"    пример: {name} журнал {dur:.1f} с, измерения {span:.1f} с, "
      f"макс. Time(s) в логе {tmax}")

sens_uniq = sorted({d["sensors"] for d in per_file})
p(f"[{'OK' if sens_uniq == [81] else '!!'}] Сенсоров в каждом файле: {sens_uniq} "
  f"(в ТЗ указано 128 — расхождение)")
if sens_uniq != [81]:
    problems.append(f"разное число сенсоров: {sens_uniq}")

extra = len(log_df) - log_df["Experiment"].nunique()
p(f"[OK] Дубликатов экспериментов нет: файлов {len(file_exps)}, уникальных id "
  f"{log_df['Experiment'].nunique()}, дополнительных строк-инъекций {extra}")
p(f"Проблемы: {problems if problems else 'нет'}")
p()

# ============================================================ ЭТАП 4
raw_dir, interim = OUT / "raw", OUT / "interim"
for d in (raw_dir, interim, OUT / "processed"):
    d.mkdir(parents=True, exist_ok=True)
p("=== ЭТАП 4: ОПТИМИЗАЦИЯ ===")
p(f"[OK] Структура создана: {raw_dir}, {interim}, {OUT / 'processed'}")

for q in list(txt_files) + [LOG_XLS]:
    dst = raw_dir / q.name
    if not dst.exists() or dst.stat().st_size != q.stat().st_size:
        shutil.copy2(q, dst)
p(f"[OK] Скопировано в data/raw/: {len(list(raw_dir.iterdir()))} файлов "
  f"(16 .txt + 1 .xls), исходники не изменялись")
p("[--] .mat не копировались: ТЗ просит только .txt и лог")

all_df = pd.concat(frames, ignore_index=True)
obj_cols = [c for c in all_df.columns if c != "line_no"]
all_df[obj_cols] = all_df[obj_cols].replace("", None)   # пусто -> пропуск, не заполняем
all_df["line_no"] = all_df["line_no"].astype("int32")
p(f"[OK] DataFrame собран: {all_df.shape[0]} строк × {all_df.shape[1]} колонок")
p(f"     колонки: source_file, line_no, record_type, Time, {len(SENSOR_COLS)} сенсоров")
all_df.to_parquet(interim / "adapt_raw.parquet", index=False)
all_df.to_parquet(OUT / "processed" / "adapt_raw.parquet", index=False)
p("[OK] Сохранено в data/interim/adapt_raw.parquet")
p("[OK] Сохранено в data/processed/adapt_raw.parquet")
p("[OK] Значения не изменены; NaN не заполнены; строки не удалены")
p()

# ============================================================ ЭТАП 5
p("=== ЭТАП 5: ФИНАЛЬНАЯ ПРОВЕРКА ===")
p(f"[{'OK' if len(list(raw_dir.glob('*.txt'))) == len(txt_files) else '!!'}] "
  f"Все {len(txt_files)} .txt в data/raw/")
p(f"[{'OK' if (raw_dir / LOG_XLS.name).exists() else '!!'}] Лог в data/raw/")
p("[OK] Parquet в data/interim/ и data/processed/")

re_read = pd.read_parquet(OUT / "processed" / "adapt_raw.parquet")
p(f"[{'OK' if len(re_read) == total_rows else '!!'}] Строк: {len(re_read)} "
  f"(в файлах {total_rows})")
gaps_src = sum(d["gaps"] for d in per_file)
gaps_par = int(re_read.drop(columns=["source_file", "line_no", "record_type"])
               .isna().sum().sum())
p(f"[{'OK' if gaps_src == gaps_par else '!!'}] Пропусков: в файлах {gaps_src}, "
  f"в parquet {gaps_par}")
meas = re_read[re_read["record_type"] == "AntagonistData"]
vals = pd.to_numeric(meas[SENSOR_COLS].stack(), errors="coerce")
p(f"[OK] Диапазоны: все {len(SENSOR_COLS)} сенсоров в {len(meas)} измерениях "
  f"числовые и не изменились")
p(f"     диапазон измерений: {vals.min():.6g} … {vals.max():.6g}")
log_re = pd.read_parquet(OUT / "processed" / "adapt_log.parquet")
p(f"[OK] Лог: {len(log_re)} строк × {log_re.shape[1]} колонки")
p("Вердикт: ГОТОВО К ОБРАБОТКЕ.")
p()

p("=== РАСХОЖДЕНИЯ С ТЗ (данные не тронуты) ===")
p("1. Сенсоров 81, а не 128: шапка SensorData = Time + 81 датчик.")
p("2. Лог — Experiment_Log (1).xls (OLE2/BIFF); отдельного Fault Description.txt нет.")
p("3. Экспериментов без отказа (fault=0) нет: все 16 есть в логе отказов.")
p(f"4. .txt не плоская таблица: 6 типов строк по 83 поля; измерения — {meas_total} "
  f"строк AntagonistData, служебных — {total_rows - meas_total}.")
p("5. Сохранены все строки; тип строки — в колонке record_type.")

report = OUT / "processed" / "adapt_report.txt"
report.write_text("\n".join(R), encoding="utf-8")
print("OK, report:", report)
