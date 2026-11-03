# -*- coding: utf-8 -*-
"""Шаг 3. Строгая проверка: parquet ничего не потерял и исходники целы.

Запуск: py -X utf8 scripts\\step3_verify_adapt.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Satelite Second Lab")
OUT = ROOT / "data"
TXT_DIR = ROOT / "dataset_text"
LOG_XLS = ROOT / "Experiment_Log (1).xls"
LOG_TSV = OUT / "interim" / "adapt_log_sheet1.tsv"

R: list[str] = []
def p(s: str = "") -> None:
    R.append(s)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- 1. копии в data/raw бит-в-бит равны исходникам -------------------------
p("=== 1. ИСХОДНИКИ И КОПИИ ===")
originals = sorted(TXT_DIR.glob("*.txt")) + [LOG_XLS]
same = 0
for q in originals:
    copy = OUT / "raw" / q.name
    if copy.exists() and sha(q) == sha(copy):
        same += 1
    else:
        p(f"  !! РАСХОЖДЕНИЕ: {q.name}")
p(f"[{'OK' if same == len(originals) else '!!'}] sha256 совпадает у {same} "
  f"из {len(originals)} файлов (16 .txt + лог)")
raw_extra = sorted(q.name for q in (OUT / "raw").iterdir()
                   if q.name not in {x.name for x in originals})
p(f"[OK] Лишних файлов в data/raw/: {raw_extra if raw_extra else 'нет'}")

# ---- 2. parquet посимвольно воспроизводит каждый исходный .txt --------------
p()
p("=== 2. ВОССТАНОВЛЕНИЕ СТРОК ИЗ PARQUET ===")
df = pd.read_parquet(OUT / "processed" / "adapt_raw.parquet")
sensors = [c for c in df.columns if c not in ("source_file", "line_no",
                                              "record_type", "Time")]
bad = 0
checked = 0
for f in sorted(TXT_DIR.glob("*.txt")):
    sub = df[df["source_file"] == f.name].sort_values("line_no")
    lines = f.read_text(encoding="utf-8").splitlines()
    if len(sub) != len(lines):
        p(f"  !! {f.name}: строк в parquet {len(sub)}, в файле {len(lines)}")
        bad += 1
        continue
    for (_, row), line in zip(sub.iterrows(), lines):
        rec = [row["record_type"], "" if pd.isna(row["Time"]) else str(row["Time"])]
        rec += ["" if pd.isna(row[c]) else str(row[c]) for c in sensors]
        checked += 1
        if "\t".join(rec) != line:
            bad += 1
            p(f"  !! {f.name}: строка {row['line_no']} не совпала")
p(f"[{'OK' if bad == 0 else '!!'}] Восстановлено посимвольно: {checked - bad} "
  f"из {checked} строк, расхождений {bad}")

# ---- 3. interim == processed ------------------------------------------------
p()
p("=== 3. ДВА PARQUET ИДЕНТИЧНЫ ===")
a = pd.read_parquet(OUT / "interim" / "adapt_raw.parquet")
b = pd.read_parquet(OUT / "processed" / "adapt_raw.parquet")
same_ab = a.shape == b.shape and a.equals(b)
p(f"[{'OK' if same_ab else '!!'}] data/interim и data/processed: "
  f"{a.shape} == {b.shape}, равны: {same_ab}")

# ---- 4. лог ----------------------------------------------------------------
p()
p("=== 4. ЛОГ ===")
tsv = pd.read_csv(LOG_TSV, sep="\t", header=None, dtype=str, keep_default_na=False)
src = tsv.iloc[2:].reset_index(drop=True)
src.columns = list(tsv.iloc[1])
src = src.replace("", pd.NA)
src["Experiment"] = src["Experiment"].ffill()
src["Time (s)"] = pd.to_numeric(src["Time (s)"], errors="coerce")
log = pd.read_parquet(OUT / "processed" / "adapt_log.parquet")
same_log = (len(src) == len(log)
            and list(src.columns) == list(log.columns)
            and src["Experiment"].tolist() == log["Experiment"].tolist()
            and src["Component"].tolist() == log["Component"].tolist()
            and src["Mode"].tolist() == log["Mode"].tolist()
            and src["Time (s)"].astype(float).tolist()
            == log["Time (s)"].astype(float).tolist())
p(f"[{'OK' if same_log else '!!'}] adapt_log.parquet: {len(log)} строк, "
  f"значения совпадают с листом: {same_log}")
p(f"     экспериментов: {log['Experiment'].nunique()} | "
  f"компонентов: {log['Component'].nunique()} | режимов: {log['Mode'].nunique()}")
p(f"     режимы: {sorted(log['Mode'].dropna().unique())}")

p()
p("ИТОГ: " + ("все проверки пройдены" if bad == 0 and same == len(originals)
               and same_ab and same_log else "ЕСТЬ ПРОБЛЕМЫ"))
(OUT / "processed" / "adapt_verify.txt").write_text("\n".join(R), encoding="utf-8")
print("OK, report:", OUT / "processed" / "adapt_verify.txt")
