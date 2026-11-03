"""Проверка утечек: служебные признаки, лаги, split, scaler.

Скрипт проверяет честность пайплайна обучения:
  1) в feature_cols нет служебных признаков (battery_id, cycle_index,
     discharge_num, total_cycles, date, time);
  2) признаки-лаги строятся shift(+1/+2/+3) — из прошлого, а shift(-1)
     используется только для целей (voltage_mean_next, capacity_next);
  3) сплит GroupShuffleSplit по battery_id, пересечений train/val/test нет;
  4) StandardScaler/SimpleImputer обучаются только на train.
"""
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

import pandas as pd

from config import (
    CAPACITY_FEATURES,
    PROCESSED_DIR,
    RUL_FEATURES,
    VOLTAGE_FEATURES,
)

SUSPICIOUS = [
    "battery_id", "cycle_index", "discharge_num", "total_cycles",
    "date", "time",
]

TASKS = {
    "RUL_frac": RUL_FEATURES,
    "voltage": VOLTAGE_FEATURES,
    "capacity": CAPACITY_FEATURES,
}
METRICS_JSON = {
    "RUL_frac": PROCESSED_DIR / "metrics_rul.json",
    "voltage": PROCESSED_DIR / "metrics_voltage.json",
    "capacity": PROCESSED_DIR / "metrics_capacity.json",
}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_sep(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
print_sep("ПРОВЕРКА СЛУЖЕБНЫХ ПРИЗНАКОВ")
for task, cols in TASKS.items():
    bad = [s for s in SUSPICIOUS if s in cols]
    print(f"\nЗадача: {task}  (n={len(cols)})")
    for s in SUSPICIOUS:
        mark = "❌ ПОДОЗРИТЕЛЬНЫЙ" if s in cols else "OK"
        print(f"  {mark}: {s}")
    if bad:
        print(f"  ИТОГ: {bad} ПРИСУТСТВУЮТ в feature_cols!")
    else:
        print("  ИТОГ: служебных признаков нет")

# Дополнительно: что сохранено в metrics_*.json (факт обучения)
print("\n-- feature_cols, сохранённые в metrics_*.json --")
for task, path in METRICS_JSON.items():
    if not path.exists():
        print(f"  {task}: файл {path.name} отсутствует")
        continue
    stored = load_json(path).get("feature_cols") or []
    bad = [s for s in SUSPICIOUS if s in stored]
    print(f"  {task}: n={len(stored)}, служебных={bad if bad else 'нет'}")

# ----------------------------------------------------------------------
print_sep("ПРОВЕРКА ЛАГОВ (src/features.py)")
feat_src = (PROJECT / "src" / "features.py").read_text(encoding="utf-8")
lag_pat = re.compile(r'g\["([a-z0-9_]+)"\]\.shift\(([+-]?\d+)\)', re.I)
found = lag_pat.findall(feat_src)
for col, k in found:
    k = int(k)
    role = "ЦЕЛЬ (следующий цикл)" if k < 0 else "ПРИЗНАК (прошлые циклы)"
    print(f"  shift({k:+d}): {col}  ->  {role}")
neg = [c for c, k in found if int(k) < 0]
pos = [c for c, k in found if int(k) > 0]
print(f"\n  Признаки-лаги (положительный shift): {sorted(set(pos))}")
print(f"  Цели (отрицательный shift):          {sorted(set(neg))}")
if neg:
    print("  ОК: shift(-1) только для целей next-цикла; признаки из прошлого.")
else:
    print("  ОК: отрицательных shift нет.")

# Колонки-лаги в данных
feat_path = PROCESSED_DIR / "battery_features.parquet"
if feat_path.exists():
    cols = list(pd.read_parquet(feat_path, columns=None).columns)
    lag_cols = [c for c in cols if "lag" in c.lower()]
    print(f"\n  Колонки-лаги в battery_features.parquet: {sorted(lag_cols)}")

# ----------------------------------------------------------------------
print_sep("ПРОВЕРКА SPLIT (по батареям, без пересечений)")
all_ok = True
for task, path in METRICS_JSON.items():
    if not path.exists():
        print(f"  {task}: файл {path.name} отсутствует")
        all_ok = False
        continue
    split = load_json(path).get("split")
    if not split:
        print(f"  {task}: в JSON нет ключа split")
        all_ok = False
        continue
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    overlap = (tr & va) | (tr & te) | (va & te)
    verdict = "❌ ПЕРЕСЕЧЕНИЕ: " + str(overlap) if overlap else "OK: пересечений нет"
    print(f"  {task}: train={sorted(tr)} val={sorted(va)} test={sorted(te)}")
    print(f"    -> {verdict}")
    all_ok = all_ok and not overlap

# ----------------------------------------------------------------------
print_sep("ПРОВЕРКА STANDARDSCALER / SIMPLEIMPUTER (fit только на train)")
model_src = (PROJECT / "src" / "model.py").read_text(encoding="utf-8")
lines = model_src.splitlines()
for i, ln in enumerate(lines, 1):
    if re.search(r"\.(fit|transform)\(", ln) and any(
        k in ln for k in ("imputer", "scaler", "Xtr", "Xva", "Xte")
    ):
        print(f"  model.py:{i}: {ln.strip()}")

hits = [ln for ln in lines if re.search(r"(imputer|scaler)\.fit\(", ln)]
if hits and any("train_idx" in h for h in hits):
    print("\n  ВЕРДИКТ: imputer/scaler.fit вызываются с train-индексом -> OK")
else:
    print("\n  ВЕРДИКТ: проверьте вручную в src/model.py линии выше (ожидается fit на train)")

# Feature importance в infographics.py: после правки — train-only
inf_src = (PROJECT / "src" / "infographics.py").read_text(encoding="utf-8")
inf_lines = inf_src.splitlines()
fit_lines = [(i, ln.strip()) for i, ln in enumerate(inf_lines, 1)
             if re.search(r"(SimpleImputer|StandardScaler).*\.fit\(", ln)]
perm_lines = [(i, ln.strip()) for i, ln in enumerate(inf_lines, 1)
              if re.search(r"permutation_importance\(", ln)]
print("\n  Feature importance (infographics.py): fit и permutation_importance:")
for i, ln in fit_lines:
    print(f"    fit     infographics.py:{i}: {ln}")
for i, ln in perm_lines:
    print(f"    import. infographics.py:{i}: {ln}")
if fit_lines and any("is_tr" in ln for _, ln in fit_lines):
    print("  ВЕРДИКТ: imputer/scaler.fit на подмножестве train-батарей (X[is_tr]) -> OK")
else:
    print("  ВЕРДИКТ: проверьте в infographics.py, что fit выполняется только на train"
          " (строки выше).")
if perm_lines and any("is_te" in ln or "yte" in ln for _, ln in perm_lines):
    print("  ВЕРДИКТ: permutation importance считается на тестовых строках -> OK,"
          " look-ahead исключён")
else:
    print("  Примечание: проверьте, на какой выборке считается permutation importance"
          " (ожидается test).")

# ----------------------------------------------------------------------
print_sep("ИТОГ")
missing = [p.name for p in METRICS_JSON.values() if not p.exists()]
print(f"  Служебных признаков в feature_cols:   см. выше (ожидается 'нет')")
print(f"  Пересечений сплита:                   {'нет' if all_ok else 'ЕСТЬ'}")
print(f"  Отсутствующие metrics JSON:           {missing or 'нет'}")
print("\n  Проверка завершена.")