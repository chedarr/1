"""
Полная проверка парсера: 7 проверок.
Запуск: python scripts/check_parser.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np

# Ищем raw_cycles.parquet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "raw_cycles.parquet",
    PROJECT_ROOT / "data" / "interim" / "raw_cycles.parquet",
]

RAW_CYCLES = None
for p in CANDIDATES:
    if p.exists():
        RAW_CYCLES = p
        break

if RAW_CYCLES is None:
    print("Нет raw_cycles.parquet. Сначала запусти парсер.")
    sys.exit(1)

df = pd.read_parquet(RAW_CYCLES)
print(f"Файл: {RAW_CYCLES}")
print(f"Строк: {len(df)}, Колонок: {len(df.columns)}\n")

print("=" * 70)
print("ПРОВЕРКА 1: ЧИСЛО DISCHARGE-ЦИКЛОВ")
print("=" * 70)
print(df.groupby("battery_id").size().to_string())

print("\n" + "=" * 70)
print("ПРОВЕРКА 2: ЁМКОСТЬ УБЫВАЕТ?")
print("=" * 70)
for b in sorted(df["battery_id"].unique()):
    g = df[df["battery_id"] == b].sort_values("cycle_index")
    cap = g["capacity"].dropna().values
    if len(cap) < 4:
        print(f"{b}: мало данных ({len(cap)})")
        continue
    n = len(cap)
    first = np.mean(cap[:n//3])
    last = np.mean(cap[-n//3:])
    trend = "OK" if last < first else "!! РАСТЁТ !!"
    print(f"{b}: first={first:.3f}, last={last:.3f} -> {trend}")

print("\n" + "=" * 70)
print("ПРОВЕРКА 3: ДИАПАЗОНЫ ЗНАЧЕНИЙ")
print("=" * 70)
for col in ["voltage_mean", "capacity", "temp_mean", "current_mean", "re", "rct"]:
    if col not in df.columns:
        print(f"{col}: НЕТ КОЛОНКИ")
        continue
    s = df[col].dropna()
    if len(s) == 0:
        print(f"{col}: все NaN")
        continue
    print(f"{col}: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, n={len(s)}")

print("\n" + "=" * 70)
print("ПРОВЕРКА 4: CYCLE_INDEX МОНОТОНЕН?")
print("=" * 70)
for b in sorted(df["battery_id"].unique()):
    g = df[df["battery_id"] == b].sort_values("cycle_index")
    idx = list(g["cycle_index"].values)
    expected = list(range(len(idx)))
    status = "OK" if idx == expected else "!! НЕ МОНОТОНЕН !!"
    print(f"{b}: {status} ({len(idx)} циклов)")

print("\n" + "=" * 70)
print("ПРОВЕРКА 5: MERGE_ASOF ЗАПОЛНИЛ RE/RCT/DURATION_CHARGE?")
print("=" * 70)
for col in ["re", "rct", "duration_charge"]:
    if col not in df.columns:
        print(f"{col}: НЕТ КОЛОНКИ")
        continue
    n_valid = df[col].notna().sum()
    print(f"{col}: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")

print("\n" + "=" * 70)
print("ПРОВЕРКА 6: ДУБЛИКАТЫ")
print("=" * 70)
dupes = df.duplicated(subset=["battery_id", "cycle_index"]).sum()
print(f"Дубликатов: {dupes}")

print("\n" + "=" * 70)
print("ПРОВЕРКА 7: ВСЕ КОЛОНКИ НА МЕСТЕ?")
print("=" * 70)
expected = [
    "battery_id", "cycle_index", "capacity",
    "voltage_mean", "voltage_min", "voltage_max", "voltage_std", "voltage_range",
    "current_mean", "temp_mean", "temp_max", "temp_rise",
    "duration_discharge", "duration_charge", "re", "rct",
]
missing = [c for c in expected if c not in df.columns]
if missing:
    print("ОТСУТСТВУЮТ:", missing)
else:
    print("Все колонки на месте.")
print(f"\nВсего колонок: {len(df.columns)}")
print(f"Колонки: {list(df.columns)}")