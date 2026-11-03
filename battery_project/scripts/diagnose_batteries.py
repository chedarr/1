"""
Диагностика батарей NASA PCoE Battery.

Работает с таблицей discharge-циклов, которую создаёт parser.py.
Колонки: battery_id, cycle_index, capacity, ...

Считает EOL тремя формулами:
   - median(3:8)
   - robust max
   - nominal 2.0 Ahr
и показывает, сколько батарей проходят фильтр rows_to_eol >= 10.

Запуск:
    cd D:\\battery_project
    python scripts\\diagnose_batteries.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
from config import ALL_BATTERIES

# ---- Ищем raw_cycles.parquet ----
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
    print("Нет raw_cycles.parquet ни в data/processed, ни в data/interim.")
    print("Сначала запусти парсер:")
    print("    python scripts/run_parser.py")
    sys.exit(1)

print(f"Использую: {RAW_CYCLES}\n")

OUT = PROJECT_ROOT / "data" / "processed" / "battery_diagnostics.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- Загружаем ----
df = pd.read_parquet(RAW_CYCLES)
print(f"Строк: {len(df)}")
print(f"Колонок: {len(df.columns)}")
print(f"Колонки: {list(df.columns)}\n")

# ---- Определяем имена колонок ----
# Парсер использует battery_id и cycle_index
battery_col = "battery_id" if "battery_id" in df.columns else "battery"
cycle_col = "cycle_index" if "cycle_index" in df.columns else "discharge_num"

if battery_col not in df.columns:
    print(f"!!! Нет колонки '{battery_col}' — не могу диагностировать.")
    sys.exit(1)

if "capacity" not in df.columns:
    print("!!! Нет колонки 'capacity' — не могу диагностировать.")
    sys.exit(1)

# Нормализуем: переименуем в battery / cycle_index для удобства
df = df.rename(columns={battery_col: "battery", cycle_col: "cycle_index"})

# Убедимся, что cycle_index существует и монотонен
df = df.sort_values(["battery", "cycle_index"]).reset_index(drop=True)

# ---- Диагностика по каждой батарее ----
rows = []
for b in sorted(df["battery"].unique()):
    g = df[df["battery"] == b].copy()
    cap = g["capacity"].dropna().values
    n = len(cap)
    if n == 0:
        rows.append({
            "battery": b, "n_cycles": 0, "note": "нет capacity",
            "cap0_med": None, "cap0_robust": None,
            "eol_med": None, "eol_robust": None, "eol_nominal": None,
            "rows_to_eol_med": 0, "rows_to_eol_robust": 0, "rows_to_eol_nominal": 0,
        })
        continue

    # Формула 1: median(3:8)
    cap0_med = float(np.median(cap[2:8])) if n >= 8 else float(np.median(cap))

    # Формула 2: robust max (max среди capacity <= 1.5 * p90)
    p90 = float(np.percentile(cap, 90))
    valid = cap[cap <= 1.5 * p90]
    cap0_robust = float(np.max(valid)) if len(valid) else float("nan")

    # Формула 3: номинал
    cap0_nom = 2.0

    def eol_rows(cap0):
        if cap0 is None or np.isnan(cap0) or cap0 <= 0:
            return None
        threshold = 0.7 * cap0
        below = np.where(cap <= threshold)[0]
        return int(below[0]) if len(below) else None

    eol_med = eol_rows(cap0_med)
    eol_rob = eol_rows(cap0_robust)
    eol_nom = eol_rows(cap0_nom)

    rows.append({
        "battery": b,
        "n_cycles": n,
        "note": "",
        "cap0_med": round(cap0_med, 4),
        "cap0_robust": round(cap0_robust, 4) if not np.isnan(cap0_robust) else None,
        "eol_med": eol_med,
        "eol_robust": eol_rob,
        "eol_nominal": eol_nom,
        "rows_to_eol_med": eol_med if eol_med is not None else 0,
        "rows_to_eol_robust": eol_rob if eol_rob is not None else 0,
        "rows_to_eol_nominal": eol_nom if eol_nom is not None else 0,
    })

diag = pd.DataFrame(rows)

# ---- Вывод ----
print("=" * 120)
print("ДИАГНОСТИКА БАТАРЕЙ")
print("=" * 120)
print(diag.to_string(index=False))

print("\n" + "=" * 120)
print("БАТАРЕИ С rows_to_eol >= 10")
print("=" * 120)

for col in ["rows_to_eol_med", "rows_to_eol_robust", "rows_to_eol_nominal"]:
    passed = diag[diag[col] >= 10]
    print(f"\n{col}: {len(passed)} батарей")
    if len(passed):
        print(f"  {list(passed['battery'])}")

diag.to_csv(OUT, index=False)
print(f"\nСохранено: {OUT}")