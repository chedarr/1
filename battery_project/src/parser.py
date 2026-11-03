"""Парсер .mat файлов NASA PCoE → таблица discharge-циклов.

Каждый .mat содержит структуру <BATTERY_ID>.cycle — массив циклов:
    cycle.type             — 'charge' | 'discharge' | 'impedance'
    cycle.ambient_temperature
    cycle.time             — вектор [год, мес, день, час, мин, сек]
    cycle.data             — измерения:
        discharge: Voltage_measured, Current_measured, Temperature_measured,
                   Current_load, Voltage_load, Time, Capacity
        charge:    Voltage_measured, Current_measured, Temperature_measured,
                   Current_charge, Voltage_charge, Time
        impedance: Re, Rct (+ комплексные массивы)

Выход: одна строка на discharge-цикл (в порядке записи в файле, cycle_index 0-based),
с агрегированной статистикой измерений этого цикла и ближайшими предыдущими
событиями (merge_asof, direction='backward', внутри батареи):
    duration_charge — длительность предыдущего charge-цикла,
    re, rct         — значения импеданса из предыдущего impedance-цикла.

Строки discharge-циклов без измеренной ёмкости (пустой массив Capacity)
отбрасываются: по ним невозможно определить RUL/признаки.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import ALL_BATTERIES, METADATA_AMBIENT_C, RAW_CYCLES_PATH, RAW_DIR


def _scalar(values) -> float:
    """Скалярное значение из 0-d/1-d массива mat_struct; NaN если пусто/нет."""
    arr = np.asarray(values).ravel()
    if arr.size == 0:
        return float("nan")
    try:
        return float(np.real(arr[0]))
    except (TypeError, ValueError):
        return float("nan")


def _cycle_date(cycle) -> pd.Timestamp:
    t = np.asarray(cycle.time).ravel()
    if t.size < 6:
        return pd.NaT
    try:
        y, mo, d, h, mi = (int(x) for x in t[:5])
        s = min(max(int(round(float(t[5]))), 0), 59)
        return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, second=s)
    except (ValueError, OverflowError, TypeError):
        return pd.NaT


def _mean_var(values) -> tuple[float, float, float, float]:
    """mean / min / max / std по конечным значениям; NaN если пусто."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr)), float(np.std(arr))


def _first(values) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr[0]) if arr.size else float("nan")


def _span_seconds(values) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr) - np.min(arr))


def parse_battery(battery_id: str, mat_path) -> pd.DataFrame:
    """Распарсить один .mat → DataFrame discharge-циклов батареи."""
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    key = [k for k in mat if not k.startswith("__")][0]
    cycles = getattr(mat[key], "cycle")

    discharge_rows: list[dict] = []
    charge_events: list[dict] = []      # (date, duration)
    impedance_events: list[dict] = []   # (date, re, rct)

    for cycle in cycles:
        ctype = str(cycle.type)
        data = cycle.data
        date = _cycle_date(cycle)

        if ctype == "charge":
            duration = _span_seconds(data.Time)
            charge_events.append({"date": date, "duration_charge": duration})

        elif ctype == "impedance":
            re, rct = _scalar(getattr(data, "Re", None)), _scalar(getattr(data, "Rct", None))
            impedance_events.append({"date": date, "re": re, "rct": rct})

        elif ctype == "discharge":
            capacity = _scalar(data.Capacity)
            if not np.isfinite(capacity):
                # нет измерения ёмкости — непригодная запись, пропускаем
                continue
            ambient = _scalar(cycle.ambient_temperature)
            if not np.isfinite(ambient):
                ambient = METADATA_AMBIENT_C.get(battery_id, 24.0)

            v_mean, v_min, v_max, v_std = _mean_var(data.Voltage_measured)
            i_mean, _, _, i_std = _mean_var(data.Current_measured)
            i_mean_abs = float(np.mean(np.abs(np.asarray(data.Current_measured, dtype=float).ravel())))
            t_mean, t_min, t_max, _ = _mean_var(data.Temperature_measured)
            t_start = _first(data.Temperature_measured)
            time_total = _span_seconds(data.Time)

            discharge_rows.append({
                "battery_id": battery_id,
                "cycle_index": np.nan,          # заполним ниже по порядку
                "date": date,
                "ambient_temperature": ambient,
                "capacity": capacity,
                "voltage_mean": v_mean,
                "voltage_min": v_min,
                "voltage_max": v_max,
                "voltage_std": v_std,
                "voltage_range": v_max - v_min,
                "current_mean": i_mean,
                "current_std": i_std,
                "current_mean_abs": i_mean_abs,
                "temp_mean": t_mean,
                "temp_min": t_min,
                "temp_max": t_max,
                "temp_range": t_max - t_min,
                "temp_rise": t_max - t_start,
                "time_total": time_total,
                "duration_discharge": time_total,
            })

    df = pd.DataFrame(discharge_rows)
    if df.empty:
        return df
    df["cycle_index"] = np.arange(len(df), dtype=int)

    # ближайший предыдущий charge-цикл внутри батареи (merge_asof, backward)
    charge_df = pd.DataFrame(charge_events).dropna(subset=["date"])
    if not charge_df.empty:
        df = pd.merge_asof(
            df.sort_values("date"), charge_df.sort_values("date"),
            on="date", direction="backward",
        )
    else:
        df["duration_charge"] = float("nan")
    # ближайший предыдущий impedance-цикл внутри батареи (merge_asof, backward)
    imp_df = pd.DataFrame(impedance_events).dropna(subset=["date"])
    if not imp_df.empty:
        df = pd.merge_asof(
            df.sort_values("date"), imp_df.sort_values("date"),
            on="date", direction="backward",
        )
    else:
        df["re"] = float("nan")
        df["rct"] = float("nan")
    # фиксируем канонический порядок строк по записи в файле
    df = df.sort_values("cycle_index").reset_index(drop=True)
    return df


def parse_all(battery_ids: list[str] | None = None, out_path=None) -> pd.DataFrame:
    """Распарсить все .mat → общая таблица discharge-циклов."""
    battery_ids = battery_ids or list(ALL_BATTERIES)
    frames = []
    for bid in battery_ids:
        df = parse_battery(bid, RAW_DIR / f"{bid}.mat")
        print(f"  {bid}: {len(df)} discharge-строк")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    return out


if __name__ == "__main__":
    df = parse_all(out_path=RAW_CYCLES_PATH)
    print("Всего строк:", len(df))
    print("Колонки:", list(df.columns))