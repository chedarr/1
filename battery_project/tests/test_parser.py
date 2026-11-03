"""Тесты парсера (Лаб. №1)."""
from __future__ import annotations

from src.config import ALL_BATTERIES, RAW_DIR
from src.parser import parse_battery


def test_parse_b0005_nonempty():
    df = parse_battery("B0005", RAW_DIR / "B0005.mat")
    assert len(df) > 0


def test_parse_b0005_columns():
    df = parse_battery("B0005", RAW_DIR / "B0005.mat")
    assert "capacity" in df.columns
    assert "voltage_mean" in df.columns
    assert df["capacity"].notna().all()


def test_parse_b0005_cycle_index_consecutive():
    df = parse_battery("B0005", RAW_DIR / "B0005.mat")
    ci = df["cycle_index"].to_numpy()
    assert (ci == range(len(df))).all()


def test_parse_b0005_extra_measurements():
    """Признаки задачи A: temp_min, temp_range, time_total присутствуют."""
    df = parse_battery("B0005", RAW_DIR / "B0005.mat")
    for col in ("temp_min", "temp_range", "time_total"):
        assert col in df.columns
    assert (df["temp_min"] <= df["temp_max"]).all()
    assert (df["time_total"] > 0).all()


def test_parse_all_batteries_ok():
    for bid in ALL_BATTERIES:
        df = parse_battery(bid, RAW_DIR / f"{bid}.mat")
        assert len(df) > 0, f"{bid} пуст"
        assert df["capacity"].notna().all(), f"{bid}: есть пустые ёмкости"