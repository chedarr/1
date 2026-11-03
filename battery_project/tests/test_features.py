"""Тесты признаков (Лаб. №1): лаги, производные, цели B/C."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import clean
from src.config import RAW_DIR
from src.features import build_feature_table
from src.parser import parse_battery


@pytest.fixture(scope="module")
def feature_frame():
    raw = pd.concat(
        [parse_battery(b, RAW_DIR / f"{b}.mat") for b in ("B0005", "B0006")],
        ignore_index=True,
    )
    cleaned = clean(raw)
    return build_feature_table(cleaned)


def test_has_lag_and_derived_columns(feature_frame):
    for col in (
        "capacity_ratio", "discharge_num", "total_cycles", "discharge_rate",
        "voltage_mean_lag1", "voltage_mean_lag2",
        "capacity_lag1", "capacity_lag2", "capacity_lag3",
        "voltage_diff", "capacity_diff", "capacity_slope",
        "voltage_mean_next", "capacity_next",
    ):
        assert col in feature_frame.columns, col


def test_total_cycles_is_max_discharge_num(feature_frame):
    for b, g in feature_frame.groupby("battery_id", sort=False):
        assert (g["total_cycles"] == g["discharge_num"].max()).all(), b
        assert g["total_cycles"].iloc[0] == len(g) - 1, b


def test_lag_is_previous_cycle_within_battery(feature_frame):
    g = feature_frame.sort_values(["battery_id", "cycle_index"])
    for b, grp in g.groupby("battery_id", sort=False):
        lags = grp["voltage_mean_lag1"].to_numpy()
        vals = grp["voltage_mean"].to_numpy()
        assert np.isnan(lags[0]), b
        assert np.allclose(lags[1:], vals[:-1]), b
        lags_c = grp["capacity_lag1"].to_numpy()
        vals_c = grp["capacity"].to_numpy()
        assert np.isnan(lags_c[0]), b
        assert np.allclose(lags_c[1:], vals_c[:-1]), b


def test_target_next_is_next_cycle(feature_frame):
    g = feature_frame.sort_values(["battery_id", "cycle_index"])
    for b, grp in g.groupby("battery_id", sort=False):
        nxt = grp["voltage_mean_next"].to_numpy()
        vals = grp["voltage_mean"].to_numpy()
        assert np.isnan(nxt[-1]), b
        assert np.allclose(nxt[:-1], vals[1:]), b
        nxt_c = grp["capacity_next"].to_numpy()
        vals_c = grp["capacity"].to_numpy()
        assert np.isnan(nxt_c[-1]), b
        assert np.allclose(nxt_c[:-1], vals_c[1:]), b


def test_target_nan_only_on_last_cycle(feature_frame):
    for col in ("voltage_mean_next", "capacity_next"):
        for b, g in feature_frame.groupby("battery_id"):
            m = g[g[col].isna()]
            assert len(m) == 1
            assert m["cycle_index"].iloc[0] == g["cycle_index"].max()


def test_capacity_ratio_in_range(feature_frame):
    ratio = feature_frame["capacity_ratio"].dropna()
    assert ((ratio > 0) & (ratio <= 1.1)).all()
    assert feature_frame["capacity_ratio"].notna().all()


def test_voltage_diff_consistent(feature_frame):
    vd = feature_frame["voltage_diff"]
    exp = feature_frame["voltage_mean"] - feature_frame["voltage_mean_lag1"]
    assert np.allclose(vd.fillna(-999), exp.fillna(-999), equal_nan=True)