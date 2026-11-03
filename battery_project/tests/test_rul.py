"""Тесты RUL (Лаб. №1): RUL_cycles = max(discharge_num) - discharge_num,
только для 8 батарей с полной кривой."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import clean
from src.config import RAW_DIR, RUL_BATTERIES
from src.features import build_feature_table
from src.parser import parse_battery
from src.rul import compute_rul, make_rul_dataset


@pytest.fixture(scope="module")
def rul_frame():
    raw = pd.concat(
        [parse_battery(b, RAW_DIR / f"{b}.mat") for b in ("B0005", "B0006")],
        ignore_index=True,
    )
    feat = build_feature_table(clean(raw))
    return compute_rul(feat, batteries=["B0005", "B0006"])


def test_rul_equals_remaining_cycles(rul_frame):
    g = rul_frame[rul_frame["battery_id"] == "B0005"].sort_values("discharge_num")
    n = len(g)
    assert g["rul_cycles"].iloc[0] == n - 1
    assert g["rul_cycles"].iloc[-1] == 0


def test_rul_nonnegative(rul_frame):
    m = rul_frame.dropna(subset=["rul_cycles"])
    assert (m["rul_cycles"] >= 0).all()


def test_rul_decreasing(rul_frame):
    for b, g in rul_frame.groupby("battery_id"):
        g = g.dropna(subset=["rul_cycles"]).sort_values("discharge_num")
        assert (g["rul_cycles"].diff().dropna() <= 0).all()
        assert (g["rul_cycles"].diff().dropna() < 0).any()


def test_rul_frac_in_unit_range(rul_frame):
    m = rul_frame.dropna(subset=["rul_frac"])
    assert (m["rul_frac"] >= 0).all()
    assert (m["rul_frac"] <= 1).all()
    g = rul_frame[rul_frame["battery_id"] == "B0005"].dropna(subset=["rul_frac"])
    assert g["rul_frac"].iloc[0] == pytest.approx(1.0)
    assert g["rul_frac"].iloc[-1] == pytest.approx(0.0)


def test_rul_frac_matches_cycles_over_total(rul_frame):
    m = rul_frame.dropna(subset=["rul_frac", "rul_cycles", "total_cycles"])
    expected = m["rul_cycles"] / m["total_cycles"]
    assert np.allclose(m["rul_frac"], expected)


def test_rul_excludes_non_rul_batteries():
    raw = pd.concat(
        [parse_battery(b, RAW_DIR / f"{b}.mat") for b in ("B0005", "B0025", "B0050")],
        ignore_index=True,
    )
    feat = build_feature_table(clean(raw))
    dset = make_rul_dataset(feat)
    assert set(dset["battery_id"].unique()) == {"B0005"}
    assert "B0025" not in dset["battery_id"].values
    assert "B0050" not in dset["battery_id"].values


def test_rul_battery_list_matches_spec():
    assert set(RUL_BATTERIES) == {
        "B0005", "B0006", "B0018", "B0045", "B0046", "B0047", "B0048", "B0053",
    }