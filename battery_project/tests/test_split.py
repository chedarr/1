"""Тесты сплита (§8): пересечение групп пусто, все батареи покрыты.

Проверяет рецепт из §5 (GroupShuffleSplit дважды, random_state=42) на
синтетическом наборе групп — регрессионная защита функции сплита.
"""
from __future__ import annotations

import numpy as np

from src.split import make_split


def test_no_group_overlap():
    ids = [f"B{i:04d}" for i in range(22)]   # столько же, сколько батарей с EOL
    train, val, test = make_split(ids, random_state=42)
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))


def test_all_batteries_covered():
    ids = [f"B{i:04d}" for i in range(22)]
    train, val, test = make_split(ids, random_state=42)
    assert set(train) | set(val) | set(test) == set(ids)


def test_sizes():
    ids = [f"B{i:04d}" for i in range(22)]
    train, val, test = make_split(ids, random_state=42)
    # GroupShuffleSplit: ceil(0.2 * 22) = 5 в test, затем ceil(0.2 * 17) = 4 в val
    assert len(test) == 5
    assert len(val) == 4
    assert len(train) == 13


def test_deterministic():
    ids = [f"B{i:04d}" for i in range(22)]
    a = make_split(ids, random_state=42)
    b = make_split(ids, random_state=42)
    assert a == b


def test_small_subset_ok():
    # минимум 2 группы: test и val не могут быть пустыми
    ids = ["B0006", "B0045", "B0046", "B0047"]
    train, val, test = make_split(ids, random_state=42)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0