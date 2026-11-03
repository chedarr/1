"""Сплит строго по батареям (§5): GroupShuffleSplit дважды.

    1. test_size=0.2 → test;
    2. внутри оставшихся test_size=0.2 → val.

groups = battery_id, random_state = 42. Пересечение групп между
train/val/test — пусто (AssertionError при нарушении).

Запрещено: стратификация по RUL, «хвост train», деление по порядку ID.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .config import RANDOM_STATE, SPLIT_PATH


def make_split(battery_ids, random_state: int = RANDOM_STATE, test_size: float = 0.2):
    """Сплит списка батарей → (train, val, test), каждая — отсортированный список."""
    ids = np.array(sorted(battery_ids), dtype=object)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(gss1.split(ids, groups=ids))
    trainval, test = ids[tr_idx].tolist(), ids[te_idx].tolist()

    tv_arr = np.array(sorted(trainval), dtype=object)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx2, va_idx2 = next(gss2.split(tv_arr, groups=tv_arr))
    train, val = tv_arr[tr_idx2].tolist(), tv_arr[va_idx2].tolist()

    _assert_no_overlap(train, val, test, battery_ids)
    return sorted(train), sorted(val), sorted(test)


def _assert_no_overlap(train, val, test, battery_ids):
    assert set(train).isdisjoint(set(val)), "пересечение train/val"
    assert set(train).isdisjoint(set(test)), "пересечение train/test"
    assert set(val).isdisjoint(set(test)), "пересечение val/test"
    covered = set(train) | set(val) | set(test)
    assert covered == set(battery_ids), f"не покрыты: {set(battery_ids) - covered}"


def build_split(battery_ids, out_path=None) -> dict:
    """Сплит + сохранение split.json."""
    train, val, test = make_split(list(battery_ids))
    result = {
        "train": train,
        "val": val,
        "test": test,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
    }
    path = out_path or SPLIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    ids = [f"B{i:04d}" for i in range(22)]
    print(build_split(ids))