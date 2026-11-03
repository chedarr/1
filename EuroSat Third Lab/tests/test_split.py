# -*- coding: utf-8 -*-
"""Проверки стратифицированного split."""
import numpy as np

from src import config
from src.split import (
    build_split_info,
    check_no_overlap,
    check_stratification,
    stratified_split,
)


def _labels(n_per_class=20, n_classes=10):
    return np.repeat(np.arange(n_classes), n_per_class)


def test_split_sizes_and_order():
    labels = _labels()
    splits = stratified_split(labels, seed=config.SEED)
    assert set(splits) == {"train", "val", "test"}
    total = sum(idx.size for idx in splits.values())
    assert total == labels.size
    assert splits["train"].size == int(round(0.70 * labels.size))
    for idx in splits.values():
        assert np.all(np.diff(idx) > 0)


def test_split_no_overlap():
    labels = _labels()
    splits = stratified_split(labels, seed=config.SEED)
    check_no_overlap(splits)
    all_idx = np.concatenate([splits[k] for k in splits])
    assert np.unique(all_idx).size == labels.size


def test_split_is_stratified():
    labels = _labels()
    splits = stratified_split(labels, seed=config.SEED)
    shares = check_stratification(labels, splits)
    for name in splits:
        for class_id, share in shares[name].items():
            expected = np.bincount(labels, minlength=10)[class_id] / labels.size
            assert abs(share - expected) <= 0.005


def test_split_deterministic_with_seed():
    labels = _labels()
    a = stratified_split(labels, seed=config.SEED)
    b = stratified_split(labels, seed=config.SEED)
    for name in a:
        assert np.array_equal(a[name], b[name])


def test_split_info_uses_indices_not_paths():
    labels = _labels()
    splits = stratified_split(labels, seed=config.SEED)
    info = build_split_info(labels, splits, seed=config.SEED)
    assert len(info["index"]) == labels.size
    assert set(info["split"]) == {"train", "val", "test"}
    assert all(isinstance(i, int) and 0 <= i < labels.size for i in info["index"])
    assert "\\\\" not in str(info) and "/" not in str(info["class_id"])
