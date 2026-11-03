# -*- coding: utf-8 -*-
"""Проверки нормализации: статистики считаются по train и применяются без утечки."""
import numpy as np

from src.normalize import compute_channel_stats, load_stats, normalize, save_stats


def test_channel_stats_match_formula():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(7, 4, 4, 3), dtype=np.uint8)
    stats = compute_channel_stats(images)
    flat = images.reshape(-1, 3).astype(np.float64)
    assert np.allclose(stats["mean_rgb"], flat.mean(axis=0), atol=1e-4)
    assert np.allclose(stats["std_rgb"], flat.std(axis=0), atol=1e-4)
    assert stats["n_pixels"] == 7 * 4 * 4
    assert stats["source_split"] == "train"


def test_stats_roundtrip(tmp_path):
    images = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    stats = compute_channel_stats(images)
    path = tmp_path / "normalization_stats.json"
    save_stats(stats, path)
    loaded = load_stats(path)
    assert np.allclose(loaded["mean_rgb"], stats["mean_rgb"])
    assert np.allclose(loaded["std_rgb"], stats["std_rgb"])


def test_normalize_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(50, 8, 8, 3), dtype=np.uint8)
    stats = compute_channel_stats(images)
    out = normalize(images, stats)
    assert out.dtype == np.float32
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(out.reshape(-1, 3).std(axis=0), 1.0, atol=1e-3)


def test_stats_computed_from_subset_do_not_change():
    """Статистики считаются только по переданным (train) снимкам."""
    train = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    val = np.full((3, 2, 2, 3), 255, dtype=np.uint8)
    stats = compute_channel_stats(train)
    expected = train.reshape(-1, 3).astype(np.float64).mean(axis=0)
    assert np.allclose(stats["mean_rgb"], expected)
    # Если бы val попал в расчёт, среднее было бы заметно выше.
    assert stats["mean_rgb"][0] < expected[0] + 1.0
    assert normalize(val, stats).max() > 1.0
