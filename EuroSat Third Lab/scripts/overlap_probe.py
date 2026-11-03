# -*- coding: utf-8 -*-
"""Косвенная проверка потенциального overlap по сцене.

Идея: снимки одной сцены Sentinel-2 (соседние патчи) похожи визуально.
Считаем cosine similarity между эмбеддингами test и train (128-мерный слой
обученной CNN) и ищем аномально близкие пары. Это НЕ доказательство overlap,
а честная косвенная оценка: scene_id в датасете отсутствует.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src import config
from src.cnn import make_dataset
from src.loader import load_arrays
from src.normalize import load_stats


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def dhash(image: np.ndarray) -> np.uint64:
    """dHash 64 бита: градиенты соседних пикселей на сетке 9x8."""
    from PIL import Image

    gray = Image.fromarray(image).convert("L").resize((9, 8), Image.BILINEAR)
    values = np.asarray(gray, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]


def hamming_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Попарные расстояния Хэмминга между двумя наборами 64-битных хэшей."""
    xor = np.bitwise_xor(left[:, None], right[None, :])
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(xor).astype(np.int16)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    bytes_view = xor.view(np.uint8).reshape(xor.shape + (8,))
    return table[bytes_view].sum(axis=-1).astype(np.int16)


def main() -> int:
    config.configure_runtime()
    t0 = time.time()

    images, _ = load_arrays()
    info = json.load(open(config.REPORTS / "split_info.json", encoding="utf-8"))
    idx = np.asarray(info["index"], dtype=np.int64)
    split = np.asarray(info["split"])
    labels = np.asarray(info["class_id"], dtype=np.int64)
    train_idx = idx[split == "train"]
    test_idx = idx[split == "test"]

    model_path = config.REPORTS / "control_patience_max" / "eurosat_cnn_patience_max.keras"
    model = tf.keras.models.load_model(model_path)
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    embedding_layer = dense_layers[0]
    print(f"слой эмбеддингов: {embedding_layer.name} units={embedding_layer.units}", flush=True)
    encoder = tf.keras.Model(model.input, embedding_layer.output)

    stats = load_stats()
    train_ds = make_dataset(images[train_idx], labels[train_idx], stats, training=False)
    test_ds = make_dataset(images[test_idx], labels[test_idx], stats, training=False)
    train_emb = l2_normalize(encoder.predict(train_ds, verbose=0).astype(np.float32))
    test_emb = l2_normalize(encoder.predict(test_ds, verbose=0).astype(np.float32))
    print(f"эмбеддинги: train {train_emb.shape}, test {test_emb.shape}", flush=True)

    top1_sim = np.empty(test_emb.shape[0], dtype=np.float32)
    top1_idx = np.empty(test_emb.shape[0], dtype=np.int64)
    chunk = 256
    for start in range(0, test_emb.shape[0], chunk):
        block = test_emb[start:start + chunk] @ train_emb.T
        top1_idx[start:start + chunk] = block.argmax(axis=1)
        top1_sim[start:start + chunk] = block.max(axis=1)

    y_train = labels[train_idx]
    y_test = labels[test_idx]
    same_class = y_train[top1_idx] == y_test
    counts = {
        "gt_0.95": int(np.sum(top1_sim > 0.95)),
        "gt_0.99": int(np.sum(top1_sim > 0.99)),
        "gt_0.999": int(np.sum(top1_sim > 0.999)),
    }
    # Контроль: similarity случайных пар test-train. Если она близка к top1,
    # метрика неинформативна (эмбеддинги коллапсируют).
    rng = np.random.default_rng(config.SEED)
    n_pairs = 200_000
    rand_test = rng.integers(0, test_emb.shape[0], size=n_pairs)
    rand_train = rng.integers(0, train_emb.shape[0], size=n_pairs)
    random_sim = np.sum(test_emb[rand_test] * train_emb[rand_train], axis=1)

    # Контроль 2: top1 внутри train (holdout-подобная ссылка).
    sample_train = rng.choice(train_emb.shape[0], size=1000, replace=False)
    sample_sim = train_emb[sample_train] @ train_emb.T
    sample_sim[np.arange(sample_train.size), sample_train] = -1.0
    train_top1 = sample_sim.max(axis=1)

    # dHash: почти-дубликаты между test и train (сигнал overlap по сцене).
    print("dHash: считаю хэши...", flush=True)
    hashes = np.array([dhash(image) for image in images], dtype=np.uint64)
    h_train, h_test = hashes[train_idx], hashes[test_idx]
    min_hamming = np.empty(h_test.size, dtype=np.int16)
    near_pairs: list[tuple[int, int, int]] = []
    for start in range(0, h_test.size, 256):
        block = hamming_matrix(h_test[start:start + 256], h_train)
        row_min = block.min(axis=1)
        row_arg = block.argmin(axis=1)
        min_hamming[start:start + 256] = row_min
        for offset in np.flatnonzero(row_min <= 4):
            near_pairs.append((start + int(offset), int(row_arg[offset]), int(row_min[offset])))
    pixel_diff = [
        float(np.mean(np.abs(images[test_idx[i]].astype(np.int16)
                            - images[train_idx[j]].astype(np.int16))))
        for i, j, _ in near_pairs
    ]
    phash_stats = {
        "method": "dHash 64 бита (9x8 градиенты), расстояние Хэмминга",
        "min_hamming_min": int(min_hamming.min()),
        "min_hamming_median": int(np.median(min_hamming)),
        "pairs_le_4_bits": int(np.sum(min_hamming <= 4)),
        "pairs_le_2_bits": int(np.sum(min_hamming <= 2)),
        "exact_duplicates": int(np.sum(min_hamming == 0)),
        "classes_le_4_bits": dict(collections.Counter(
            config.CLASS_NAMES[int(labels[test_idx[i]])] for i, _, _ in near_pairs
        ).most_common()),
        "mad_lt_5_class_share": dict(collections.Counter(
            config.CLASS_NAMES[int(labels[test_idx[i]])]
            for (i, _, _), mad in zip(near_pairs, pixel_diff) if mad < 5
        ).most_common()),
        "mad_lt_5_same_class": int(sum(
            1 for (i, j, _), mad in zip(near_pairs, pixel_diff)
            if mad < 5 and labels[test_idx[i]] == labels[train_idx[j]]
        )),
        "near_duplicate_pairs": [
            {
                "test_index": int(test_idx[i]),
                "train_index": int(train_idx[j]),
                "class": config.CLASS_NAMES[int(labels[test_idx[i]])],
                "hamming": int(bits),
                "pixel_mad": round(float(mad), 3),
            }
            for (i, j, bits), mad in zip(near_pairs, pixel_diff) if mad < 5
        ],
        "pixel_mad_for_le_4_bits": {
            "n_pairs": len(near_pairs),
            "min": round(min(pixel_diff), 3) if pixel_diff else None,
            "median": round(float(np.median(pixel_diff)), 3) if pixel_diff else None,
            "lt_5": int(np.sum(np.asarray(pixel_diff) < 5)) if pixel_diff else 0,
            "lt_10": int(np.sum(np.asarray(pixel_diff) < 10)) if pixel_diff else 0,
        },
        "examples_le_2_bits": [
            {
                "test_index": int(test_idx[i]),
                "test_class": config.CLASS_NAMES[int(labels[test_idx[i]])],
                "train_index": int(train_idx[j]),
                "train_class": config.CLASS_NAMES[int(labels[train_idx[j]])],
                "hamming": int(bits),
                "pixel_mad": round(pixel_diff[k], 3),
            }
            for k, (i, j, bits) in enumerate(near_pairs[:10])
            if bits <= 2
        ],
    }
    print(f"dHash: пар <=4 бит {phash_stats['pairs_le_4_bits']}, "
          f"<=2 бит {phash_stats['pairs_le_2_bits']}, 0 бит {phash_stats['exact_duplicates']}")

    order = np.argsort(-top1_sim)[:10]
    examples = [
        {
            "test_index": int(test_idx[i]),
            "test_class": config.CLASS_NAMES[int(y_test[i])],
            "train_index": int(train_idx[top1_idx[i]]),
            "train_class": config.CLASS_NAMES[int(y_train[top1_idx[i]])],
            "similarity": round(float(top1_sim[i]), 6),
            "same_class": bool(same_class[i]),
        }
        for i in order
    ]

    payload = {
        "method": "cosine similarity между 128-мерными эмбеддингами test и train",
        "model": str(model_path.relative_to(config.PROJECT_ROOT)),
        "layer": embedding_layer.name,
        "embedding_dim": int(train_emb.shape[1]),
        "n_train": int(train_emb.shape[0]),
        "n_test": int(test_emb.shape[0]),
        "top1_similarity": {
            "min": round(float(top1_sim.min()), 6),
            "mean": round(float(top1_sim.mean()), 6),
            "median": round(float(np.median(top1_sim)), 6),
            "p95": round(float(np.percentile(top1_sim, 95)), 6),
            "max": round(float(top1_sim.max()), 6),
        },
        "counts_above": counts,
        "control_random_pairs": {
            "n_pairs": int(n_pairs),
            "mean": round(float(random_sim.mean()), 6),
            "median": round(float(np.median(random_sim)), 6),
            "p95": round(float(np.percentile(random_sim, 95)), 6),
            "max": round(float(random_sim.max()), 6),
        },
        "control_train_top1": {
            "n_sample": 1000,
            "median": round(float(np.median(train_top1)), 6),
            "p95": round(float(np.percentile(train_top1, 95)), 6),
            "max": round(float(train_top1.max()), 6),
        },
        "top1_minus_random_mean": round(float(top1_sim.mean() - random_sim.mean()), 6),
        "phash_near_duplicates": phash_stats,
        "same_class_share_above_0.95": (
            round(float(same_class[top1_sim > 0.95].mean()), 4)
            if counts["gt_0.95"] else None
        ),
        "top_pairs": examples,
        "caveat": (
            "Косвенная оценка, не доказательство: CNN обучена на train, поэтому "
            "высокая similarity ожидаема и для разных классов. scene_id в датасете нет."
        ),
        "seconds": round(time.time() - t0, 1),
    }
    out = config.REPORTS / "overlap_probe.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"top1: median={payload['top1_similarity']['median']} "
          f"p95={payload['top1_similarity']['p95']} max={payload['top1_similarity']['max']}")
    print(f"пар >0.95: {counts['gt_0.95']}; >0.99: {counts['gt_0.99']}; >0.999: {counts['gt_0.999']}")
    print(f"ИТОГ: overlap_probe.json за {payload['seconds']} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
