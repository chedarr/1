# -*- coding: utf-8 -*-
r"""Шаг 6. Датасет обучения с частой сеткой окон (шаг 5 вместо 10).

Вход:  data\processed\adapt_raw.parquet, data\processed\adapt_log.parquet
Выход: data\processed\adapt_train_step5.parquet
       data\processed\adapt_step5_report.txt

Правила:
* пересобираются ТОЛЬКО обучающие эксперименты (12 записей), тест не трогается;
* тестовые окна остаются на шаге 10 в adapt_test.parquet — исходный тест сохраняется;
* файл-источник тестовой выборки не переписывается: до и после сборки берётся
  его контрольная сумма, расхождение считается нарушением;
* окна по-прежнему строятся внутри одного эксперимента, поэтому утечки между
  обучением и тестом нет по построению; это дополнительно проверяется.

Запуск:  .venv\Scripts\python.exe scripts\step6_build_step5.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.experiments_adapt import build_windows, load_measurements  # noqa: E402

WINDOW = config.WINDOW_SIZE
STRIDE = config.STRIDE_SMALL
OTHER = ("source_file", "line_no", "record_type", "Time")


def sha256(path: Path) -> str:
    """Контрольная сумма файла: доказательство, что тест не переписан."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_frame(meas: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Обучающие окна с частой сеткой в той же схеме, что и принятый датасет."""
    x5, y5, meta = build_windows(meas, sensors, STRIDE, (0,), test=False)
    by_file = {
        fname: sub.sort_values("line_no").reset_index(drop=True)
        for fname, sub in meas.groupby("source_file", sort=True)
    }
    feat_names = [f"{t:02d}_{s}" for t in range(WINDOW) for s in sensors]
    rows: list[dict] = []
    for i in range(len(y5)):
        fname = meta["meta__source_file"].iloc[i]
        start = int(meta["meta__start_index"].iloc[i])
        end = start + WINDOW - 1
        sub = by_file[fname]
        span = (sub["ts"].iloc[end] - sub["ts"].iloc[start]) / np.timedelta64(1, "s")
        record = {
            "meta__split": "train",
            "meta__experiment": str(meta["meta__experiment"].iloc[i]),
            "meta__source_file": fname,
            "meta__start_index": start,
            "meta__end_index": end,
            "meta__t_start": str(sub["Time"].iloc[start]),
            "meta__t_end": str(sub["Time"].iloc[end]),
            "meta__span_s": float(span),
            "meta__n_fault_steps": int(sub["fault"].iloc[start : end + 1].sum()),
            "fault": int(y5[i]),
        }
        flat = x5[i].reshape(-1)
        record.update({feat_names[j]: float(flat[j]) for j in range(len(feat_names))})
        rows.append(record)
    frame = pd.DataFrame(rows)
    # номер окна внутри эксперимента — как в принятом датасете
    frame.insert(3, "meta__window_id",
                 frame.groupby("meta__experiment").cumcount())
    return frame


def main() -> int:
    log = config.setup_logging()
    config.set_seed()
    config.limit_resources()

    test_path = config.TEST_PATH
    if not test_path.exists():
        log.error("нет тестовой выборки: %s", test_path)
        return 1
    test_size_before = test_path.stat().st_size
    digest_before = sha256(test_path)

    log.info("сборка обучающих окон с шагом %d (эталонный шаг %d)", STRIDE, config.STRIDE)
    meas, sensors = load_measurements()
    frame = build_frame(meas, sensors)
    frame.to_parquet(config.STEP5_PATH, index=False)

    digest_after = sha256(test_path)
    test_untouched = digest_before == digest_after

    test = pd.read_parquet(test_path)
    train_experiments = sorted(set(frame["meta__experiment"]))
    test_experiments = sorted(set(test["meta__experiment"]))
    overlap = sorted(set(train_experiments) & set(test_experiments))
    files_train = set(frame["meta__source_file"])
    files_test = set(test["meta__source_file"])
    files_overlap = sorted(files_train & files_test)

    pairs_train = set(zip(frame["meta__source_file"], frame["meta__start_index"]))
    pairs_test = set(zip(test["meta__source_file"], test["meta__start_index"]))
    pair_overlap = pairs_train & pairs_test

    sizes = meas.groupby("source_file").size().to_dict()
    bad_bounds = int(
        (frame.apply(lambda r: r["meta__end_index"] != r["meta__start_index"] + WINDOW - 1,
                     axis=1)).sum()
        + (frame.apply(lambda r: r["meta__end_index"] >= sizes[r["meta__source_file"]],
                       axis=1)).sum()
    )
    feat_cols = [c for c in frame.columns
                 if not c.startswith(config.META_PREFIX) and c != config.TARGET]
    nan_features = int(frame[feat_cols].isna().sum().sum())
    label_mismatch = 0
    for fname, sub in meas.groupby("source_file", sort=True):
        sub = sub.sort_values("line_no").reset_index(drop=True)
        lab = sub["fault"].to_numpy()
        part = frame[frame["meta__source_file"] == fname]
        label_mismatch += int((part["fault"].to_numpy() != lab[part["meta__end_index"].to_numpy()]).sum())

    # независимая пересборка: окно = срез измерений по своему началу
    step_ok = bool((frame["meta__start_index"] % STRIDE == 0).all())
    lines = [
        "=== ДАТАСЕТ ОБУЧЕНИЯ С ЧАСТОЙ СЕТКОЙ ОКОН ===",
        f"размер окна: {WINDOW} шагов; шаг обучения: {STRIDE}; эталонный шаг: {config.STRIDE}",
        f"окон в новом датасете: {len(frame)}; экспериментов: {len(train_experiments)}",
        f"признаков: {len(feat_cols)}; метка: fault",
        "",
        "=== ПРОВЕРКИ ===",
        f"[{'OK' if not overlap else '!!'}] эксперименты обучения и теста не пересекаются: "
        f"{overlap or 'пусто'}",
        f"[{'OK' if not files_overlap else '!!'}] файлы обучения и теста не пересекаются: "
        f"{files_overlap or 'пусто'}",
        f"[{'OK' if not pair_overlap else '!!'}] совпадающих окон (файл, старт) с тестом: "
        f"{len(pair_overlap)}",
        f"[{'OK' if test_untouched else '!!'}] тестовая выборка не переписана: "
        f"размер {test_size_before} байт, контрольная сумма совпадает: {test_untouched}",
        f"[{'OK' if bad_bounds == 0 else '!!'}] окно целиком внутри своего эксперимента: "
        f"нарушений {bad_bounds}",
        f"[{'OK' if step_ok else '!!'}] все начала окон кратны шагу {STRIDE}",
        f"[{'OK' if label_mismatch == 0 else '!!'}] метка равна состоянию последнего шага окна: "
        f"расхождений {label_mismatch}",
        f"[{'OK' if nan_features == 0 else '!!'}] пропусков в признаках: {nan_features}",
        "",
        "=== ОБЪЁМ ===",
        f"обучение: окон {len(frame)} (норма {int((frame['fault'] == 0).sum())}, "
        f"отказ {int((frame['fault'] == 1).sum())})",
        f"тест:     окон {len(test)} (не изменялся)",
        f"файл: {config.STEP5_PATH}",
    ]
    report = config.PROCESSED_DIR / "adapt_step5_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")

    ok = (not overlap and not files_overlap and not pair_overlap and test_untouched
          and bad_bounds == 0 and step_ok and label_mismatch == 0 and nan_features == 0)
    for line in lines:
        log.info("%s", line)
    if not ok:
        log.error("проверки датасета с частой сеткой не пройдены")
        return 1
    log.info("датасет с шагом %d готов: %s", STRIDE, config.STEP5_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
