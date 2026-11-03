# -*- coding: utf-8 -*-
"""Готовит окна, scaler, split_info.json и data_summary.json."""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import build_data_summary
from src.pipeline import build_dataset, save_artifacts


def main() -> int:
    t0 = time.time()
    summary = build_data_summary()
    data = build_dataset()
    info = save_artifacts(data)
    print(
        f"строк: {summary['n_rows_raw']} -> {summary['n_rows_clean']} "
        f"(NaN {summary['nan_fraction_raw'] * 100:.3f}%, серий >{summary['max_gap_steps']}: "
        f"{summary['nan_runs_long']}, макс {summary['longest_run_steps']} шагов "
        f"= {summary['longest_run_days']} сут)"
    )
    print(f"окна: train={info['windows']['train']} val={info['windows']['val']} "
          f"test={info['windows']['test']}, пропущено по границам {info['windows_dropped_by_boundaries']}, "
          f"по разрывам {info['skipped_windows']}")
    print(f"корреляции: {summary['channel_correlation']['PHT10180_mean']}")
    print(f"автокорреляция: {summary['target_autocorrelation']}")
    print(f"ИТОГ: данные готовы за {time.time() - t0:.1f} c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
