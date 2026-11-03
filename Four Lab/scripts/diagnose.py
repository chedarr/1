# -*- coding: utf-8 -*-
"""Диагностика: сравнение прогонов, худшие шаги горизонта, вклад каналов."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def main() -> int:
    exp_dir = config.REPORTS / "experiments"
    files = sorted(exp_dir.glob("*.json"))
    if not files:
        print("нет прогонов")
        return 1
    print(f"{'run':6} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'skill':>7} {'ep':>4}")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        test = data["test"]
        print(
            f"{data['key']:6} {test['mae']:8.5f} {test['rmse']:8.5f} "
            f"{test['r2']:8.5f} {test.get('skill_score', float('nan')):7.4f} "
            f"{data['epochs_ran']:4d}"
        )

    best_path = min(files, key=lambda p: json.loads(p.read_text(encoding="utf-8"))["test"]["mae"])
    best = json.loads(best_path.read_text(encoding="utf-8"))
    steps = best["test"]["mae_by_step"]
    worst = sorted(range(len(steps)), key=lambda i: steps[i], reverse=True)[:5]
    print(f"\nлучший: {best['key']} ({best['label']}), MAE={best['test']['mae']:.5f}")
    print("худшие шаги горизонта:", ", ".join(f"{i + 1}: {steps[i]:.4f}" for i in worst))
    print(f"MAE на 1-м шаге {steps[0]:.4f}, на последнем {steps[-1]:.4f}")
    history = best["history"]
    print(f"эпох: {best['epochs_ran']}, лучшая: {best['best_epoch']}, "
          f"val_mae={min(history.get('val_mae', [float('nan')])):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
