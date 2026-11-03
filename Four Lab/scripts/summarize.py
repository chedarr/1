# -*- coding: utf-8 -*-
"""Собирает metrics_experiments.json и ablation_results.json из reports/experiments/*.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

GROUPS = {
    "univariate_vs_multivariate": ["A", "B"],
    "lookback": ["A", "lb72", "lb288"],
    "horizon": ["A", "hz72", "hz28"],
    "architecture": ["A", "C", "E"],
    "dropout": ["A", "D", "E0"],
}
ORDER = ["A", "B", "C", "D", "E", "F", "G", "E0", "lb72", "lb288", "hz72", "hz28"]


def main() -> int:
    folder = config.REPORTS / "experiments"
    runs = {}
    for path in folder.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[payload["key"]] = payload

    ordered = [runs[k] for k in ORDER if k in runs]
    experiments = {
        "runs": [
            {
                "key": run["key"],
                "label": run["label"],
                "config": run["config"],
                "params": run["params"],
                "epochs_ran": run["epochs_ran"],
                "best_epoch": run["best_epoch"],
                "training_time_sec": run["training_time_sec"],
                "val": {"mae": run["val"]["mae"], "rmse": run["val"]["rmse"], "r2": run["val"]["r2"]},
                "test": {
                    "mae": run["test"]["mae"],
                    "rmse": run["test"]["rmse"],
                    "r2": run["test"]["r2"],
                    "skill_score": run["test"].get("skill_score"),
                },
            }
            for run in ordered
        ]
    }
    (config.REPORTS / "metrics_experiments.json").write_text(
        json.dumps(experiments, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    ablations = {}
    for name, keys in GROUPS.items():
        rows = []
        base = runs.get(keys[0])
        for key in keys:
            if key not in runs:
                continue
            run = runs[key]
            rows.append({
                "key": key,
                "label": run["label"],
                "mae": run["test"]["mae"],
                "rmse": run["test"]["rmse"],
                "r2": run["test"]["r2"],
                "delta_mae_vs_first": round(run["test"]["mae"] - base["test"]["mae"], 5) if base else None,
            })
        ablations[name] = rows
    (config.REPORTS / "ablation_results.json").write_text(
        json.dumps(ablations, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"экспериментов: {len(ordered)}; абляций: {len(ablations)}")
    for name, rows in ablations.items():
        print(f"  {name}: " + ", ".join(f"{r['key']} {r['mae']:.5f}" for r in rows))
    print("ИТОГ: сводки собраны")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
