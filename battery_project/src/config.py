"""Конфигурация проекта: пути, константы, списки батарей, признаки задач."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- пути
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR = BASE_DIR / "reports"

RAW_CYCLES_PATH = PROCESSED_DIR / "raw_cycles.parquet"
CLEAN_CYCLES_PATH = PROCESSED_DIR / "clean_cycles.parquet"
FEATURES_PATH = PROCESSED_DIR / "battery_features.parquet"
RUL_DATASET_PATH = PROCESSED_DIR / "battery_rul.parquet"
VOLTAGE_DATASET_PATH = PROCESSED_DIR / "battery_voltage.parquet"
CAPACITY_DATASET_PATH = PROCESSED_DIR / "battery_capacity.parquet"

METRICS_RUL_PATH = PROCESSED_DIR / "metrics_rul.json"
METRICS_VOLTAGE_PATH = PROCESSED_DIR / "metrics_voltage.json"
METRICS_CAPACITY_PATH = PROCESSED_DIR / "metrics_capacity.json"
SUMMARY_METRICS_PATH = PROCESSED_DIR / "summary_metrics.json"

MODEL_RUL_PATH = PROCESSED_DIR / "model_rul.keras"
MODEL_VOLTAGE_PATH = PROCESSED_DIR / "model_voltage.keras"
MODEL_CAPACITY_PATH = PROCESSED_DIR / "model_capacity.keras"

# совместимость со вспомогательным src/split.py
SPLIT_PATH = ARTIFACTS_DIR / "split.json"

EXPORT_FEATURES_CSV = PROCESSED_DIR / "battery_features.csv"
EXPORT_FEATURES_XLSX = PROCESSED_DIR / "battery_features.xlsx"
EXPORT_FEATURES_JSON = PROCESSED_DIR / "battery_features.json"
EXPORT_FEATURES_REPORT = PROCESSED_DIR / "battery_features_report.md"

LAB1_REPORT_PATH = REPORTS_DIR / "lab1_report.md"
LAB1_REPORT_HTML_PATH = REPORTS_DIR / "lab1_report.html"
STYLE_CSS_PATH = REPORTS_DIR / "style.css"

# ---------------------------------------------------------------- батареи
ALL_BATTERIES = [
    "B0005", "B0006", "B0007", "B0018",
    "B0025", "B0026", "B0027", "B0028",
    "B0029", "B0030", "B0031", "B0032",
    "B0033", "B0034", "B0036",
    "B0038", "B0039", "B0040", "B0041",
    "B0042", "B0043", "B0044", "B0045", "B0046", "B0047", "B0048",
    "B0049", "B0050", "B0051", "B0052", "B0053", "B0054", "B0055", "B0056",
]

# 8 батарей с полной кривой деградации — только для задачи RUL
RUL_BATTERIES = ["B0005", "B0006", "B0018", "B0045", "B0046", "B0047", "B0048", "B0053"]

# аномальные батареи с резкими/мусорными падениями ёмкости (для инфографики)
ANOMALOUS_BATTERIES = [
    "B0033", "B0034", "B0036", "B0039", "B0041",
    "B0050", "B0051", "B0052", "B0054", "B0055", "B0056",
]

DEBUG_BATTERIES = ["B0005", "B0042", "B0043", "B0045"]

# ---------------------------------------------------------------- константы чистки
EOL_RATIO = 0.7
N_INIT = 5
SKIP_FORMATION = 3
LOW_CAP_THRESHOLD = 0.5
CAPACITY_MAX = 2.6
IMPEDANCE_MAD_K = 5.0

# Метаданные по батарее (fallback, если ambient_temperature в .mat = NaN)
METADATA_AMBIENT_C: dict[str, float] = {}
for _b in ALL_BATTERIES:
    if _b in {"B0029", "B0030", "B0031", "B0032"}:
        METADATA_AMBIENT_C[_b] = 43.0
    elif _b >= "B0041":
        METADATA_AMBIENT_C[_b] = 4.0
    else:
        METADATA_AMBIENT_C[_b] = 24.0

# ---------------------------------------------------------------- модель
RANDOM_STATE = 42
MLP_EPOCHS = 200
MLP_BATCH_SIZE = 32
MLP_DENSE_UNITS = (64, 32)
MLP_DROPOUT = 0.2
EARLY_STOP_PATIENCE = 10
TEST_SIZE = 0.25

# ---------------------------------------------------------------- признаки задач
# A: RUL (8 батарей)
RUL_FEATURES = [
    "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
    "current_mean",
    "temp_mean", "temp_min", "temp_max", "temp_range",
    "time_total",
    "capacity", "capacity_ratio", "voltage_range", "discharge_rate",
]
# Цель обучения задачи A — нормированная доля остаточного ресурса:
#   RUL_frac = RUL_cycles / total_cycles,  total_cycles = max(discharge_num)
# в батарее. RUL_cycles остаётся для интерпретации.
RUL_TARGET = "rul_frac"
RUL_TARGET_CYCLES = "rul_cycles"

# B: напряжение следующего цикла (все 34)
VOLTAGE_FEATURES = [
    "voltage_mean", "voltage_min", "voltage_max", "voltage_std",
    "current_mean", "temp_mean", "temp_max",
    "capacity", "capacity_ratio",
    "voltage_mean_lag1", "voltage_mean_lag2", "capacity_lag1",
    "voltage_diff",
]
VOLTAGE_TARGET = "voltage_mean_next"

# C: ёмкость следующего цикла (все 34)
CAPACITY_FEATURES = [
    "capacity", "capacity_ratio", "voltage_mean", "current_mean", "temp_mean",
    "capacity_lag1", "capacity_lag2", "capacity_lag3",
    "capacity_diff", "capacity_slope",
]
CAPACITY_TARGET = "capacity_next"

ID_COLS = ["battery_id", "cycle_index", "date"]