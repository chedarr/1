# -*- coding: utf-8 -*-
"""Конфигурация проекта лабораторной №2 (ADAPT, бинарная классификация).

Единая точка правды: пути, seed, лимиты ресурсов, имена колонок,
параметры окна и чёрный список терминов для текста отчёта.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

# --------------------------------------------------------------------- пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "adapt_train.parquet"
TEST_PATH = PROCESSED_DIR / "adapt_test.parquet"
LOG_PATH = PROCESSED_DIR / "adapt_log.parquet"

SRC_DIR = BASE_DIR / "src"
SCRIPTS_DIR = BASE_DIR / "scripts"
TESTS_DIR = BASE_DIR / "tests"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = REPORTS_DIR / "model_adapt.keras"
METRICS_PATH = REPORTS_DIR / "metrics_adapt.json"
BASELINES_PATH = REPORTS_DIR / "summary_baselines.json"
HISTORY_PATH = REPORTS_DIR / "training_history.json"
SPLIT_PATH = REPORTS_DIR / "split_info.json"
LOEO_PATH = REPORTS_DIR / "loeo_adapt.json"

REPORT_HTML = REPORTS_DIR / "lab2_report.html"
LIKBEZ_MD = DOCS_DIR / "lab2_likbez.md"
LIKBEZ_HTML = REPORTS_DIR / "lab2_likbez.html"
STYLE_CSS = REPORTS_DIR / "style_likbez.css"

# ------------------------------------------------------------------ задача
TASK = "binary_classification"
TARGET = "fault"                      # 0 — норма, 1 — отказ
WINDOW_SIZE = 20
STRIDE = 10                           # шаг эталонных окон (обучение и тест)
STRIDE_SMALL = 5                      # шаг обучающих окон в варианте с частой сеткой
STEP5_PATH = PROCESSED_DIR / "adapt_train_step5.parquet"
N_SENSORS = 81
N_FEATURES = WINDOW_SIZE * N_SENSORS  # 1620
META_PREFIX = "meta__"                # служебные колонки: в модель не подавать
TEST_EXPERIMENTS = ["441", "442", "443", "445"]
N_TRAIN_EXPERIMENTS = 12
N_TEST_EXPERIMENTS = 4
N_EXPERIMENTS = N_TRAIN_EXPERIMENTS + N_TEST_EXPERIMENTS  # 16
CV_FOLDS = 5                          # фолды кросс-валидации по экспериментам

# термин «fault» используется только как имя колонки-цели;
# в тексте отчёта метка описывается словами «норма» / «отказ»
LABEL_NAMES = {0: "норма", 1: "отказ"}

# --------------------------------------------------------------- ресурсы
SEED = 42
# GPU на этой машине недоступна для TensorFlow, поэтому лимиты задаются
# по CPU/RAM. Значения подобраны под Ryzen 5 3600 (6 ядер / 12 потоков).
CPU_THREADS = 6
INTEROP_THREADS = 2
RAM_LIMIT_GB_PER_JOB = 3.0
MAX_PARALLEL_JOBS = 2
# заявленный в ТЗ лимит видеопамяти сохраняется как константа, но не
# применяется: обучающее устройство — CPU
VRAM_LIMIT_GB = 3.5
DEVICE = "CPU"

# ----------------------------------------------------------- воспроизводимость
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


def set_seed(seed: int = SEED) -> None:
    """Фиксирует все источники случайности: random, numpy, tensorflow."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def limit_resources() -> None:
    """Ограничивает потоки CPU: конкурирующие задания не должны съедать всё."""
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    try:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(INTEROP_THREADS)
        # GPU-лимиты применяются только если устройство реально существует
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ----------------------------------------------------------- логирование
LOGGER_NAME = "adapt"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO, log_path: Path | str | None = None):
    """Настраивает журнал уровня INFO: вывод в поток вывода и, если задан, в файл.

    Журнал ведётся отдельным именованным регистратором, поэтому настройка
    не влияет на библиотечные журналы и не дублирует сообщения.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Регистратор проекта; до вызова setup_logging пишет на уровне WARNING."""
    return logging.getLogger(name)


# --------------------------------------- чёрный список для текста отчёта
# Эти строки не должны встречаться в тексте отчёта и ликбеза.
BANNED_IN_TEXT = [
    META_PREFIX,
    "FaultType",
    "FaultMode",
    "FaultLocation",
]
