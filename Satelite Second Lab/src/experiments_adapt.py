# -*- coding: utf-8 -*-
"""Поиск улучшений классификатора ADAPT.

Механика экспериментов: пересборка окон из сырых измерений с любым шагом и
смещением, расширение признаков, аугментация, кросс-валидация по экспериментам,
подбор порога решения.

Железное правило: архитектура и порог выбираются ТОЛЬКО по валидации внутри
обучающей выборки. Тестовая выборка используется исключительно для итогового
отчёта и никогда — для выбора.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src import config
from src.metrics_adapt import compute_metrics
from src.split_adapt import group_kfold_indices, holdout_by_experiment

WINDOW = config.WINDOW_SIZE
FMT = "%Y-%m-%d %H:%M:%S.%f"
OTHER_COLUMNS = ("source_file", "line_no", "record_type", "Time")


# ------------------------------------------------------------------ описание
@dataclass(frozen=True)
class Config:
    """Одна конфигурация эксперимента."""

    name: str
    note: str = ""
    hidden: tuple = (64, 32)
    dropout: float = 0.2
    batchnorm: bool = False
    l2: float = 0.0
    class_weight: bool = False
    augment: bool = False
    blocks: tuple = ("raw",)
    stride: int = 10
    offsets: tuple = (0,)

    @property
    def extras(self) -> str:
        bits = []
        if self.batchnorm:
            bits.append("нормализация")
        if self.l2:
            bits.append(f"L2={self.l2:g}")
        if self.class_weight:
            bits.append("веса классов")
        if self.augment:
            bits.append("аугментация")
        if self.blocks != ("raw",):
            bits.append("признаки: " + "+".join(self.blocks))
        if self.stride != 10 or self.offsets != (0,):
            bits.append(f"шаг {self.stride}")
        return ", ".join(bits) if bits else "как в базовой модели"


# -------------------------------------------------------------- сырые данные
def load_measurements() -> tuple[pd.DataFrame, list[str]]:
    """Измерения с признаком отказа и моментом его начала (как при сборке датасета)."""
    raw = pd.read_parquet(config.PROCESSED_DIR / "adapt_raw.parquet")
    log = pd.read_parquet(config.LOG_PATH)
    sensors = [c for c in raw.columns if c not in OTHER_COLUMNS]

    meas = raw[raw["record_type"] == "AntagonistData"].copy()
    meas["exp"] = meas["source_file"].str.extract(r"Exp_(\d+)_", expand=False)
    meas["ts"] = pd.to_datetime(meas["Time"].str[:23], format=FMT)
    meas = meas.sort_values(["source_file", "line_no"]).reset_index(drop=True)
    t0 = meas.groupby("source_file")["ts"].transform("first")
    meas["elapsed"] = (meas["ts"] - t0).dt.total_seconds()
    onset = log.groupby("Experiment")["Time (s)"].min()
    meas["onset"] = meas["exp"].map(onset).astype(float)
    if meas["onset"].isna().any():
        raise ValueError("есть измерения без записи в журнале отказов")
    meas["fault"] = (meas["elapsed"] >= meas["onset"]).astype("int8")
    return meas, sensors


def build_windows(
    meas: pd.DataFrame, sensors: list[str], stride: int, offsets: tuple, test: bool
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Окна длиной WINDOW для выбранной половины; окно не пересекает файл."""
    blocks: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict] = []

    for fname, sub in meas.groupby("source_file", sort=True):
        exp = str(sub["exp"].iloc[0])
        if (exp in config.TEST_EXPERIMENTS) != test:
            continue
        sub = sub.sort_values("line_no").reset_index(drop=True)
        values = sub[sensors].astype(float).to_numpy()
        lab = sub["fault"].to_numpy(dtype=int)
        n = len(sub)
        starts: set[int] = set()
        for off in offsets:
            starts.update(range(off, n - WINDOW + 1, stride))
        for s in sorted(starts):
            if s < 0 or s + WINDOW > n:
                continue
            blocks.append(values[s : s + WINDOW])
            labels.append(int(lab[s + WINDOW - 1]))
            meta.append(
                {
                    "meta__experiment": exp,
                    "meta__source_file": fname,
                    "meta__start_index": s,
                    "meta__end_index": s + WINDOW - 1,
                }
            )
    if not blocks:
        raise ValueError("не построено ни одного окна")
    return np.stack(blocks).astype(np.float32), np.array(labels, dtype=np.int32), pd.DataFrame(meta)


def verify_against_reference(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> dict:
    """Сверяет пересобранные окна с уже принятым датасетом обучения."""
    ref = pd.read_parquet(config.TRAIN_PATH)
    feats = [c for c in ref.columns if not c.startswith(config.META_PREFIX) and c != config.TARGET]
    ref_x = ref[feats].to_numpy(dtype=np.float32).reshape(len(ref), WINDOW, -1)
    ref_y = ref[config.TARGET].to_numpy(dtype=np.int32)
    same_shape = ref_x.shape == X.shape
    same_labels = bool(np.array_equal(ref_y, y)) if same_shape else False
    max_diff = float(np.abs(ref_x - X).max()) if same_shape else float("nan")
    same_starts = bool(
        same_shape
        and np.array_equal(
            ref["meta__start_index"].to_numpy(),
            meta["meta__start_index"].to_numpy(),
        )
    )
    return {
        "совпадает_размер": same_shape,
        "совпадают_метки": same_labels,
        "максимальное_расхождение": max_diff,
        "совпадают_начала_окон": same_starts,
    }


# -------------------------------------------------------------------- признаки
def moving_average(x3: np.ndarray, width: int) -> np.ndarray:
    """Скользящее среднее по времени; края дополняются повтором крайнего шага."""
    pad = width // 2
    padded = np.pad(x3, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    n, t, s = x3.shape
    flat = padded.transpose(0, 2, 1).reshape(n * s, -1)
    smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, flat)
    return smooth.reshape(n, s, t).transpose(0, 2, 1)


def build_features(x3: np.ndarray, blocks: tuple = ("raw",)) -> np.ndarray:
    """Собирает матрицу признаков из окна (число окон, шагов, сенсоров)."""
    parts: list[np.ndarray] = []
    if "raw" in blocks:
        parts.append(x3.reshape(len(x3), -1))
    if "stats" in blocks:
        parts.extend([x3.mean(axis=1), x3.std(axis=1), x3.min(axis=1), x3.max(axis=1)])
    if "delta" in blocks:
        parts.append(np.diff(x3, axis=1).reshape(len(x3), -1))
    if "smooth3" in blocks:
        parts.append(moving_average(x3, 3).reshape(len(x3), -1))
    if "smooth5" in blocks:
        parts.append(moving_average(x3, 5).reshape(len(x3), -1))
    if not parts:
        raise ValueError("не выбран ни один блок признаков")
    return np.concatenate(parts, axis=1).astype(np.float32)


def augment(
    x3: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    sensor_std: np.ndarray,
    copies: int = 1,
    noise: float = 0.03,
    scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Добавляет шум и лёгкое масштабирование; смещение окна задаётся смещениями."""
    xs = [x3]
    ys = [y]
    for _ in range(copies):
        noisy = x3 + rng.normal(0.0, noise, x3.shape) * sensor_std
        factor = 1.0 + rng.uniform(-scale, scale, size=(len(x3), 1, 1))
        xs.append((noisy * factor).astype(np.float32))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


# --------------------------------------------------------------------- модель
def architecture_text(cfg: Config) -> str:
    """Словесное описание сети для отчёта."""
    chain = " → ".join([str(cfg.hidden[0])] + [str(h) for h in cfg.hidden[1:]])
    bits = [f"прореживание {cfg.dropout:g}"]
    if cfg.batchnorm:
        bits.append("нормализация слоёв")
    if cfg.l2:
        bits.append(f"L2 {cfg.l2:g}")
    if cfg.class_weight:
        bits.append("веса классов")
    return f"вход → {chain} → 1 ({', '.join(bits)})"


def build_model(cfg: Config, n_features: int, seed: int = config.SEED) -> keras.Model:
    """Полносвязная сеть по описанию конфигурации."""
    keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(cfg.l2) if cfg.l2 else None
    layers: list = [keras.layers.Input(shape=(n_features,), name="sensors_in")]
    for i, width in enumerate(cfg.hidden, start=1):
        layers.append(keras.layers.Dense(width, kernel_regularizer=reg, name=f"dense_{i}"))
        if cfg.batchnorm:
            layers.append(keras.layers.BatchNormalization(name=f"norm_{i}"))
        layers.append(keras.layers.Activation("relu", name=f"relu_{i}"))
        if cfg.dropout > 0:
            layers.append(keras.layers.Dropout(cfg.dropout, name=f"drop_{i}"))
    layers.append(keras.layers.Dense(1, activation="sigmoid", name="output"))
    model = keras.Sequential(layers, name="mlp_adapt")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def class_weights(y: np.ndarray, enabled: bool) -> dict | None:
    """Сбалансированные веса классов."""
    if not enabled:
        return None
    n = len(y)
    n1 = int((y == 1).sum())
    n0 = n - n1
    return {0: n / (2.0 * n0), 1: n / (2.0 * n1)}


def _fit(
    cfg: Config,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_es: np.ndarray,
    y_es: np.ndarray,
    epochs: int,
    seed: int,
):
    model = build_model(cfg, x_fit.shape[1], seed)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=0
        ),
    ]
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_es, y_es),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
        class_weight=class_weights(y_fit, cfg.class_weight),
    )
    return model, history


def fit_with_early_stopping(
    cfg: Config,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_es: np.ndarray,
    y_es: np.ndarray,
    epochs: int = 120,
    seed: int = config.SEED,
):
    """Обучение с ранней остановкой по отдельному контрольному эксперименту.

    Открытая обёртка над внутренней процедурой обучения: используется и
    кросс-валидацией, и оценкой «по одному эксперименту».
    """
    return _fit(cfg, x_fit, y_fit, x_es, y_es, epochs, seed)


# ------------------------------------------------------------------ порог
def f_beta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def threshold_curve(y: np.ndarray, prob: np.ndarray) -> list[dict]:
    """Точки кривой «точность–полнота» по всем наблюдаемым порогам."""
    rows = []
    for t in np.unique(np.round(prob, 6)):
        m = compute_metrics(y, prob, threshold=float(t))
        rows.append(
            {
                "порог": float(t),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "f2": f_beta(m["precision"], m["recall"], 2.0),
                "accuracy": m["accuracy"],
            }
        )
    return sorted(rows, key=lambda r: r["порог"])


PRECISION_FLOOR = 0.70


def best_thresholds(curve: list[dict], precision_floor: float = PRECISION_FLOOR) -> dict:
    """Пороги по кривой «точность–полнота».

    Чистая максимизация F-меры может выродиться в «объявлять отказом почти всё»:
    при доле отказов около половины такой ответ даёт неплохую F-меру, но не решает
    задачу. Поэтому порог ищется при требовании точности не ниже заданного уровня;
    вариант без ограничения возвращается отдельно для прозрачности.
    """
    allowed = [r for r in curve if r["precision"] >= precision_floor]
    limited = bool(allowed)
    rows = allowed or curve
    by_f1 = max(rows, key=lambda r: (r["f1"], -r["порог"]))
    by_f2 = max(rows, key=lambda r: (r["f2"], -r["порог"]))
    free = max(curve, key=lambda r: (r["f1"], -r["порог"]))
    return {
        "f1": by_f1,
        "f2": by_f2,
        "без_ограничения": free,
        "требование_точности": precision_floor,
        "ограничение_применено": limited,
    }


# ------------------------------------------------------------- кросс-валидация
EXP_COL = config.META_PREFIX + "experiment"


def windows_for_config(cfg: Config, meas, sensors, x_ref, y_ref, meta_ref):
    """Обучающие окна конфигурации; базовая конфигурация использует готовые окна."""
    if cfg.stride == 10 and cfg.offsets == (0,):
        return x_ref, y_ref, meta_ref
    return build_windows(meas, sensors, cfg.stride, cfg.offsets, test=False)


def run_cv(
    cfg: Config,
    meas,
    sensors,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    meta_ref: pd.DataFrame,
    n_splits: int = 5,
    epochs: int = 120,
) -> tuple[np.ndarray, list[dict], int]:
    """Кросс-валидация по экспериментам.

    Обучающие окна берутся в конфигурации (возможно, с другим шагом), а
    проверка всегда идёт на одних и тех же эталонных окнах. Только так
    конфигурации сравнимы между собой.
    """
    x_fit_all, y_fit_all, meta_fit_all = windows_for_config(
        cfg, meas, sensors, x_ref, y_ref, meta_ref
    )
    sensor_std = x_fit_all.reshape(-1, x_fit_all.shape[-1]).std(axis=0)
    oof = np.full(len(y_ref), np.nan, dtype=float)
    folds: list[dict] = []
    used_epochs: list[int] = []

    for i, (tr, va) in enumerate(group_kfold_indices(meta_ref, n_splits), start=1):
        train_exps = set(meta_ref.iloc[tr][EXP_COL])
        inner = meta_ref.iloc[tr].reset_index(drop=True)
        _, idx_es, _ = holdout_by_experiment(inner, 1, seed=config.SEED + i)
        es_exps = set(inner.iloc[idx_es][EXP_COL])

        fit_mask = meta_fit_all[EXP_COL].isin(train_exps - es_exps).to_numpy()
        es_mask = meta_fit_all[EXP_COL].isin(es_exps).to_numpy()

        x_fit, y_fit = x_fit_all[fit_mask], y_fit_all[fit_mask]
        if cfg.augment:
            x_fit, y_fit = augment(
                x_fit, y_fit, np.random.default_rng(config.SEED + i), sensor_std
            )
        scaler = StandardScaler().fit(build_features(x_fit, cfg.blocks))
        model, history = _fit(
            cfg,
            scaler.transform(build_features(x_fit, cfg.blocks)),
            y_fit,
            scaler.transform(build_features(x_fit_all[es_mask], cfg.blocks)),
            y_fit_all[es_mask],
            epochs,
            seed=config.SEED + i,
        )
        oof[va] = model.predict(
            scaler.transform(build_features(x_ref[va], cfg.blocks)), verbose=0
        ).ravel()
        used_epochs.append(len(history.history["loss"]))
        m = compute_metrics(y_ref[va], oof[va])
        m["фолд"] = i
        m["эксперименты"] = sorted(set(meta_ref.iloc[va][EXP_COL]))
        folds.append(m)

    return oof, folds, int(np.median(used_epochs)), history


def run_final(
    cfg: Config,
    meas,
    sensors,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    meta_ref: pd.DataFrame,
    x_test: np.ndarray,
    epochs: int,
) -> tuple[keras.Model, np.ndarray]:
    """Итоговая модель: обучение на всей обучающей выборке, тест не меняется."""
    x_fit, y_fit, _ = windows_for_config(cfg, meas, sensors, x_ref, y_ref, meta_ref)
    sensor_std = x_fit.reshape(-1, x_fit.shape[-1]).std(axis=0)
    if cfg.augment:
        x_fit, y_fit = augment(
            x_fit, y_fit, np.random.default_rng(config.SEED), sensor_std
        )
    scaler = StandardScaler().fit(build_features(x_fit, cfg.blocks))
    x_train = scaler.transform(build_features(x_fit, cfg.blocks))
    model = build_model(cfg, x_train.shape[1], config.SEED)
    model.fit(
        x_train,
        y_fit,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        class_weight=class_weights(y_fit, cfg.class_weight),
    )
    prob = model.predict(
        scaler.transform(build_features(x_test, cfg.blocks)), verbose=0
    ).ravel()
    return model, prob


# --------------------------------------------------------- список конфигураций
def default_configs() -> list[Config]:
    """Набор экспериментов: от базовой модели к комбинации улучшений."""
    tuned = dict(dropout=0.3, batchnorm=True, l2=1e-4)
    return [        Config("A_базовая", "исходная модель отчёта"),
        Config("B_веса_классов", "повышенный вес редкого класса", class_weight=True),
        Config("C_нормализация", "нормализация слоёв, прореживание 0.3, L2", **tuned),
        Config(
            "D_ёмкость",
            "слои 128–64–32",
            hidden=(128, 64, 32),
            **tuned,
        ),
        Config(
            "E_ёмкость_веса",
            "слои 128–64–32 и веса классов",
            hidden=(128, 64, 32),
            class_weight=True,
            **tuned,
        ),
        Config(
            "F_аугментация",
            "шум и масштабирование обучающих окон",
            hidden=(128, 64, 32),
            class_weight=True,
            augment=True,
            **tuned,
        ),
        Config(
            "G_признаки",
            "средние, разброс, экстремумы и разности",
            hidden=(128, 64, 32),
            class_weight=True,
            blocks=("raw", "stats", "delta"),
            **tuned,
        ),
        Config(
            "H_шаг_5",
            "вдвое больше обучающих окон",
            hidden=(128, 64, 32),
            class_weight=True,
            stride=5,
            **tuned,
        ),
        Config(
            "I_смещения",
            "окна со смещением ±2 шага",
            hidden=(128, 64, 32),
            class_weight=True,
            offsets=(-2, -1, 0, 1, 2),
            **tuned,
        ),
        Config(
            "J_комбинация",
            "ёмкость, веса, признаки, аугментация и шаг 5",
            hidden=(128, 64, 32),
            class_weight=True,
            augment=True,
            blocks=("raw", "stats", "delta"),
            stride=5,
            **tuned,
        ),
        Config(
            "K_шаг_5_базовая",
            "базовая сеть на вдвое более частой сетке окон",
            stride=config.STRIDE_SMALL,
        ),
    ]


def find_config(name: str, configs: list[Config] | None = None) -> Config | None:
    """Ищет конфигурацию по имени в списке по умолчанию."""
    for cfg in configs or default_configs():
        if cfg.name == name:
            return cfg
    return None


def _evaluate(cfg: Config, y, prob, curve_pick: dict) -> dict:
    """Метрики на заданном пороге."""
    return compute_metrics(y, prob, threshold=curve_pick["порог"])


def run_search(
    meas: pd.DataFrame,
    sensors: list[str],
    configs: list[Config] | None = None,
    n_splits: int = 5,
    epochs: int = 120,
    progress=print,
) -> dict:
    """Прогоняет все конфигурации и возвращает сводку.

    Выбор победителя делается по валидации (средние по фолдам и общий
    валидационный набор), тест в выборе не участвует.
    """
    configs = configs or default_configs()
    x_ref, y_ref, meta_ref = build_windows(meas, sensors, 10, (0,), test=False)
    x_test, y_test, _ = build_windows(meas, sensors, 10, (0,), test=True)
    verification = verify_against_reference(x_ref, y_ref, meta_ref)

    results: list[dict] = []
    for cfg in configs:
        progress(f"  {cfg.name}: {cfg.extras}")
        try:
            oof, folds, epochs_used, last_history = run_cv(
                cfg, meas, sensors, x_ref, y_ref, meta_ref, n_splits, epochs
            )
            curve = threshold_curve(y_ref, oof)
            picks = best_thresholds(curve)
            by_f1, by_f2 = picks["f1"], picks["f2"]
            model, prob_test = run_final(
                cfg, meas, sensors, x_ref, y_ref, meta_ref, x_test, epochs_used
            )
            f1_cv = compute_metrics(y_ref, oof, by_f1["порог"])
            folds_f1 = float(np.mean([f["f1"] for f in folds]))
            _, y_cfg, _ = windows_for_config(cfg, meas, sensors, x_ref, y_ref, meta_ref)
            results.append(
                {
                    "имя": cfg.name,
                    "описание": cfg.note,
                    "дополнения": cfg.extras,
                    "архитектура": architecture_text(cfg),
                    "окон_обучения": int(len(y_cfg)),
                    "признаков": int(build_features(x_ref[:1], cfg.blocks).shape[1]),
                    "эпох": epochs_used,
                    "порог_f1": by_f1["порог"],
                    "порог_f2": by_f2["порог"],
                    "порог_без_ограничения": picks["без_ограничения"]["порог"],
                    "f1_без_ограничения": picks["без_ограничения"]["f1"],
                    "требование_точности": picks["требование_точности"],
                    "валидация_порог_0.5": compute_metrics(y_ref, oof, 0.5),
                    "валидация_порог_f1": f1_cv,
                    "валидация_f1_по_фолдам": folds_f1,
                    "валидация_отклонение_f1": float(np.std([f["f1"] for f in folds])),
                    "тест_порог_0.5": compute_metrics(y_test, prob_test, 0.5),
                    "тест_порог_f1": _evaluate(cfg, y_test, prob_test, by_f1),
                    "тест_порог_f2": _evaluate(cfg, y_test, prob_test, by_f2),
                    "фолды": folds,
                    "_oof": oof,
                    "_test": prob_test,
                    "_model": model,
                    "_history": last_history,
                }
            )
        except Exception as exc:  # одна неудачная конфигурация не срывает поиск
            results.append(
                {"имя": cfg.name, "описание": cfg.note, "дополнения": cfg.extras,
                 "ошибка": f"{type(exc).__name__}: {exc}"}
            )

    lr = logreg_probe(x_ref, y_ref, meta_ref, x_test, n_splits)
    lr_curve = threshold_curve(y_ref, lr["oof"])
    lr_picks = best_thresholds(lr_curve)
    lr_f1, lr_f2 = lr_picks["f1"], lr_picks["f2"]
    logreg_result = {
        "имя": "логистическая регрессия",
        "описание": "линейная модель на тех же окнах",
        "порог_f1": lr_f1["порог"],
        "порог_f2": lr_f2["порог"],
        "порог_без_ограничения": lr_picks["без_ограничения"]["порог"],
        "валидация_порог_0.5": compute_metrics(y_ref, lr["oof"], 0.5),
        "валидация_порог_f1": compute_metrics(y_ref, lr["oof"], lr_f1["порог"]),
        "тест_порог_0.5": compute_metrics(y_test, lr["test"], 0.5),
        "тест_порог_f1": compute_metrics(y_test, lr["test"], lr_f1["порог"]),
        "тест_порог_f2": compute_metrics(y_test, lr["test"], lr_f2["порог"]),
        "_oof": lr["oof"],
        "_test": lr["test"],
    }

    ok = [r for r in results if "ошибка" not in r]
    winner = max(ok, key=lambda r: r["валидация_порог_f1"]["f1"]) if ok else None

    return {
        "проверка_окон": verification,
        "конфигурации": results,
        "логистическая_регрессия": logreg_result,
        "победитель": winner["имя"] if winner else None,
        "правило_выбора": "наибольшая F-мера на валидации при пороге, подобранном "
                          "по валидации; тест в выборе не участвует",
        "эталонные_окна": {"обучение": int(len(y_ref)), "тест": int(len(y_test))},
    }


# ------------------------------------------------------- простые модели с порогом
def original_protocol_probe(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    meta_ref: pd.DataFrame,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Воспроизводит прежний порядок обучения из первого отчёта.

    Отличие от нового порядка: контроль из двух экспериментов вместо одного,
    обучение итоговой модели своим числом эпох, порог 0.5. Нужно, чтобы
    отделить вклад пересмотра порядка обучения от вклада архитектуры.
    """
    from src.model_adapt import build_mlp
    from src.model_adapt import callbacks as mlp_callbacks

    x_flat = x_ref.reshape(len(x_ref), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    idx_fit, idx_val, val_experiments = holdout_by_experiment(
        meta_ref, 2, seed=config.SEED
    )

    scaler = StandardScaler().fit(x_flat[idx_fit])
    model = build_mlp(n_features=x_flat.shape[1], seed=config.SEED)
    history = model.fit(
        scaler.transform(x_flat[idx_fit]).astype(np.float32),
        y_ref[idx_fit],
        validation_data=(
            scaler.transform(x_flat[idx_val]).astype(np.float32),
            y_ref[idx_val],
        ),
        epochs=150,
        batch_size=32,
        callbacks=mlp_callbacks(),
        verbose=0,
    )
    epochs_used = len(history.history["loss"])
    prob_val = model.predict(
        scaler.transform(x_flat[idx_val]).astype(np.float32), verbose=0
    ).ravel()

    scaler_full = StandardScaler().fit(x_flat)
    final = build_mlp(n_features=x_flat.shape[1], seed=config.SEED)
    final.fit(
        scaler_full.transform(x_flat).astype(np.float32),
        y_ref,
        epochs=epochs_used,
        batch_size=32,
        verbose=0,
    )
    prob_test = final.predict(
        scaler_full.transform(x_test_flat).astype(np.float32), verbose=0
    ).ravel()

    picks = best_thresholds(threshold_curve(y_ref[idx_val], prob_val))
    return {
        "имя": "Порядок обучения из первого отчёта",
        "описание": "контроль из двух экспериментов, порог 0.5",
        "контрольные_эксперименты": val_experiments,
        "эпох": epochs_used,
        "порог_f1": picks["f1"]["порог"],
        "валидация_порог_0.5": compute_metrics(y_ref[idx_val], prob_val, 0.5),
        "валидация_порог_f1": compute_metrics(y_ref[idx_val], prob_val, picks["f1"]["порог"]),
        "тест_порог_0.5": compute_metrics(y_test, prob_test, 0.5),
        "тест_порог_f1": compute_metrics(y_test, prob_test, picks["f1"]["порог"]),
        "_test": prob_test,
        "_model": final,
    }


def logreg_probe(
    x3: np.ndarray, y: np.ndarray, meta: pd.DataFrame, x3_test: np.ndarray, n_splits: int = 5
) -> dict:
    """Логистическая регрессия: валидационные и тестовые вероятности."""
    x = x3.reshape(len(x3), -1)
    xt = x3_test.reshape(len(x3_test), -1)
    oof = np.full(len(y), np.nan, dtype=float)
    for i, (tr, va) in enumerate(group_kfold_indices(meta, n_splits), start=1):
        scaler = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, random_state=config.SEED + i)
        clf.fit(scaler.transform(x[tr]), y[tr])
        oof[va] = clf.predict_proba(scaler.transform(x[va]))[:, 1]
    scaler = StandardScaler().fit(x)
    final = LogisticRegression(max_iter=2000, random_state=config.SEED)
    final.fit(scaler.transform(x), y)
    return {"oof": oof, "test": final.predict_proba(scaler.transform(xt))[:, 1]}
