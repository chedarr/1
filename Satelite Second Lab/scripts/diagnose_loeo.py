# -*- coding: utf-8 -*-
r"""Диагностика разброса F-меры по фолдам «обучение по одному эксперименту».

Задача: проверить числами, объясняется ли разброс качества между записями
доступными метаданными окна — долей отказов, положением первого отказа,
разбросом сенсоров и удалённостью записи от обучающей части.

Что считается (вход — только уже готовые артефакты):
* метаданные окон берутся из data\processed\adapt_train.parquet и adapt_test.parquet;
* качество фолда (F-мера, порог, контроль порога и т.п.) берётся КАК ЕСТЬ из
  reports\loeo_adapt.json.

Жёсткие правила этого скрипта:
* сеть не читается, обучение не запускается, tensorflow не импортируется;
* тестовые записи (441, 442, 443, 445) НИ НА ЧТО не влияют: они только
  описываются готовыми числами, никакой подгонки по ним нет;
* z-нормализация обучается ТОЛЬКО на обучающих экспериментах, тест в расчёт
  среднего и разброса не входит;
* ни один parquet-файл и ни один существующий артефакт не изменяется.

Определение расстояния (каноническое, одно на всю работу):
* в расстояние входят ТОЛЬКО признаки, которые варьируются на обучении
  (разброс по обучению строго больше нуля). Признаки, постоянные на обучении,
  в расстояние НЕ входят: у них разброс равен нулю, и деление на него давало бы
  неограниченный вклад, несопоставимый между записями;
* расстояние = корень из среднего квадрата z-отклонения по этим информативным
  признакам (сначала средний z-вектор записи, затем среднеквадратичное по
  признакам). Метрика сопоставима между записями и всегда конечна;
* выход телеметрии за пределы, наблюдавшиеся на обучении, по постоянным
  признакам измеряется отдельно (`сдвиг_в_постоянных_признаках`) и в расстояние
  не подмешивается.

Выход:
* reports\loeo_diagnostics.json
* reports\figures\09_diag_f1_vs_frac_positive.png
* reports\figures\10_diag_f1_vs_dist_to_train.png
* reports\figures\11_diag_f1_vs_first_fault.png

Запуск:  .venv\Scripts\python.exe scripts\diagnose_loeo.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # без окна: сохранение прямо в файл

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import config  # noqa: E402

EXP = config.META_PREFIX + "experiment"
START = config.META_PREFIX + "start_index"
TAU = 0.5  # порог для правила вывода: |r| < данного значения => связи не видно

# Характеристики окон, по которым проверяется связь с F-мерой.
CHARS = [
    "frac_positive",
    "first_fault_pos_frac",
    "mean_sensor_std",
    "dist_to_train",
    "dist_to_train_looe",
]

TARGETS = ["f1", "f1_в_пороге_0.5"]

# Поля, которые переносятся из loeo-артефакта без изменений.
LOEO_FIELDS = ["f1", "f1_в_пороге_0.5", "recall", "precision", "порог",
               "roc_auc", "эпох", "контроль_порога"]

# Диагностический вариант расстояния без ограничения z. Оставлен РОВНО ОДИН,
# под явно другим именем, и помечен непригодным для сравнения и корреляций:
# признаки с почти нулевым (но не нулевым) разбросом на обучении дают вклад в
# сотни сигм и перекрывают все остальные.
UNBOUNDED_DIST_FIELD = "dist_to_train_без_ограничения"
Z_CLIP = 10.0  # предел модуля z-отклонения на признак (сигмы)


# ------------------------------------------------------------------ загрузка
def load_loeo(path: Path | str | None = None) -> dict:
    """Готовый артефакт LOEO: числа модели берутся отсюда, а не считаются заново."""
    path = Path(path or config.LOEO_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def fold_table(report: dict) -> dict[str, dict]:
    """Словарь «эксперимент -> фолд» из отчёта."""
    return {str(f["эксперимент"]): f for f in report["фолды"]}


def split_info(path: Path | str | None = None) -> dict:
    """Разбиение на обучение и тест (для отметки источника записи)."""
    path = Path(path or config.SPLIT_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Обучающая и тестовая таблицы окон и список признаковых колонок."""
    train = pd.read_parquet(config.TRAIN_PATH)
    test = pd.read_parquet(config.TEST_PATH)
    return train, test, feature_columns(train)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Признаковые колонки: всё, кроме служебных и метки."""
    return [c for c in frame.columns
            if not c.startswith(config.META_PREFIX) and c != config.TARGET]


def combined_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Обучающие и тестовые окна в одной таблице с отметкой источника."""
    train = train.assign(источник="обучение")
    test = test.assign(источник="тест")
    return pd.concat([train, test], ignore_index=True)


def source_of(experiment: str, train_experiments: set[str]) -> str:
    """«обучение» или «тест» — по разбиению проекта."""
    return "обучение" if experiment in train_experiments else "тест"


# --------------------------------------------- статистики обучающей части
def train_stats(frame: pd.DataFrame, features: list[str], exclude: str | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и стандартное отклонение признаков ТОЛЬКО по обучающим записям.

    Тестовые окна в расчёт не входят никогда. ``exclude`` выбрасывает ещё один
    обучающий эксперимент (для честного варианта «без самовклада»). Разброс
    возвращается как есть, без замены нулей: нулевой разброс далее просто
    исключается из расстояния.
    """
    train_mask = frame["источник"] == "обучение"
    if exclude is not None:
        train_mask = train_mask & (frame[EXP] != exclude)
    part = frame.loc[train_mask, features]
    mu = part.mean(axis=0).to_numpy(dtype=float)
    sigma = part.std(axis=0, ddof=0).to_numpy(dtype=float)
    return mu, sigma


def informative_mask(sigma: np.ndarray) -> np.ndarray:
    """Маска признаков, варьирующихся на обучении (разброс строго больше нуля)."""
    return sigma > 0.0


def mean_z_vector(sub: pd.DataFrame, features: list[str], mu: np.ndarray,
                  sigma: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Средний z-вектор записи по информативным признакам."""
    F = sub[features].to_numpy(dtype=float)
    return ((F[:, mask] - mu[mask]) / sigma[mask]).mean(axis=0)


def rms_distance(sub: pd.DataFrame, features: list[str], mu: np.ndarray,
                 sigma: np.ndarray, mask: np.ndarray) -> float:
    """Корень из среднего квадрата ОГРАНИЧЕННЫХ z по информативным признакам.

    Сначала берётся средний z-вектор записи (по её окнам), каждое z ограничивается
    по модулю числом 10 (знак сохраняется), затем — среднее квадратов по
    информативным признакам, затем корень. Метрика ограничена сверху числом 10 и
    сопоставима между записями; постоянные на обучении признаки не входят.
    """
    z = mean_z_vector(sub, features, mu, sigma, mask)
    z = np.sign(z) * np.minimum(np.abs(z), Z_CLIP)
    return float(np.sqrt(np.mean(z ** 2)))


def unbounded_distance(sub: pd.DataFrame, features: list[str], mu: np.ndarray,
                       sigma: np.ndarray, mask: np.ndarray) -> float:
    """То же без ограничения z. ДИАГНОСТИЧЕСКИЙ вариант, непригоден для сравнения.

    Признак с почти нулевым разбросом на обучении даёт вклад в сотни сигм и
    перекрывает остальные 1240 признаков, поэтому величины теряют сопоставимость,
    а корреляции по ним вводят в заблуждение. Только для контроля.
    """
    z = mean_z_vector(sub, features, mu, sigma, mask)
    return float(np.sqrt(np.mean(z ** 2)))


def sigmas_over_limit(sub: pd.DataFrame, features: list[str], mu: np.ndarray,
                      sigma: np.ndarray, mask: np.ndarray) -> int:
    """Сколько информативных признаков записи дали |z| > 10 (выход за пределы)."""
    z = mean_z_vector(sub, features, mu, sigma, mask)
    return int((np.abs(z) > Z_CLIP).sum())


def constant_shift(sub: pd.DataFrame, features: list[str], mu: np.ndarray,
                   const_mask: np.ndarray) -> tuple[float | None, int]:
    """Выход телеметрии за пределы обучения по постоянным признакам.

    Это НЕ расстояние: берётся максимум модуля отклонения среднего значения
    записи от среднего на обучении среди признаков, постоянных на обучении,
    без деления на разброс (в исходных единицах). Второе число — сколько таких
    признаков отклонилось больше чем на 1.
    """
    if not const_mask.any():
        return None, 0
    F = sub[features].to_numpy(dtype=float)
    deviation = np.abs(F.mean(axis=0) - mu)[const_mask]
    return round(float(deviation.max()), 12), int((deviation > 1.0).sum())


# ------------------------------------------------- характеристики окон
def describe_experiment(sub: pd.DataFrame, features: list[str]) -> dict:
    """Метаданные окон одного эксперимента (без обращения к результатам модели)."""
    sub = sub.sort_values(START)
    y = sub[config.TARGET].to_numpy(dtype=int)
    n = int(len(sub))
    n_positive = int((y == 1).sum())
    frac_positive = round(n_positive / n, 10) if n else None

    positions = np.flatnonzero(y == 1)
    first_fault_pos_frac = (round(float(positions[0]) / n, 10)
                            if positions.size and n else None)

    F = sub[features].to_numpy(dtype=float)
    mean_sensor_std = round(float(F.std(axis=0, ddof=0).mean()), 12)
    return {
        "окон": n,
        "n_positive": n_positive,
        "frac_positive": frac_positive,
        "first_fault_pos_frac": first_fault_pos_frac,
        "mean_sensor_std": mean_sensor_std,
    }


def build_characteristics(train: pd.DataFrame, test: pd.DataFrame,
                          report: dict) -> tuple[list[dict], list[str], dict]:
    """Таблица характеристик по всем 16 экспериментам и блок здоровья данных."""
    features = feature_columns(train)
    if feature_columns(test) != features:
        raise ValueError("схема тестовой выборки не совпадает с обучающей")

    frame = combined_frame(train, test)

    folds = fold_table(report)
    info = split_info()
    train_experiments = {str(e) for e in info["обучение_эксперименты"]}

    experiments = list(folds)
    # порядок как в отчёте LOEO (нумерация фолдов), он же устойчив для вывода
    experiments.sort(key=lambda e: folds[e]["фолд"])

    mu, sigma = train_stats(frame, features)
    mask = informative_mask(sigma)
    const_mask = ~mask
    n_informative = int(mask.sum())

    rows: list[dict] = []
    warnings: list[str] = []
    for experiment in experiments:
        sub = frame[frame[EXP] == experiment]
        source = source_of(experiment, train_experiments)
        row = {"эксперимент": experiment, "источник": source}
        row.update(describe_experiment(sub, features))

        # каноническое расстояние: только варьирующиеся на обучении признаки,
        # z ограничено по модулю числом 10
        row["признаков_в_расстоянии"] = n_informative
        dist = rms_distance(sub, features, mu, sigma, mask)
        row["dist_to_train"] = round(dist, 12)
        row["признаков_с_выходом_за_10_сигм"] = sigmas_over_limit(
            sub, features, mu, sigma, mask)

        # честный вариант: для обучающего эксперимента обучающая статистика
        # пересчитывается без него самого, тест остаётся в общем пространстве
        if source == "обучение":
            mu_e, sigma_e = train_stats(frame, features, exclude=experiment)
            mask_e = informative_mask(sigma_e)
            dist_looe = rms_distance(sub, features, mu_e, sigma_e, mask_e)
            n_looe = int(mask_e.sum())
        else:
            dist_looe, n_looe = dist, n_informative
        row["dist_to_train_looe"] = round(dist_looe, 12)
        row["признаков_в_расстоянии_looe"] = n_looe

        # выход за пределы обучения по постоянным признакам — не расстояние
        shift, n_shift = constant_shift(sub, features, mu, const_mask)
        row["сдвиг_в_постоянных_признаках"] = shift
        row["постоянных_признаков_с_отклонением_больше_1"] = n_shift

        # диагностический вариант без ограничения z, помечен непригодным
        row[UNBOUNDED_DIST_FIELD] = round(
            unbounded_distance(sub, features, mu, sigma, mask), 12)

        fold = folds[experiment]
        for field in LOEO_FIELDS:
            row[field] = fold.get(field)
        if row["окон"] != int(fold["объектов"]):
            warnings.append(
                f"эксперимент {experiment}: окон в таблице {row['окон']}, "
                f"в отчёте LOEO {int(fold['объектов'])}"
            )
        rows.append(row)

    health = {
        "всего_признаков": int(len(features)),
        "варьирующихся_на_обучении": n_informative,
        "постоянных_на_обучении": int(const_mask.sum()),
        "входят_в_расстояние": "только варьирующиеся на обучении признаки",
        "по_экспериментам": [
            {
                "эксперимент": row["эксперимент"],
                "источник": row["источник"],
                "сдвиг_в_постоянных_признаках": row["сдвиг_в_постоянных_признаках"],
                "постоянных_признаков_с_отклонением_больше_1":
                    row["постоянных_признаков_с_отклонением_больше_1"],
            }
            for row in rows
        ],
        "смысл_сдвига": (
            "максимум модуля отклонения среднего значения записи от среднего на "
            "обучении среди признаков, постоянных на обучении, в исходных единицах"
        ),
        "чем_это_рискованно": (
            "постоянные на обучении признаки модель либо не использует (они не несут "
            "различий на обучении), либо использует с неустановленной надёжностью: "
            "наблюдённого разброса по ним нет. Большой сдвиг означает, что телеметрия "
            "вышла за пределы, наблюдавшиеся на обучении, и трактовать его как "
            "расстояние нельзя."
        ),
    }
    return rows, warnings, health


# ---------------------------------------------------------- корреляции
def spearman_block(rows: list[dict], target: str,
                   chars: list[str] = CHARS) -> tuple[dict, list[dict]]:
    """Коэффициент Спирмена и p-значение для каждой характеристики.

    Возвращает словарь «характеристика -> {коэффициент, p_value, n}» и плоский
    список пар для дальнейшего поиска максимального модуля.
    """
    out: dict[str, dict] = {}
    flat: list[dict] = []
    for char in chars:
        pairs = [(r[char], r[target]) for r in rows
                 if r.get(char) is not None and r.get(target) is not None]
        if len(pairs) < 3:
            out[char] = {"коэффициент": None, "p_value": None, "n": len(pairs)}
            continue
        xs = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)
        if np.ptp(xs) == 0 or np.ptp(ys) == 0:
            out[char] = {"коэффициент": None, "p_value": None, "n": len(pairs)}
            continue
        coef, p_value = spearmanr(xs, ys)
        coef = float(coef)
        p_value = float(p_value)
        if np.isnan(coef) or np.isnan(p_value):
            out[char] = {"коэффициент": None, "p_value": None, "n": len(pairs)}
            continue
        out[char] = {"коэффициент": round(coef, 10),
                     "p_value": round(p_value, 10), "n": len(pairs)}
        flat.append({"цель": target, "характеристика": char,
                     "коэффициент": out[char]["коэффициент"],
                     "p_value": out[char]["p_value"], "n": len(pairs)})
    return out, flat


def build_correlations(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Два набора: все 16 фолдов и только 12 обучающих."""
    all_rows = list(rows)
    train_rows = [r for r in rows if r["источник"] == "обучение"]

    blocks: dict[str, dict] = {}
    flat: list[dict] = []
    for name, subset in (("все_16", all_rows), ("обучение_12", train_rows)):
        block: dict[str, dict] = {}
        for target in TARGETS:
            by_char, flat_part = spearman_block(subset, target)
            block[target] = by_char
            flat.extend({"набор": name, **item} for item in flat_part)
        blocks[name] = block
    return blocks, flat


# --------------------------------------------------------------- группы
def summarize_group(rows: list[dict]) -> dict:
    """Состав и средние характеристики группы."""
    def mean_of(key: str):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(values)), 10) if values else None

    return {
        "эксперименты": [r["эксперимент"] for r in rows],
        "размер": len(rows),
        "f1_среднее": mean_of("f1"),
        "frac_positive_среднее": mean_of("frac_positive"),
        "first_fault_pos_frac_среднее": mean_of("first_fault_pos_frac"),
        "mean_sensor_std_среднее": mean_of("mean_sensor_std"),
        "dist_to_train_среднее": mean_of("dist_to_train"),
        "dist_to_train_looe_среднее": mean_of("dist_to_train_looe"),
    }


def build_groups(rows: list[dict]) -> dict:
    """«Хорошие» (F-мера > 0.8) и «плохие» (F-мера < 0.5) фолды."""
    good = [r for r in rows if r["f1"] is not None and r["f1"] > 0.8]
    bad = [r for r in rows if r["f1"] is not None and r["f1"] < 0.5]
    return {
        "хорошие": {**summarize_group(good), "порог": "F-мера > 0.8"},
        "плохие": {**summarize_group(bad), "порог": "F-мера < 0.5"},
        "примечание": ("группы малы, поэтому сравнение средних описательное, "
                       "а не доказательное"),
    }


# --------------------------------------------------------------- вывод
def build_conclusion(flat: list[dict]) -> str:
    """Вывод строго по правилу: сначала ищем максимальный модуль корреляции."""
    all16 = [item for item in flat if item["набор"] == "все_16"]
    scored = [item for item in all16 if item["коэффициент"] is not None]
    if not scored:
        return ("Корреляции не вычислимы: характеристики постоянны. Разброс F-меры "
                "не объясняется доступными метаданными.")
    best = max(scored, key=lambda item: abs(item["коэффициент"]))
    max_abs = abs(best["коэффициент"])

    if max_abs < TAU:
        subset = [item for item in flat
                  if item["набор"] == "обучение_12" and item["коэффициент"] is not None
                  and abs(item["коэффициент"]) >= TAU]
        subset_note = ""
        if subset:
            names12 = ", ".join(
                f"{item['цель']} ~ {item['характеристика']} (r = {item['коэффициент']:.3f}, "
                f"p = {item['p_value']:.3f}, n = {item['n']})"
                for item in sorted(subset, key=lambda i: -abs(i["коэффициент"]))
            )
            subset_note = (
                " Отдельно оговорено: только на 12 обучающих фолдах такие связи "
                f"сильнее — {names12}. Но это подмножество тех же наблюдений, "
                "отобранное по источнику записи, поэтому при n = 12 и без поправки "
                "на множественные сравнения оно тоже не является доказательством."
            )
        return (
            "Максимальный модуль коэффициента Спирмена по всем 16 фолдам равен "
            f"{max_abs:.3f} и меньше {TAU:.1f}. Значит, разброс F-меры НЕ объясняется "
            "доступными метаданными — долей отказов, положением первого отказа, "
            "разбросом сенсоров и удалённостью записи от обучающей части. "
            "Это указывает на нехватку данных, а не на найденную закономерность."
            + subset_note
        )

    strong = [item for item in scored if abs(item["коэффициент"]) >= TAU]
    names = ", ".join(
        f"{item['цель']} ~ {item['характеристика']} (r = {item['коэффициент']:.3f}, "
        f"p = {item['p_value']:.3f})"
        for item in sorted(strong, key=lambda i: -abs(i["коэффициент"]))
    )
    return (
        f"Максимальный модуль коэффициента Спирмена по всем 16 фолдам равен "
        f"{max_abs:.3f}. Порог {TAU:.1f} превышают: {names}. Однако при n = 16 "
        "(и тем более при n = 12) и без поправки на множественные сравнения это "
        "не доказательство связи: даже при отсутствии реального эффекта один "
        "коэффициент из десяти может случайно оказаться большим. Требуется больше "
        "экспериментов, прежде чем считать связь установленной."
    )


# ------------------------------------------------------------ графики
def _scatter(rows: list[dict], xkey: str, xlabel: str, fname: str,
             correlation: dict, target: str = "f1",
             title_label: str | None = None) -> Path:
    """Одна диаграмма рассеяния: F-мера фолда против характеристики окна.

    ``xlabel`` подписывает ось (может быть в несколько строк), ``title_label``
    задаёт короткое имя характеристики для заголовка.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))

    styles = {"обучение": ("#2b6cb0", "o"), "тест": ("#b3261e", "s")}
    for source, (color, marker) in styles.items():
        subset = [r for r in rows if r["источник"] == source and r[xkey] is not None]
        if not subset:
            continue
        ax.scatter([r[xkey] for r in subset], [r[target] for r in subset],
                   c=color, marker=marker, s=55, label=source, zorder=3)
        for r in subset:
            ax.annotate(str(r["эксперимент"]), (r[xkey], r[target]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

    coef = correlation.get("коэффициент")
    p_value = correlation.get("p_value")
    if coef is None:
        title_stat = "коэффициент Спирмена не определён"
    else:
        title_stat = f"коэффициент Спирмена = {coef:.3f}, p = {p_value:.3f}"
    ax.set_title(f"F-мера против «{title_label or xlabel}»: {title_stat}", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F-мера фолда (порог подобран внутри фолда)")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(title="источник записи")
    fig.tight_layout()
    path = config.FIGURES_DIR / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


TEXT_FRAC = "доля окон с отказом"
TEXT_FIRST = "доля позиции первого отказа"
TEXT_DIST = "ограниченное z-расстояние"
TEXT_DIST_AXIS = ("ограниченное z-расстояние до обучающих записей,\n"
                  "сигмы на признак (каждое z ограничено по модулю 10)")
TEXT_STD = "средний разброс сенсоров"


def make_figures(rows: list[dict], correlations: dict) -> list[Path]:
    """Три обязательные диаграммы рассеяния."""
    block = correlations["все_16"]["f1"]
    return [
        _scatter(rows, "frac_positive", TEXT_FRAC, "09_diag_f1_vs_frac_positive.png",
                 block["frac_positive"]),
        _scatter(rows, "dist_to_train", TEXT_DIST_AXIS,
                 "10_diag_f1_vs_dist_to_train.png", block["dist_to_train"],
                 title_label=TEXT_DIST),
        _scatter(rows, "first_fault_pos_frac", TEXT_FIRST,
                 "11_diag_f1_vs_first_fault.png", block["first_fault_pos_frac"]),
    ]


# ------------------------------------------------------------- сборка
def compute() -> dict:
    """Полный результат диагностики (без записи файлов)."""
    report = load_loeo()
    train, test, _ = load_frames()
    rows, warnings, health = build_characteristics(train, test, report)
    correlations, flat = build_correlations(rows)
    groups = build_groups(rows)
    conclusion = build_conclusion(flat)

    good_size = groups["хорошие"]["размер"]
    bad_size = groups["плохие"]["размер"]
    warnings.append(
        "размеры групп малы (обучающих записей всего "
        f"{config.N_TRAIN_EXPERIMENTS}): «хорошие» — {good_size}, "
        f"«плохие» — {bad_size}; сравнение средних описательное, а не доказательное"
    )

    rules = (
        "Числа модели (F-мера, порог, recall, precision, число эпох, контроль порога) "
        "взяты как есть из отчёта LOEO по обучению с исключением одного эксперимента. "
        "Тестовые записи (441, 442, 443, 445) не использованы ни для подбора модели, "
        "ни для подбора порога, ни для любой настройки: они только описываются готовыми "
        "числами. Среднее и разброс для z-нормализации считаются ТОЛЬКО по обучающим "
        "записям; тестовые окна в них не входят. Расстояние считается только по "
        "признакам, варьирующимся на обучении, как корень из среднего квадрата "
        "z-отклонения, причём каждое z ограничено по модулю числом 10 сигм: "
        "единичный признак с почти нулевым разбросом не может перекрыть остальные, "
        "поэтому величина ограничена сверху числом 10 и сопоставима между записями. "
        "Признаки, постоянные на обучении, в расстояние не входят и описываются "
        "отдельной мерой выхода за пределы обучения."
    )
    formulas = {
        "frac_positive": "доля окон с отказом в записи (число окон отказа / число окон)",
        "first_fault_pos_frac": (
            "положение первого окна с отказом при нумерации с нуля в порядке "
            "возрастания начала окна, делённое на число окон; если отказов нет — пусто"
        ),
        "mean_sensor_std": (
            "среднее по всем 1620 признакам от их стандартного отклонения, "
            "посчитанного по окнам этой записи"
        ),
        "признаков_в_расстоянии": (
            "число признаков, варьирующихся на обучении (разброс по обучению строго "
            "больше нуля); только они входят в dist_to_train"
        ),
        "dist_to_train": (
            "ограниченное z-расстояние до обучающих записей: корень из среднего "
            "квадрата z-отклонения по признакам, варьирующимся на обучении, где "
            "каждое z ограничено по модулю числом 10 сигм (знак сохраняется); "
            "ограничено сверху числом 10 и сопоставимо между записями"
        ),
        "dist_to_train_looe": (
            "то же ограниченное расстояние, но для обучающей записи обучающие "
            "статистики пересчитаны без неё самой; для тестовой записи совпадает "
            "с dist_to_train"
        ),
        "признаков_с_выходом_за_10_сигм": (
            "сколько признаков записи дали |z| > 10 до ограничения; индикатор выхода "
            "за пределы обучающего диапазона. В само расстояние такие z входят уже "
            "ограниченными"
        ),
        "сдвиг_в_постоянных_признаках": (
            "максимум модуля отклонения среднего значения записи от среднего на "
            "обучении среди признаков, постоянных на обучении, в исходных единицах; "
            "это не расстояние, а мера выхода телеметрии за пределы обучения"
        ),
        "постоянных_признаков_с_отклонением_больше_1": (
            "сколько постоянных на обучении признаков отклонились от среднего на "
            "обучении больше чем на 1 в исходных единицах"
        ),
        UNBOUNDED_DIST_FIELD: (
            "ДИАГНОСТИЧЕСКИЙ вариант: то же расстояние, но БЕЗ ограничения z. "
            "НЕПРИГОДЕН для сравнения записей и для корреляций: признак с почти "
            "нулевым (но не нулевым) разбросом на обучении даёт вклад в сотни сигм "
            "и перекрывает остальные признаки. Приведён только для контроля, "
            "использовать вместо dist_to_train нельзя"
        ),
        "f1": "F-мера фолда при пороге, подобранном внутри фолда",
        "f1_в_пороге_0.5": "F-мера того же фолда при фиксированном пороге 0.5",
    }

    return {
        "правила": rules,
        "формулы": formulas,
        "характеристики": [_clean_row(r) for r in rows],
        "здоровье_данных": health,
        "корреляции": correlations,
        "группы": groups,
        "вывод": conclusion,
        "предупреждения": warnings,
    }


def _clean_row(row: dict) -> dict:
    """Приведение строки к обычным типам: без numpy и без NaN (пусто -> null)."""
    out: dict = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        out[key] = value
    return out


def write_report(result: dict) -> Path:
    """Запись итогового JSON строго по-русски и без NaN."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "loeo_diagnostics.json"
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------- вход
def main() -> int:
    log = config.setup_logging()
    # ВНИМАНИЕ: config.set_seed() не вызывается намеренно — он тянет tensorflow,
    # а этот скрипт обязан его не импортировать. Все расчёты детерминированы,
    # но семя всё же фиксируем для numpy и random.
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    log.info("диагностика разброса LOEO: старт")
    if not config.LOEO_PATH.exists():
        log.error("нет отчёта LOEO: %s", config.LOEO_PATH)
        return 1
    result = compute()

    path = write_report(result)
    figures = make_figures(result["характеристики"], result["корреляции"])

    health = result["здоровье_данных"]
    log.info("признаков всего %d; варьируются на обучении %d; постоянны на обучении %d",
             health["всего_признаков"], health["варьирующихся_на_обучении"],
             health["постоянных_на_обучении"])
    for row in result["характеристики"]:
        log.info(
            "%-4s %-8s окон=%2d отказов=%2d доля=%.2f первый=%-5s разброс=%.4f "
            "признаков_в_расстоянии=%d выход_за_10=%d удал=%.3f удал_честная=%.3f "
            "сдвиг=%.3f сдвиг>1=%d F1=%.3f F1@0.5=%.3f",
            row["эксперимент"], row["источник"], row["окон"], row["n_positive"],
            row["frac_positive"],
            "пусто" if row["first_fault_pos_frac"] is None
            else f"{row['first_fault_pos_frac']:.2f}",
            row["mean_sensor_std"], row["признаков_в_расстоянии"],
            row["признаков_с_выходом_за_10_сигм"],
            row["dist_to_train"], row["dist_to_train_looe"],
            row["сдвиг_в_постоянных_признаках"],
            row["постоянных_признаков_с_отклонением_больше_1"],
            row["f1"], row["f1_в_пороге_0.5"],
        )
    for name, block in result["корреляции"].items():
        for target, by_char in block.items():
            for char, stats in by_char.items():
                log.info("%s | %s ~ %s: r=%s, p=%s, n=%d", name, target, char,
                         stats["коэффициент"], stats["p_value"], stats["n"])
    log.info("группа «хорошие»: %s", result["группы"]["хорошие"]["эксперименты"])
    log.info("группа «плохие»: %s", result["группы"]["плохие"]["эксперименты"])
    for warning in result["предупреждения"]:
        log.warning("%s", warning)
    log.info("вывод: %s", result["вывод"])
    log.info("отчёт: %s", path)
    for figure in figures:
        log.info("график: %s", figure)
    log.info("диагностика разброса LOEO: готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
