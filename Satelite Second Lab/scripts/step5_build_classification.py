# -*- coding: utf-8 -*-
r"""Шаг 5. Датасет ADAPT для бинарной классификации «Норма/Отказ».

Вход:  data\processed\adapt_raw.parquet, data\processed\adapt_log.parquet
Выход: data\processed\adapt_train.parquet, data\processed\adapt_test.parquet

Ключевые решения (обоснованы в отчёте):
* только record_type == "AntagonistData";
* Time (s) лога отсчитывается от первого отсчёта эксперимента (проверено
  по событиям FaultInject: расхождение <= 0.24 с против 0.97 с от StartTime);
* у 9 из 16 экспериментов несколько отказов -> берётся первый (min Time (s));
* окна строятся ВНУТРИ каждого эксперимента: окно никогда не пересекает
  границу между файлами, поэтому утечки между Train и Test нет;
* метка окна = fault последнего шага.

Запуск: py -X utf8 scripts\step5_build_classification.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Satelite Second Lab")
PROC = ROOT / "data" / "processed"
FMT = "%Y-%m-%d %H:%M:%S.%f"

WINDOW = 20
STRIDE = 10
TEST_EXPS = {"441", "442", "443", "445"}

R: list[str] = []


def p(s: str = "") -> None:
    R.append(s)


# ------------------------------------------------------------------ загрузка
raw = pd.read_parquet(PROC / "adapt_raw.parquet")
log = pd.read_parquet(PROC / "adapt_log.parquet")
SENSORS = [c for c in raw.columns
           if c not in ("source_file", "line_no", "record_type", "Time")]

p("=== ДАТАСЕТ ADAPT: БИНАРНАЯ КЛАССИФИКАЦИЯ ===")
p(f"window_size={WINDOW}, stride={STRIDE}, сенсоров={len(SENSORS)}")
p(f"Test-эксперименты: {sorted(TEST_EXPS, key=int)}")
p(f"Всего строк в adapt_raw: {len(raw)}; измерения: "
  f"{int((raw['record_type'] == 'AntagonistData').sum())}")
p()

# ------------------------------------------------------- подготовка измерений
meas = raw[raw["record_type"] == "AntagonistData"].copy()
meas["exp"] = meas["source_file"].str.extract(r"Exp_(\d+)_", expand=False)
meas["ts"] = pd.to_datetime(meas["Time"].str[:23], format=FMT)
meas = meas.sort_values(["source_file", "line_no"]).reset_index(drop=True)
meas["t0"] = meas.groupby("source_file")["ts"].transform("first")
meas["elapsed"] = (meas["ts"] - meas["t0"]).dt.total_seconds()

onset = log.groupby("Experiment")["Time (s)"].min()
n_log = log.groupby("Experiment").size()
meas["onset"] = meas["exp"].map(onset).astype(float)
if meas["onset"].isna().any():
    raise SystemExit("!! есть измерения без записи в логе")
meas["fault"] = (meas["elapsed"] >= meas["onset"]).astype("int8")

p("=== ВРЕМЯ ОТКАЗА И РАЗМЕТКА ===")
p(f"{'exp':>5} {'N':>5} {'лог.записей':>11} {'onset,с':>8} {'индекс 1-го fault':>18} "
  f"{'fault=0':>8} {'fault=1':>8}")
for e, sub in meas.groupby("exp", sort=True):
    i1 = int(np.argmax(sub["fault"].to_numpy() == 1)) if (sub["fault"] == 1).any() else -1
    p(f"{e:>5} {len(sub):>5} {int(n_log[e]):>11} {onset[e]:>8.3f} {i1:>18} "
      f"{int((sub['fault'] == 0).sum()):>8} {int((sub['fault'] == 1).sum()):>8}")
p("(для экспериментов с несколькими отказами onset = первый отказ, min Time (s))")
p("(индекс 1-го fault = -1, если отказ не наступил в пределах измерений)")

# проверка: не появились ли команды после отказа (ремонт не предусмотрен ТЗ)
cmd = raw[raw["record_type"].isin(["AntagonistCommand", "UserCommand", "FaultInject"])].copy()
cmd["exp"] = cmd["source_file"].str.extract(r"Exp_(\d+)_", expand=False)
cmd["ts"] = pd.to_datetime(cmd["Time"].str[:23], format=FMT)
after = 0
for e, sub in cmd.groupby("exp"):
    t0 = meas.loc[meas["exp"] == e, "t0"].iloc[0]
    after += int((((sub["ts"] - t0).dt.total_seconds()) >= onset[e]).sum())
p(f"Служебных событий (команды/FaultInject) после отказа: {after}")
p()

# ------------------------------------------------------------------- окна
feat_names = [f"{t:02d}_{s}" for t in range(WINDOW) for s in SENSORS]


def make_windows(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    per: list[dict] = []
    for fname, sub in frame.groupby("source_file", sort=True):
        sub = sub.sort_values("line_no").reset_index(drop=True)
        n = len(sub)
        if n < WINDOW:
            per.append({"split": split, "source_file": fname, "N": n, "windows": 0,
                        "fault0": 0, "fault1": 0, "exp": sub["exp"].iloc[0]})
            continue
        X = sub[SENSORS].astype(float).to_numpy()
        lab = sub["fault"].to_numpy(dtype=int)
        ts = sub["ts"].to_numpy()
        n0 = n1 = 0
        for w, s in enumerate(range(0, n - WINDOW + 1, STRIDE)):
            e = s + WINDOW - 1
            block = X[s:e + 1]
            rec = {
                "meta__split": split,
                "meta__experiment": sub["exp"].iloc[0],
                "meta__source_file": fname,
                "meta__window_id": w,
                "meta__start_index": s,
                "meta__end_index": e,
                "meta__t_start": str(sub["Time"].iloc[s]),
                "meta__t_end": str(sub["Time"].iloc[e]),
                "meta__span_s": float((ts[e] - ts[s]) / np.timedelta64(1, "s")),
                "meta__n_fault_steps": int(lab[s:e + 1].sum()),
                "fault": int(lab[e]),
            }
            rec.update({feat_names[t * len(SENSORS) + j]: float(block[t, j])
                        for t in range(WINDOW) for j in range(len(SENSORS))})
            rows.append(rec)
            n1 += int(lab[e])
            n0 += int(lab[e] == 0)
        per.append({"split": split, "source_file": fname, "N": n,
                    "windows": len(range(0, n - WINDOW + 1, STRIDE)),
                    "fault0": n0, "fault1": n1, "exp": sub["exp"].iloc[0]})
    return pd.DataFrame(rows), pd.DataFrame(per)


train = meas[~meas["exp"].isin(TEST_EXPS)]
test = meas[meas["exp"].isin(TEST_EXPS)]
train_w, train_per = make_windows(train, "train")
test_w, test_per = make_windows(test, "test")

p("=== ОКНА ПО ЭКСПЕРИМЕНТАМ ===")
p(f"{'split':>6} {'exp':>5} {'отсчётов':>9} {'окон':>6} {'fault=0':>8} {'fault=1':>8} {'доля 1':>7}")
for per in (train_per, test_per):
    for _, r in per.sort_values("exp").iterrows():
        frac = r["fault1"] / r["windows"] if r["windows"] else 0
        p(f"{r['split']:>6} {r['exp']:>5} {r['N']:>9} {r['windows']:>6} "
          f"{r['fault0']:>8} {r['fault1']:>8} {frac:>6.0%}")
p()

# ------------------------------------------------------------------ проверки
p("=== ПРОВЕРКИ ===")
tr_exp = set(train_per["exp"])
te_exp = set(test_per["exp"])
p(f"[{'OK' if not (tr_exp & te_exp) else '!!'}] Пересечение экспериментов Train/Test: "
  f"{sorted(tr_exp & te_exp) or 'пусто'}")
p(f"[OK] Train: {len(tr_exp)} экспериментов | Test: {len(te_exp)} экспериментов")
p(f"[{'OK' if tr_exp | te_exp == set(meas['exp']) else '!!'}] Все 16 экспериментов распределены")

# окно целиком лежит внутри своего файла (не пересекает границу экспериментов)
sizes = meas.groupby("source_file").size().to_dict()
bad_bounds = 0
for w in (train_w, test_w):
    bad_bounds += int((w["meta__end_index"] != w["meta__start_index"] + WINDOW - 1).sum())
    bad_bounds += int((w.apply(lambda r: r["meta__end_index"] >= sizes[r["meta__source_file"]],
                               axis=1)).sum())
p(f"[{'OK' if bad_bounds == 0 else '!!'}] Окно целиком внутри своего файла "
  f"(границы экспериментов не пересекаются): нарушений {bad_bounds}")
dup = int(sum(w.duplicated(subset=["meta__source_file", "meta__start_index"]).sum()
              for w in (train_w, test_w)))
p(f"[{'OK' if dup == 0 else '!!'}] Повторов (файл, старт окна): {dup}")

tr_files = set(train_w["meta__source_file"])
te_files = set(test_w["meta__source_file"])
p(f"[{'OK' if not (tr_files & te_files) else '!!'}] Файлы Train/Test не пересекаются: "
  f"{sorted(tr_files & te_files) or 'пусто'}")

# согласованность метки: label = fault последнего шага
mis = 0
for fname, sub in meas.groupby("source_file", sort=True):
    sub = sub.sort_values("line_no").reset_index(drop=True)
    lab = sub["fault"].to_numpy(dtype=int)
    for (_, r) in train_w[train_w["meta__source_file"] == fname].iterrows():
        if r["fault"] != lab[int(r["meta__end_index"])]:
            mis += 1
    for (_, r) in test_w[test_w["meta__source_file"] == fname].iterrows():
        if r["fault"] != lab[int(r["meta__end_index"])]:
            mis += 1
p(f"[{'OK' if mis == 0 else '!!'}] Метка = fault последнего шага: расхождений {mis}")

nan_feat = int(train_w[feat_names].isna().sum().sum() + test_w[feat_names].isna().sum().sum())
p(f"[{'OK' if nan_feat == 0 else '!!'}] Пропусков в признаках: {nan_feat}")

# сколько окон «шагают через» момент отказа (метка 0, но внутри есть fault=1)
straddle_tr = int(((train_w["fault"] == 0) & (train_w["meta__n_fault_steps"] > 0)).sum())
straddle_te = int(((test_w["fault"] == 0) & (test_w["meta__n_fault_steps"] > 0)).sum())
p(f"[i] Окон с меткой 0, но содержащих шаги отказа: Train {straddle_tr}, Test {straddle_te}")
p(f"[i] Длительность окна, с: Train {train_w['meta__span_s'].min():.2f}…"
  f"{train_w['meta__span_s'].max():.2f}, Test {test_w['meta__span_s'].min():.2f}…"
  f"{test_w['meta__span_s'].max():.2f}")
p()

# ------------------------------------------------------------------ запись
train_w.to_parquet(PROC / "adapt_train.parquet", index=False)
test_w.to_parquet(PROC / "adapt_test.parquet", index=False)

p("=== СТАТИСТИКА ===")
for name, w in (("Train", train_w), ("Test", test_w)):
    n = len(w)
    n1 = int(w["fault"].sum())
    p(f"{name}: окон {n} | fault=0 {n - n1} | fault=1 {n1} "
      f"({n1 / n:.0%} отказ) | экспериментов {w['meta__experiment'].nunique()}")
p(f"Итого окон: {len(train_w) + len(test_w)} "
  f"({len(train_w)} train + {len(test_w)} test)")
p(f"Признаков в окне: {len(feat_names)} ({WINDOW} шагов × {len(SENSORS)} сенсоров)")
p()
p("=== ФАЙЛЫ ===")
p(f"{PROC / 'adapt_train.parquet'} — {len(train_w)} × {train_w.shape[1]}")
p(f"{PROC / 'adapt_test.parquet'} — {len(test_w)} × {test_w.shape[1]}")
p()
p("=== КАК ПОЛЬЗОВАТЬСЯ ===")
p("- признаки: колонки вида NN_SENSOR (например 00_E135 … 19_IT340), всего 1620;")
p("- целевая метка: колонка fault (0/1);")
p("- колонки meta__* — служебные, в модель НЕ подавать (индекс/время коррелируют "
  "с меткой);")
p("- разбиение уже честное по экспериментам: Test-эксперименты в Train отсутствуют;")
p("- окна берутся только внутри одного файла, поэтому границы не пересекаются.")

(PROC / "adapt_classification_report.txt").write_text("\n".join(R), encoding="utf-8")
(PROC / "adapt_classification_meta.json").write_text(json.dumps({
    "window_size": WINDOW, "stride": STRIDE, "n_sensors": len(SENSORS),
    "sensors": SENSORS, "test_experiments": sorted(TEST_EXPS, key=int),
    "train_experiments": sorted(set(meas["exp"]) - TEST_EXPS, key=int),
    "feature_columns": feat_names, "target": "fault",
    "window_label_rule": "fault последнего шага окна",
    "time_reference": "первый отсчёт эксперимента (проверено по FaultInject)",
    "multi_fault_rule": "onset = min(Time (s)) по эксперименту",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("OK, report:", PROC / "adapt_classification_report.txt")
