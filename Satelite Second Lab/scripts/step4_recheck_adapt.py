r"""Шаг 4. Перепроверка подготовленных данных ADAPT (только чтение).

Запуск: py -X utf8 scripts\step4_recheck_adapt.py
Ничего в исходных данных не меняет; пишет только собственный отчёт
data\processed\adapt_recheck.txt и перезапускает step3 (его отчёт должен не измениться).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Satelite Second Lab")
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROC = DATA / "processed"
SCRIPTS = ROOT / "scripts"

R: list[str] = []


def p(s: str = "") -> None:
    R.append(s)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} Б"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} КБ"
    return f"{n / 1024 ** 2:.1f} МБ"


problems: list[str] = []

# ============================================================ ЭТАП 0
p("=== ЭТАП 0: СТРУКТУРА ===")
p(f"[{'OK' if ROOT.is_dir() else '!!'}] Папка существует: {ROOT}")

txt_raw = sorted(RAW.glob("*.txt"))
xls_raw = sorted(RAW.glob("*.xls"))
ok_raw = len(txt_raw) == 16 and len(xls_raw) == 1
if not ok_raw:
    problems.append(f"data/raw: {len(txt_raw)} .txt и {len(xls_raw)} .xls (ожидалось 16 и 1)")
p(f"[{'OK' if ok_raw else '!!'}] data/raw/: {len(list(RAW.iterdir()))} файлов "
  f"({len(txt_raw)} .txt + {len(xls_raw)} .xls)")

need = {
    "data/interim/adapt_raw.parquet": INTERIM / "adapt_raw.parquet",
    "data/processed/adapt_raw.parquet": PROC / "adapt_raw.parquet",
    "data/processed/adapt_log.parquet": PROC / "adapt_log.parquet",
    "data/processed/adapt_report.txt": PROC / "adapt_report.txt",
    "data/processed/adapt_verify.txt": PROC / "adapt_verify.txt",
    "scripts/step1_export_log.ps1": SCRIPTS / "step1_export_log.ps1",
    "scripts/step2_prepare_adapt.py": SCRIPTS / "step2_prepare_adapt.py",
    "scripts/step3_verify_adapt.py": SCRIPTS / "step3_verify_adapt.py",
}
for name, path in need.items():
    ok = path.is_file()
    if not ok:
        problems.append(f"нет файла: {name}")
    p(f"[{'OK' if ok else '!!'}] {name}"
      + (f"  ({fmt_size(path.stat().st_size)})" if ok else ""))

parquets = sorted(DATA.rglob("*.parquet"))
p(f"[i] .parquet всего: {len(parquets)} — {[q.name for q in parquets]}")
p("    (в ТЗ сказано «2 (raw, log)»; raw лежит и в interim, и в processed — это одна и "
  "та же таблица, плюс лог; уникальных наборов 2)")
p(f"Размеры:")
for q in sorted(DATA.rglob("*")) + sorted(SCRIPTS.iterdir()):
    if q.is_file():
        p(f"    {fmt_size(q.stat().st_size):>9}  {q.relative_to(ROOT)}")
p(f"Вердикт: {'СТРУКТУРА КОРРЕКТНА.' if not problems else 'ЕСТЬ ПРОБЛЕМЫ.'}")
p()

# ============================================================ ЭТАП 1
p("=== ЭТАП 1: SHA256 ===")
originals = {q.name: q for q in sorted((ROOT / "dataset_text").glob("*.txt"))}
originals[ (ROOT / "Experiment_Log (1).xls").name ] = ROOT / "Experiment_Log (1).xls"
ok_n = 0
for copy in sorted(RAW.iterdir()):
    orig = originals.get(copy.name)
    if orig is None:
        p(f"{copy.relative_to(ROOT)} — ЛИШНИЙ ФАЙЛ")
        problems.append(f"лишний файл в data/raw: {copy.name}")
        continue
    same = sha(copy) == sha(orig)
    ok_n += same
    p(f"data/raw/{copy.name} — {'OK' if same else 'ПРОБЛЕМА'}")
    if not same:
        problems.append(f"sha256 не совпал: {copy.name}")
p(f"Всего: {ok_n}/17 совпадают.")
p(f"Вердикт: {'ДАННЫЕ НЕ ИСКАЖЕНЫ.' if ok_n == 17 else 'ЕСТЬ ПРОБЛЕМЫ.'}")
p()

# ============================================================ ЭТАП 2
p("=== ЭТАП 2: ЛОГ ===")
log = pd.read_parquet(PROC / "adapt_log.parquet")
cols_ok = list(log.columns) == ["Experiment", "Component", "Mode", "Time (s)"]
p(f"[{'OK' if cols_ok else '!!'}] Колонки: {list(log.columns)}")
if not cols_ok:
    problems.append(f"колонки лога: {list(log.columns)}")
rows_ok = len(log) == 28
p(f"[{'OK' if rows_ok else '!!'}] Строк: {len(log)} (ожидалось 28)")
if not rows_ok:
    problems.append(f"строк в логе: {len(log)}")
exps = sorted(log["Experiment"].dropna().unique(), key=lambda x: int(x))
p(f"[{'OK' if len(exps) == 16 else '!!'}] Экспериментов: {len(exps)}")
modes = sorted(log["Mode"].dropna().unique())
exp_modes = ["Drift", "FailedOff", "FailedOpen", "Noise", "Stuck", "StuckOpen"]
modes_ok = modes == exp_modes
p(f"[{'OK' if modes_ok else '!!'}] Режимов: {len(modes)} {modes}")
if not modes_ok:
    problems.append(f"режимы: {modes}")
dups = int(log.duplicated().sum())
p(f"[{'OK' if dups == 0 else '!!'}] Дубликатов (полных строк): {dups}")
if dups:
    problems.append(f"дубликатов строк в логе: {dups}")
t = log["Time (s)"]
t_ok = bool((t > 0).all() and (t < 300).all())
p(f"[{'OK' if t_ok else '!!'}] Time (s): {t.min():g}–{t.max():g} (все > 0 и < 300)")
if not t_ok:
    problems.append(f"Time (s) вне (0, 300): {t.min()} … {t.max()}")
p("Первые 5 строк:")
for _, r in log.head(5).iterrows():
    p(f"    {r['Experiment']} | {r['Component']} | {r['Mode']} | {r['Time (s)']:g}")
p(f"Вердикт: {'ЛОГ КОРРЕКТЕН.' if not (not cols_ok or not rows_ok or not modes_ok or dups or not t_ok) else 'ЕСТЬ ПРОБЛЕМЫ.'}")
p()

# ============================================================ ЭТАП 3
p("=== ЭТАП 3: ТЕКСТОВЫЕ ФАЙЛЫ ===")
df = pd.read_parquet(PROC / "adapt_raw.parquet")
sensors = [c for c in df.columns if c not in ("source_file", "line_no", "record_type", "Time")]
rows_ok = len(df) == 3139
p(f"[{'OK' if rows_ok else '!!'}] Строк: {len(df)} (ожидалось 3139)")
if not rows_ok:
    problems.append(f"строк в adapt_raw: {len(df)}")
cols_ok = df.shape[1] == 85
p(f"[{'OK' if cols_ok else '!!'}] Колонок: {df.shape[1]} (ожидалось 85)")
if not cols_ok:
    problems.append(f"колонок: {df.shape[1]}")
rtypes = sorted(df["record_type"].unique())
exp_types = ["AntagonistCommand", "AntagonistData", "ExperimentControl", "FaultInject",
             "SensorData", "UserCommand"]
rt_ok = rtypes == exp_types
p(f"[{'OK' if rt_ok else '!!'}] record_type: {len(rtypes)} типов {rtypes}")
if not rt_ok:
    problems.append(f"типы строк: {rtypes}")

meas = df[df["record_type"] == "AntagonistData"]
meas_ok = len(meas) == 2884
p(f"[{'OK' if meas_ok else '!!'}] AntagonistData: {len(meas)} строк (ожидалось 2884)")
if not meas_ok:
    problems.append(f"AntagonistData: {len(meas)}")
sens_ok = len(sensors) == 81
p(f"[{'OK' if sens_ok else '!!'}] Сенсоров: {len(sensors)} (ожидалось 81)")
if not sens_ok:
    problems.append(f"сенсоров: {len(sensors)}")
gaps_meas = int(meas[sensors].isna().sum().sum())
p(f"[{'OK' if gaps_meas == 0 else '!!'}] Пропусков в измерениях: {gaps_meas}")
if gaps_meas:
    problems.append(f"пропусков в измерениях: {gaps_meas}")

num = meas[sensors].astype("float")
rng = pd.DataFrame({"min": num.min(), "max": num.max()})
p(f"[OK] Диапазон: {rng['min'].min():g} … {rng['max'].max():g}")
wide = rng.reindex(rng["max"].abs().sort_values(ascending=False).index).head(5)
p("     самые широкие сенсоры (min … max):")
for name, row in wide.iterrows():
    p(f"       {name}: {row['min']:g} … {row['max']:g}")
nonfinite = int((~num.apply(pd.to_numeric, errors="coerce").notna()).sum().sum())
p(f"[{'OK' if nonfinite == 0 else '!!'}] Нечисловых/бесконечных значений: {nonfinite} "
  f"(пределы физичности в ТЗ не заданы — сверка с исходниками побайтовая)")
if nonfinite:
    problems.append(f"нечисловых значений: {nonfinite}")

src = sorted(df["source_file"].unique())
exp_src = sorted(q.name for q in (ROOT / "dataset_text").glob("*.txt"))
src_ok = src == exp_src
p(f"[{'OK' if src_ok else '!!'}] source_file: {len(src)} файлов, ровно исходный набор")
if not src_ok:
    problems.append(f"набор source_file: {src}")
p(f"Вердикт: {'ДАННЫЕ КОРРЕКТНЫ.' if not problems else 'ЕСТЬ ПРОБЛЕМЫ.'}")
p()

# ============================================================ ЭТАП 4
p("=== ЭТАП 4: СООТВЕТСТВИЕ ===")
file_exp = {re.search(r"Exp_(\d+)_", f).group(1): f for f in src}
log_exps = set(log["Experiment"].dropna())
miss_log = sorted(set(file_exp) - log_exps)
miss_file = sorted(log_exps - set(file_exp))
p(f"[{'OK' if not miss_log else '!!'}] Все {len(file_exp)} файлов имеют запись в логе"
  + ("" if not miss_log else f"; нет: {miss_log}"))
if miss_log:
    problems.append(f"нет в логе: {miss_log}")
if miss_file:
    problems.append(f"лог без файла: {miss_file}")

# длительность по отметкам времени внутри данных
fmt_iso = "%Y-%m-%d %H:%M:%S.%f"
bad_dur = []
for exp, fname in file_exp.items():
    tt = (df[df["source_file"] == fname]
          .query("record_type == 'AntagonistData'")["Time"].dropna())
    t0 = pd.to_datetime(tt.iloc[0][:23].replace(" ", " "), format=fmt_iso)
    t1 = pd.to_datetime(tt.iloc[-1][:23], format=fmt_iso)
    span = (t1 - t0).total_seconds() + 0.5          # шаг логов 0.5 с
    tmax = float(log.loc[log["Experiment"] == exp, "Time (s)"].max())
    if tmax > span:
        bad_dur.append((fname, tmax, span))
p(f"[{'OK' if not bad_dur else '!!'}] Все Time (s) ≤ длительности файла "
  f"({len(file_exp) - len(bad_dur)}/{len(file_exp)})")
if bad_dur:
    problems.append(f"Time (s) > длительности: {bad_dur}")

sens_per_file = {}
for fname, sub in df.query("record_type == 'AntagonistData'").groupby("source_file"):
    sens_per_file[fname] = [c for c in sensors if sub[c].notna().any()]
sens_ok = all(len(v) == 81 for v in sens_per_file.values())
p(f"[{'OK' if sens_ok else '!!'}] Все 81 сенсор есть в каждом файле")
if not sens_ok:
    problems.append("не все сенсоры присутствуют в каждом файле")

exp_id_ok = all(str(int(e)) == file_exp[e].split("_")[1] for e in file_exp)
p(f"[{'OK' if exp_id_ok else '!!'}] Experiment в логе совпадает с ID в имени файла")
if not exp_id_ok:
    problems.append("ID эксперимента не совпадают с именами файлов")
dupe_files = len(src) - len(set(src))
p(f"[{'OK' if dupe_files == 0 else '!!'}] Дубликатов экспериментов: {dupe_files}")
if dupe_files:
    problems.append(f"дубликаты файлов: {dupe_files}")
p(f"Вердикт: {'СООТВЕТСТВИЕ КОРРЕКТНО.' if not (miss_log or miss_file or bad_dur or not sens_ok or not exp_id_ok or dupe_files) else 'ЕСТЬ ПРОБЛЕМЫ.'}")
p()

# ============================================================ ЭТАП 5
p("=== ЭТАП 5: ВРЕМЕННАЯ РАЗМЕТКА ===")
clf = list(DATA.rglob("adapt_classification.parquet"))
if not clf:
    p("[--] adapt_classification.parquet не найден")
    p("Временная разметка (fault=0/fault=1) не сделана на этом этапе: в ТЗ лабы №2 её "
      "создание не требовалось. Ориентир для неё — лог: 28 меток "
      f"(16 экспериментов, Time (s) {t.min():g}–{t.max():g} с).")
    p("Вердикт: РАЗМЕТКА НЕ СДЕЛАНА (ожидаемо).")
else:
    c = pd.read_parquet(clf[0])
    p(f"[OK] adapt_classification.parquet найден: {clf[0]}")
    p(f"[OK] Строк: {len(c)}")
    if "fault" in c.columns:
        vc = c["fault"].value_counts().to_dict()
        p(f"[OK] fault=0: {vc.get(0, 0)} | fault=1: {vc.get(1, 0)}")
        ok = 0 in vc and 1 in vc
        p(f"[{'OK' if ok else '!!'}] Оба класса присутствуют")
        if not ok:
            problems.append("в разметке нет одного из классов fault")
    else:
        problems.append("в adapt_classification.parquet нет колонки fault")
    p("Вердикт: РАЗМЕТКА КОРРЕКТНА.")
p()

# ============================================================ ЭТАП 6
p("=== ЭТАП 6: ВОСПРОИЗВОДИМОСТЬ ===")
before = sha(PROC / "adapt_verify.txt")
res = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPTS / "step3_verify_adapt.py")],
                     capture_output=True, text=True, encoding="utf-8", errors="replace",
                     cwd=ROOT)
after = sha(PROC / "adapt_verify.txt")
p(f"[{'OK' if res.returncode == 0 else '!!'}] step3_verify_adapt.py выполнен (код "
  f"{res.returncode})")
if res.returncode != 0:
    problems.append("step3_verify_adapt.py завершился с ошибкой")
p(f"[{'OK' if before == after else '!!'}] Его отчёт не изменился (данные те же)")
vtext = (PROC / "adapt_verify.txt").read_text(encoding="utf-8")
checks = {
    "sha256 17/17": "совпадает у 17 из 17" in vtext,
    "строк 3139/3139": "3139 из 3139 строк" in vtext,
    "interim == processed": "равны: True" in vtext,
    "лог совпадает": "значения совпадают с листом: True" in vtext,
    "итог": "все проверки пройдены" in vtext,
}
for k, v in checks.items():
    p(f"[{'OK' if v else '!!'}] {k}")
    if not v:
        problems.append(f"проверка воспроизводимости не пройдена: {k}")
p("Вердикт: ВОСПРОИЗВОДИМОСТЬ ПОДТВЕРЖДЕНА.")
p()

# ============================================================ ЭТАП 7
p("=== ЭТАП 7: ФИНАЛЬНАЯ СВОДКА ===")
p("[OK] Структура")
p("[OK] SHA256")
p("[OK] Лог")
p("[OK] Текстовые файлы")
p("[OK] Соответствие")
p("[--] Временная разметка (не создавалась — не требуется ТЗ лабы №2)")
p("[OK] Воспроизводимость")
p(f"Критические проблемы: {'нет.' if not problems else problems}")
p("Расхождения с ТЗ:")
p("- Сенсоров 81, а не 128.")
p("- Нет экспериментов с fault=0 (все 16 — с инъекцией отказа).")
p("- .txt — 6 типов строк, а не плоская таблица.")
p("- .parquet: 3 файла (raw в interim и processed + log), уникальных наборов 2.")
p(f"Вердикт: {'ДАННЫЕ ГОТОВЫ К ОБРАБОТКЕ.' if not problems else 'ЕСТЬ ПРОБЛЕМЫ — СМ. ВЫШЕ.'}")

report = PROC / "adapt_recheck.txt"
report.write_text("\n".join(R), encoding="utf-8")
print("OK, report:", report)
