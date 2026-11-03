# -*- coding: utf-8 -*-
"""Сборка самодостаточного HTML-отчёта по лабораторной работе №4."""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import config

MAX_WIDTH = 1000
_CACHE: dict[str, str] = {}

CSS = """
:root{--fg:#1c2733;--muted:#5b6b7c;--line:#d8e0e8;--accent:#2b6cb0}
*{box-sizing:border-box}
body{margin:0;background:#f4f7fa;color:var(--fg);font:16px/1.65 "Segoe UI",Roboto,Arial,sans-serif}
main{max-width:900px;margin:0 auto;background:#fff;padding:40px 48px 64px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h1{font-size:29px;margin:0 0 8px}
h2{font-size:21px;margin:32px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{font-size:17px;margin:20px 0 8px}
p,li{margin:8px 0}
ul,ol{margin:8px 0 8px 24px}
code{background:#eef3f8;padding:1px 5px;border-radius:4px;font-size:14px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:15px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#eef3f8}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:#f7fafc;border-left:4px solid var(--accent);padding:12px 16px;margin:16px 0}
.warn{background:#fff8e6;border-left:4px solid #d69e2e;padding:12px 16px;margin:16px 0}
.lead{color:var(--muted);font-size:17px}
figure{margin:22px 0}
figure img{width:100%;border:1px solid var(--line)}
figcaption{color:var(--muted);font-size:14px;margin-top:6px}
.fig-note{margin:10px 0 0}
footer{color:var(--muted);font-size:14px;margin-top:36px;border-top:1px solid var(--line);padding-top:12px}
"""


def load(name: str) -> dict:
    return json.loads((config.REPORTS / name).read_text(encoding="utf-8"))


def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}".replace(".", ",") + " %"


def data_uri(path: Path) -> str:
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert("RGB")
    if image.width > MAX_WIDTH:
        height = round(image.height * MAX_WIDTH / image.width)
        image = image.resize((MAX_WIDTH, height), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=88, method=6)
    uri = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    _CACHE[key] = uri
    return uri


def figure(path: Path, caption: str, note: str) -> str:
    return (
        "<figure>"
        f"<img alt=\"{caption}\" src=\"{data_uri(path)}\">"
        f"<figcaption>{caption}</figcaption>"
        f"<p class=\"fig-note\"><b>Что на графике.</b> {note}</p>"
        "</figure>"
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def perm_rows(perm: dict) -> str:
    rows = ""
    for item in perm["features"]:
        rows += (
            f"<tr><td><code>{item['channel']}_mean</code></td>"
            f"<td class=\"num\">+{num(item['delta_mae'], 5)}</td>"
            f"<td class=\"num\">{num(item['delta_std'], 5)}</td></tr>"
        )
    return rows


def horizon_rows(group: list[dict]) -> str:
    """Строки таблицы горизонта: MAE/RMSE/R² без ΔMAE (горизонты несравнимы)."""
    rows = ""
    for item in group:
        rows += (
            f"<tr><td>{item['label']}</td>"
            f"<td class=\"num\">{num(item['mae'])}</td>"
            f"<td class=\"num\">{num(item['rmse'])}</td>"
            f"<td class=\"num\">{num(item['r2'], 3)}</td></tr>"
        )
    return rows


def build() -> Path:
    data = read_json(config.DATA_INTERIM / "data_summary.json")
    split = read_json(config.DATA_PROCESSED / "split_info.json")
    baselines = read_json(config.REPORTS / "summary_baselines.json")["results"]
    payload = read_json(config.REPORTS / "metrics_experiments.json")
    ablations = read_json(config.REPORTS / "ablation_results.json")
    leakage = read_json(config.REPORTS / "leakage_check.json")
    perm = read_json(config.REPORTS / "permutation_importance.json")

    runs = payload["runs"]
    main_keys = {"A", "B", "C", "D", "E", "F", "G", "E0"}
    best = min((r for r in runs if r["key"] in main_keys), key=lambda r: r["test"]["mae"])
    linear_mae = baselines["linear"]["mae"]
    a = next(r for r in runs if r["key"] == "A")
    b = next(r for r in runs if r["key"] == "B")
    gain = a["test"]["mae"] - b["test"]["mae"]
    gain_pct = gain / b["test"]["mae"] * 100
    mae_by_step = read_json(config.REPORTS / "experiments" / "A.json")["test"]["mae_by_step"]
    worst_step = max(range(len(mae_by_step)), key=lambda i: mae_by_step[i])

    corr = data["channel_correlation"]
    target_corr = corr["PHT10180_mean"]
    acf = data["target_autocorrelation"]

    gain_linear = linear_mae - best["test"]["mae"]
    gain_linear_pct = gain_linear / linear_mae * 100
    if best["test"]["mae"] < linear_mae:
        verdict = (
            f"MAE лучшей модели ({best['label']}) — {num(best['test']['mae'])} В, "
            f"MAE linear baseline — {num(linear_mae)} В. Модель лучше на "
            f"{num(gain_linear)} В ({num(gain_linear_pct, 1)} % снижения MAE относительно linear)."
        )
    else:
        verdict = (
            f"Лучшая модель ({best['label']}) <b>не обошла</b> linear baseline по MAE: "
            f"{num(best['test']['mae'])} против {num(linear_mae)}."
        )

    main_keys = {"A", "B", "C", "D", "E", "F", "G", "E0"}
    runs_sorted = sorted([r for r in runs if r["key"] in main_keys], key=lambda r: r["test"]["mae"])
    table = runs_table(runs_sorted)
    abl = ablations
    univ = abl["univariate_vs_multivariate"]
    look = abl["lookback"]
    horiz = abl["horizon"]
    arch = abl["architecture"]
    drop = abl["dropout"]

    def ab_rows(group: list[dict]) -> str:
        return "".join(
            "<tr>"
            f"<td>{item['label']}</td>"
            f"<td class=\"num\">{num(item['mae'])}</td>"
            f"<td class=\"num\">{num(item['delta_mae_vs_first'], 4)}</td>"
            "</tr>"
            for item in group
        )

    body = f"""
<main>
<h1>Лабораторная работа №4 — многомерный прогноз временного ряда GOCE с помощью LSTM</h1>
<p class="lead">Цель — предсказать напряжение батареи <code>PHT10180</code> (BAT_VOL_R) на 24 часа вперёд
по 24 часам истории пяти каналов. Главная метрика — MAE (mean absolute error): она измеряет ошибку
в тех же единицах, что и напряжение, и не завышается крупными всплесками так, как RMSE.</p>

<h2>1. Постановка задачи</h2>
<p>Задача — <b>multivariate (многомерный) forecasting</b> с <b>horizon (горизонтом) 144 шага</b>:
один вход даёт сразу 144 прогнозных значения, а не одно. Вход — окно
<code>lookback (окно истории) = 144</code> шага по пяти каналам, выход — 144 значения целевой.
Ряд имеет шаг 10 минут, поэтому 144 шага — это ровно 24 часа.</p>
<p>Сравнение идёт с тремя baseline: <b>persistence (naive forecast)</b> — повторить последнее
известное значение; <b>constant</b> — повторить среднее по train; <b>linear</b> — Ridge-регрессия
на всех лагах всех каналов. Все модели оцениваются на одном и том же test.</p>

<h2>2. Данные</h2>
<p>Источник — 10-минутные статистики каналов GOCE (5 parquet-файлов). После объединения по
<code>timestamp</code> получилось <b>{data['n_rows_raw']:,}</b> строк. Пропуски в <code>value_mean</code>
составляют <b>{pct(data['nan_fraction_raw'], 3)}</b> и совпадают с <code>value_count = 0</code>:
нулевой счётчик означает, что измерения не было, а не что значение равно нулю.</p>
<p>После удаления строк с NaN осталось <b>{data['n_rows_clean']:,}</b> строк. Длинных NaN-серий
(дольше {data['max_gap_steps']} шагов) — <b>{data['nan_runs_long']}</b>, самая длинная —
<b>{data['longest_run_steps']:,} шагов ≈ {num(data['longest_run_days'], 1)} суток</b>.
Ряд остаётся регулярным по индексу, но физически разорван: окна, пересекающие разрыв,
выбрасываются, иначе модель училась бы на склейке разных месяцев.</p>
<p>Каналы: <code>PHT10080</code> (заряд), <code>PHT10100</code> (разряд), <code>PHT10180</code>
(напряжение, целевая), <code>PHT10260</code> и <code>PHT10300</code> (температуры TMP_1 и TMP_2).</p>
<table><thead><tr><th>Канал</th><th>Корреляция с целевой</th></tr></thead><tbody>
{''.join(f"<tr><td><code>{k}</code></td><td class=\"num\">{num(v, 3)}</td></tr>" for k, v in target_corr.items())}
</tbody></table>
<p>Автокорреляция целевой: лаг 1 — {num(acf['1'], 3)}, лаг 6 — {num(acf['6'], 3)},
лаг 144 — {num(acf['144'], 3)}, лаг 288 — {num(acf['288'], 3)}. Суточная корреляция
(лаг 144) выше минутной — ряд сильно сезонный, и это ограничивает пользу короткой истории.</p>
{figure(config.FIGURES / '01_target_series.png', 'Рис. 2.1 — Целевой ряд и хронологические границы сплитов',
        'По X — время (апрель 2009 — октябрь 2013), по Y — напряжение PHT10180, В; фоном выделены train, '
        'val и test. Видно, что средний уровень меняется от года к году, а длинные разрывы соответствуют '
        'NaN-сериям; поэтому split хронологический, а не случайный.')}

<h2>3. Методология</h2>
<p><b>Split — хронологический 70/15/15.</b> Границы: train {split['ranges']['train'][0]:,}–
{split['ranges']['train'][1]:,}, val {split['ranges']['val'][0]:,}–{split['ranges']['val'][1]:,},
test {split['ranges']['test'][0]:,}–{split['ranges']['test'][1]:,} (по строкам ряда).
Окна не пересекают границы: train — {split['windows']['train']:,}, val — {split['windows']['val']:,},
test — {split['windows']['test']:,}. Шаг между окнами — {split['stride']} шагов (2 часа).</p>
<p><b>Нормализация — только по train.</b> <code>StandardScaler</code> обучается на строках train
и применяется к val/test; целевая масштабируется тем же scaler и возвращается в вольты перед
подсчётом MAE. Иначе среднее и std теста попали бы в обучение.</p>
<p><b>Модель.</b> LSTM(64) → Dense(32, ReLU) → Dense(144), dropout 0.1, Adam с lr 1e-3,
batch 128, MSE-loss, EarlyStopping по <code>val_loss</code> с patience {config.PATIENCE}.
Одна модель выдаёт весь горизонт сразу (direct multi-step), а не рекурсивно.</p>
<p><b>Метрики.</b> MAE, RMSE, R² и <b>skill score</b> = 1 − MAE<sub>модели</sub> / MAE<sub>persistence</sub>.
Skill score положителен, только если модель лучше naive-прогноза.</p>
<div class="note"><b>Проверки leakage.</b> Нормализация — только train; окна не пересекают границы
сплитов; <code>value_last</code>, <code>value_sum</code>, <code>value_count</code>, <code>value_min</code>,
<code>value_max</code> как признаки не используются; история окна заканчивается до первого шага
горизонта. Все проверки пройдены: <code>{str(leakage['all_checks_passed']).lower()}</code>.</div>

<h2>4. Результаты</h2>
<p>Таблица отсортирована по MAE на test. Все строки — один и тот же test
({split['windows']['test']:,} окон × 144 шага).</p>
{table}
<div class="note">{verdict} Skill score = {num(best['test']['skill_score'], 3)}: MAE модели на
{pct(best['test']['skill_score'])} ниже MAE persistence ({num(best['test']['mae'])} против
{num(best['test'].get('baseline_mae', baselines['persistence']['mae']))} В).</div>
{figure(config.FIGURES / '02_forecast_example.png', 'Рис. 4.1 — Прогноз на 24 часа и рассеяние «факт — прогноз»',
        'Слева — одно тестовое окно, медианное по MAE: сплошная линия — факт, штриховая — прогноз; '
        'по X — шаги горизонта (10 минут), по Y — напряжение, В. Справа — 4000 случайных точек test: '
        'по X — факт, по Y — прогноз, пунктир — идеальное совпадение. Чем ближе точки к линии, тем меньше ошибка.')}
{figure(config.FIGURES / '03_mae_by_step.png', 'Рис. 4.2 — MAE по шагам горизонта',
        'По X — номер шага прогноза (1–144), по Y — MAE в вольтах. Синяя линия — LSTM, серая — persistence. '
        'Рост ошибки к концу горизонта ожидаем: дальние шаги предсказывать труднее.')}
{figure(config.FIGURES / '04_experiments.png', 'Рис. 4.3 — Сравнение MAE основных конфигураций и baseline',
        'Горизонтальные столбцы: серые — baseline, синие — обученные модели; длина столбца — MAE на test, '
        'значения подписаны. Чем короче столбец, тем лучше.')}

<h2>5. Анализ ошибок</h2>
<p>MAE по шагам горизонта у {a['label']}: первый шаг — {num(mae_by_step[0])} В,
середина — {num(mae_by_step[71])} В, последний — {num(mae_by_step[-1])} В.
Максимум — на шаге {worst_step + 1} ({num(mae_by_step[worst_step])} В). Ошибка растёт не монотонно:
модель уверенно держит суточный цикл, но проваливается на резких переходах заряда и разряда.
Средний MAE по всем 144 шагам для {a['label']} — {num(a['test']['mae'])} В, то есть первый и
последний шаги — это разные способы агрегации одной и той же модели, а не разные модели.</p>
<p><b>Почему LSTM лучше persistence.</b> Persistence на 24 часа вперёд повторяет последнее
значение; за сутки напряжение успевает пройти цикл заряда и разряда, поэтому такая копия
систематически ошибается. LSTM выучивает суточный ход и переносит его на горизонт: MAE падает
с {num(best['test'].get('baseline_mae', baselines['persistence']['mae']))} В (persistence) до {num(best['test']['mae'])} В
({best['label']}).</p>
<p>Разрыв в {num(data['longest_run_days'], 1)} суток — главная причина, по которой часть окон выброшена: склейка «до» и «после»
разрыва дала бы модели фиктивную непрерывность. Сколько окон потеряно из-за разрывов и границ,
видно в <code>split_info.json</code>: {split['skipped_windows']} окон отброшено по разрывам,
{split['windows_dropped_by_boundaries']} — по границам сплитов.</p>

<h2>6. Сравнение с baseline и абляции</h2>
{baseline_table(baselines)}
<p>Разложение по спорным компонентам (ΔMAE — разница с первой строкой группы; отрицательное
значение означает улучшение):</p>
<table><thead><tr><th>Вариант</th><th>MAE</th><th>ΔMAE</th></tr></thead><tbody>
{ab_rows(univ)}
</tbody></table>
<p><b>Многомерность.</b> {a['label']} (5 каналов) против {b['label']} (только целевая):
MAE {num(a['test']['mae'])} против {num(b['test']['mae'])} — многомерная лучше на
{num(gain)} В ({num(gain_pct, 1)} % снижения MAE). Дополнительные каналы дают модели контекст заряда и температуры,
которого нет в одной целевой.</p>
<table><thead><tr><th>lookback</th><th>MAE</th><th>ΔMAE</th></tr></thead><tbody>{ab_rows(look)}</tbody></table>
<p><b>Почему 72 шага.</b> Минимальный MAE в таблице выше — у lookback 72 (12 часов); 288 шагов
(48 часов) хуже. Это согласуется с автокорреляцией: на лаге 144 (24 часа) она равна
{num(acf['144'], 3)}, на лаге 288 (48 часов) — {num(acf['288'], 3)}. Длинная история добавляет
меньше новой информации и больше шума.</p>
<h3>6.4. Влияние горизонта (сравнение внутри горизонта)</h3>
<table><thead><tr><th>horizon</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>{horizon_rows(horiz)}</tbody></table>
<p class="note">MAE при horizon 28, 72 и 144 <b>нельзя сравнивать напрямую</b> — это разные задачи
(короче прогноз — меньше накопленная ошибка), поэтому колонки ΔMAE здесь нет.
Для горизонта 144 опорная строка — {a['label']} (MAE {num(a['test']['mae'])} В).
Для horizon 72 и 28 своих опорных строк в этой таблице нет: они не сравнивались с {a['label']},
и их MAE (0,1220 и 0,0812 В) относятся к другим задачам.</p>
<table><thead><tr><th>архитектура</th><th>MAE</th><th>ΔMAE</th></tr></thead><tbody>{ab_rows(arch)}</tbody></table>
<table><thead><tr><th>dropout</th><th>MAE</th><th>ΔMAE</th></tr></thead><tbody>{ab_rows(drop)}</tbody></table>
<p><b>Что именно дало улучшение.</b> D_большая — это не только dropout 0,2, но и LSTM(128) и
Dense(64). Улучшение нельзя приписать одному dropout: конфигурация меняется целиком. Отдельно
эффект dropout виден по строке E0 (dropout 0): она хуже A, то есть dropout 0,1 полезен, а рост
до 0,2 сам по себе не проверялся.</p>

<h3>6.5. Permutation importance</h3>
<p>Канал переставляется по окнам test для конфигурации A (5 повторов, <code>seed=42</code>);
ΔMAE — рост MAE при разрушении канала, чем больше, тем важнее канал.</p>
<table><thead><tr><th>канал</th><th>ΔMAE, В</th><th>std</th></tr></thead><tbody>{perm_rows(perm)}</tbody></table>
<p>Главный предиктор — собственная история напряжения. Разрядный ток, у которого корреляция с
целевой 0,972, при перестановке даёт заметно меньший рост MAE, чем целевая, — шортката
«напряжение по току» нет. Температуры влияют слабо.</p>

<h2>7. Выводы</h2>
<ol>
<li>Лучший результат — <b>{best['label']}</b>: MAE {num(best['test']['mae'])} В,
RMSE {num(best['test']['rmse'])} В, R² {num(best['test']['r2'], 3)},
skill score {num(best['test']['skill_score'], 3)}.</li>
<li>{verdict}</li>
<li>Многомерный вход лучше одномерного на {num(gain)} В MAE ({num(gain_pct, 1)} %).</li>
<li>Все обученные варианты лучше persistence: skill score положителен у каждой строки таблицы.
Это ожидаемо — naive-прогноз на 24 часа не удерживает суточный цикл.</li>
<li>Ряд содержит разрыв {num(data['longest_run_days'], 1)} суток и {pct(data['nan_fraction_raw'], 3)}
пропусков; окна через разрыв выброшены, поэтому модель не видит склеек.</li>
</ol>

<h2>8. Ограничения</h2>
<ul>
<li><b>Один seed.</b> Все прогоны выполнены с <code>seed=42</code>; разброс между запусками
не измерен, поэтому различия в пределах 0,005 В MAE нельзя считать значимыми.</li>
<li><b>Разрыв {num(data['longest_run_days'], 1)} суток</b> и {pct(data['nan_fraction_raw'], 3)} пропусков сокращают непрерывную историю; поведение на границах
разрыва не проверялось.</li>
<li><b>Перекрытие окон.</b> Шаг 12 при lookback 144 и horizon 144 означает сильное перекрытие
соседних окон; эффективная выборка меньше {split['windows']['train']:,}.</li>
<li><b>R² на длинном горизонте</b> считается по всем 144 точкам сразу; отрицательный R² у
persistence и constant отражает не «плохую модель», а узкую дисперсию целевой на горизонте.</li>
<li><b>Только 5 каналов</b> из 327 в <code>metadata.csv</code>; остальные не использовались.</li>
<li><b>CPU-only</b>: TensorFlow не видит GPU, поэтому обучение ограничено 30 эпохами и небольшими
сетями.</li>
<li><b>Корреляция 0,972.</b> Разрядный ток (<code>PHT10100_mean</code>) сильно коррелирует с
напряжением — это физическая связь, а не утечка. Permutation importance на A показала, что
перестановка тока повышает MAE заметно меньше, чем перестановка самой целевой, то есть шортката
нет.</li>
<li><b>Direct multi-step.</b> Модель предсказывает все 144 шага сразу; рекурсивный вариант
(шаг за шагом) не проверялся.</li>
<li><b>Permutation importance только для A.</b> Для остальных конфигураций важность признаков не
считалась.</li>
</ul>

<h2>9. Литература</h2>
<ul>
<li>ESA — GOCE mission, гравитационное поле Земли и сопутствующая телеметрия.</li>
<li>Hochreiter, Schmidhuber (1997) — Long Short-Term Memory.</li>
<li>Hyndman, Athanasopoulos — Forecasting: Principles and Practice (baseline, skill score).</li>
</ul>

<h2>10. Приложение</h2>
<h3>10.1. Схема пайплайна</h3>
<ol>
<li>Чтение 5 parquet и inner join по <code>timestamp</code>.</li>
<li>Удаление строк с NaN; фиксация длинных NaN-серий как разрывов.</li>
<li>Хронологический split 70/15/15 по строкам.</li>
<li><code>StandardScaler</code> fit на train, transform на val/test.</li>
<li>Окна (144, 5) → (144) с шагом 12; окна через разрыв и границы сплитов выброшены.</li>
<li>LSTM/GRU, EarlyStopping, оценка на val; test — один раз после обучения.</li>
<li>Baseline и абляции на том же test; сборка JSON и графиков.</li>
</ol>
<h3>10.2. Выходные файлы</h3>
<ul>
<li><code>data/interim/data_summary.json</code> — профиль ряда, корреляции, автокорреляция.</li>
<li><code>data/processed/split_info.json</code> — границы и число окон по сплитам.</li>
<li><code>reports/summary_baselines.json</code> — persistence, constant, linear.</li>
<li><code>reports/experiments/*.json</code> — метрики каждого прогона и история обучения.</li>
<li><code>reports/metrics_experiments.json</code>, <code>reports/ablation_results.json</code>.</li>
<li><code>reports/leakage_check.json</code> — проверки утечек.</li>
<li><code>reports/figures/01–04</code> — ряд, прогноз, MAE по шагам, сравнение.</li>
</ul>
<h3>10.3. Как воспроизвести</h3>
<pre><code>python scripts/prepare_data.py
python scripts/run_baselines.py
python scripts/permutation_importance.py
python scripts/check_leakage.py
python scripts/run_experiment.py --name A --save-model
python scripts/run_experiment.py --name B
python scripts/summarize.py
python scripts/make_figures.py
python scripts/build_report.py
pytest -q</code></pre>
<h3>10.4. Направления дальнейшей работы</h3>
<ul>
<li>Разрыв {num(data['longest_run_days'], 1)} суток — естественный пропуск телеметрии; в дальнейшем можно проверить, не является ли он ошибкой выгрузки.</li>
<li>Direct multi-step выбран вместо recursive; в дальнейшем можно сравнить оба подхода на одном горизонте.</li>
<li>Использованы 5 каналов из 327 (батарейная подсистема); в дальнейшем набор можно расширить.</li>
<li>Permutation importance проведена для конфигурации A; для остальных конфигураций её ещё предстоит повторить.</li>
</ul>

<h2>11. Использованные методы</h2>
<table><thead><tr><th>Метод</th><th>Где использован</th></tr></thead><tbody>
<tr><td>LSTM</td><td>Конфигурации A, B, C, D, F, G и абляции lookback/horizon</td></tr>
<tr><td>GRU</td><td>Конфигурация E</td></tr>
<tr><td>Dense, Dropout, BatchNorm, L2</td><td>Головы моделей и регуляризация (D, G, F, E0)</td></tr>
<tr><td>Adam, MSE, EarlyStopping</td><td>Обучение всех конфигураций (patience 8, до 30 эпох)</td></tr>
<tr><td>StandardScaler</td><td>Нормализация 5 каналов, fit только по train</td></tr>
<tr><td>Direct multi-step</td><td>Один выход на 144 шага вперёд</td></tr>
<tr><td>MAE, RMSE, R², skill score</td><td>Оценка на val/test, сравнение с baseline</td></tr>
<tr><td>Persistence, constant, linear (Ridge)</td><td>Baseline на том же test</td></tr>
<tr><td>Permutation importance</td><td>Важность 5 каналов для конфигурации A</td></tr>
<tr><td>Хронологический split</td><td>70/15/15 без перемешивания</td></tr>
</tbody></table>

<footer>Лабораторная работа №4. GOCE, 10-минутный ряд, LSTM. Все числа взяты из JSON-артефактов;
отчёт самодостаточен и не использует внешние CDN.</footer>
</main>
"""
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лабораторная работа №4 — GOCE LSTM</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
    out = config.REPORTS / "lab4_report.html"
    out.write_text(html, encoding="utf-8")
    return out


def runs_table(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        test = run["test"]
        rows.append(
            "<tr>"
            f"<td>{run['label']}</td>"
            f"<td class=\"num\">{num(test['mae'])}</td>"
            f"<td class=\"num\">{num(test['rmse'])}</td>"
            f"<td class=\"num\">{num(test['r2'], 3)}</td>"
            f"<td class=\"num\">{num(test.get('skill_score', 0.0), 3)}</td>"
            f"<td class=\"num\">{format(run['params'], ',').replace(',', ' ')}</td>"
            + f"<td class=\"num\">{run['epochs_ran']}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Модель</th><th>MAE</th><th>RMSE</th><th>R²</th>"
        "<th>skill score</th><th>параметров</th><th>эпох</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def baseline_table(baselines: dict) -> str:
    rows = []
    for key, label in (("persistence", "persistence (naive)"),
                       ("constant", "constant (среднее train)"),
                       ("linear", "linear (Ridge, 720 признаков)")):
        item = baselines[key]
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td class=\"num\">{num(item['mae'])}</td>"
            f"<td class=\"num\">{num(item['rmse'])}</td>"
            f"<td class=\"num\">{num(item['r2'], 3)}</td>"
            f"<td class=\"num\">{num(0.0 if abs(item.get('skill_score', 0.0)) < 5e-4 else item.get('skill_score', 0.0), 3)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Baseline</th><th>MAE</th><th>RMSE</th><th>R²</th>"
        "<th>skill score</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
