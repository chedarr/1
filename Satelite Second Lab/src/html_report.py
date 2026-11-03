# -*- coding: utf-8 -*-
"""Сборка единого HTML-отчёта лабораторной №2 (офлайн, изображения встроены).

Отчёт состоит из одиннадцати разделов: титул, постановка задачи, данные,
методология, результаты, сравнение, анализ, ограничения, выводы, приложение и
использованные методы. Перед каждым рисунком стоит абзац «Что на графике»
(или «Что на схеме»), подпись содержит номер вида «Рис. 5.2».
Изображения встроены как data-URI, внешних ссылок и скриптов нет.
"""
from __future__ import annotations

import base64
from pathlib import Path

from src import config, report_text

_CSS = """
:root { --fg:#1c2733; --muted:#5b6b7c; --line:#d8e0e8; --accent:#2b6cb0; --bg:#ffffff; }
* { box-sizing: border-box; }
body { margin:0; background:#f4f7fa; color:var(--fg);
       font: 16px/1.6 "Segoe UI", Roboto, Arial, sans-serif; }
main { max-width: 1020px; margin: 0 auto; background: var(--bg); padding: 40px 48px 64px;
       box-shadow: 0 1px 3px rgba(0,0,0,.08); }
h1 { font-size: 30px; margin: 0 0 6px; }
h2 { font-size: 21px; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h3 { font-size: 17px; margin: 22px 0 8px; }
.sub { color: var(--muted); margin: 0 0 26px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 6px; font-size: 15px; }
th, td { border: 1px solid var(--line); padding: 7px 10px; text-align: left; }
th { background: #eef3f8; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #1e7a45; font-weight: 600; }
.warn { color: #9a6400; font-weight: 600; }
.bad { color: #b3261e; font-weight: 600; }
figure { margin: 18px 0; }
figure img { width: 100%; border: 1px solid var(--line); border-radius: 6px; }
figcaption { color: var(--muted); font-size: 14px; margin-top: 6px; }
.fig-note { margin: 14px 0 0; }
.legend { color: var(--muted); font-size: 14px; margin: 2px 0 10px; }
.note { background: #f7fafc; border-left: 4px solid var(--accent); padding: 12px 16px; margin: 16px 0; }
ul { margin: 8px 0 8px 22px; }
code { background:#eef3f8; padding:1px 5px; border-radius:4px; font-size:14px; }
footer { color: var(--muted); font-size: 14px; margin-top: 40px; border-top: 1px solid var(--line); padding-top: 12px; }
tr.winner td { font-weight: 600; background: #eef7f0; }
"""

# Реестр рисунков: ключ -> (номер подписи, имя файла, подпись по умолчанию).
_FIGURES: dict[str, tuple[str, str, str]] = {
    "class_balance": ("3.1", "01_class_balance.png",
                      "Распределение классов в обучении и тесте"),
    "example_window": ("3.2", "02_example_window.png",
                       "Пример окна с отказом: три самых изменчивых сенсора с различимыми кривыми"),
    "confusion": ("5.1", "03_confusion_matrix.png",
                  "confusion matrix (матрица ошибок) на test"),
    "pr_curve": ("5.2", "04_pr_curve.png",
                 "precision, recall и F1 в зависимости от threshold"),
    "roc": ("5.3", "05_roc_curve.png", "ROC-кривая на test"),
    "training": ("5.4", "06_training_curves.png",
                 "Кривые обучения: loss и accuracy по эпохам"),
    "baseline_comparison": ("5.5", "07_baseline_comparison.png",
                            "Сравнение итоговой модели с простыми моделями на тестовой выборке"),
    "model_comparison": ("6.1", "08_model_comparison.png",
                         "Метрики всех моделей на тестовой выборке"),
    "diag_frac_positive": ("7.1", "09_diag_f1_vs_frac_positive.png",
                           "F1 записи против доли окон с отказом"),
    "diag_dist_to_train": ("7.2", "10_diag_f1_vs_dist_to_train.png",
                           "F1 записи против расстояния до обучающих записей"),
    "diag_first_fault": ("7.3", "11_diag_f1_vs_first_fault.png",
                         "F1 записи против положения первого отказа"),
    "loeo_folds": ("7.4", "12_loeo_folds.png",
                   "F1 по каждому исключённому эксперименту (LOEO)"),
    "pipeline_schema": ("10.1", "13_pipeline_schema.png", "Схема пайплайна обработки"),
    "model_architecture": ("10.2", "14_model_architecture.png", "Схема архитектуры сети"),
}


def _img(path: Path | str | None, number: str, caption: str) -> str:
    """Встраивает изображение в HTML как data-URI (работает без сети)."""
    if not path or not Path(path).exists():
        return f"<p class='warn'>нет файла графика: Рис. {number} — {caption}</p>"
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (
        f"<figure><img alt='Рис. {number} — {caption}' "
        f"src='data:image/png;base64,{b64}'>"
        f"<figcaption>Рис. {number} — {caption}</figcaption></figure>"
    )


def _figure(figures: dict, key: str, caption: str | None = None) -> str:
    """Абзац «Что на графике» и рисунок с номером подписи."""
    number, _filename, default_caption = _FIGURES[key]
    path = figures.get(key)
    return (
        f"{report_text.figure_note(key)}\n"
        f"{_img(path, number, caption or default_caption)}"
    )


def _cell(value, good: float, mid: float, digits: int = 4) -> str:
    """Ячейка с числом и цветовой разметкой (зелёный / оранжевый / красный)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "<td class='num'>—</td>"
    cls = "ok" if number >= good else ("warn" if number >= mid else "bad")
    return f"<td class='num'><span class='{cls}'>{number:.{digits}f}</span></td>"


def _simple_cell(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "<td class='num'>—</td>"
    return f"<td class='num'>{number:.{digits}f}</td>"


# Словесное описание сети приходит из модуля экспериментов по-русски; при выводе
# в отчёт принятые термины записываются по-английски.
_ARCHITECTURE_TERMS = (
    ("прореживание", "Dropout"),
    ("нормализация слоёв", "BatchNorm"),
    ("нормализация", "BatchNorm"),
    ("веса классов", "class weights"),
)


def _architecture_en(text: str) -> str:
    """Приводит описание архитектуры к принятым терминам (Dropout, BatchNorm)."""
    for ru, en in _ARCHITECTURE_TERMS:
        text = text.replace(ru, en)
    return text


def _metrics_rows(metrics: dict) -> str:
    order = [
        ("accuracy", "accuracy"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    rows = ""
    for key, label in order:
        if key in metrics:
            rows += (
                f"<tr><td>{label}</td>"
                f"{_cell(metrics[key], 0.85, 0.7)}</tr>"
            )
    rows += f"<tr><td>threshold</td>{_simple_cell(metrics.get('порог'))}</tr>"
    total = metrics.get("объектов")
    if total is not None:
        rows += f"<tr><td>n_windows (test)</td><td class='num'>{int(total)}</td></tr>"
    cm = metrics
    if {"верно_норма", "ложная_тревога", "пропуск_отказа", "верно_отказ"} <= set(cm):
        rows += (
            "<tr><td>Верно опознана норма</td>"
            f"<td class='num'>{cm['верно_норма']}</td></tr>"
            "<tr><td>Ложная тревога</td>"
            f"<td class='num'>{cm['ложная_тревога']}</td></tr>"
            "<tr><td>Пропущенный отказ</td>"
            f"<td class='num'>{cm['пропуск_отказа']}</td></tr>"
            "<tr><td>Верно опознан отказ</td>"
            f"<td class='num'>{cm['верно_отказ']}</td></tr>"
        )
    return rows


def _comparison_rows(comparison: dict, names: list[str]) -> str:
    rows = ""
    for name in names:
        item = comparison.get(name)
        if not item:
            continue
        m = item.get("metrics", item)
        rows += (
            f"<tr><td>{name}</td>"
            f"{_cell(m.get('accuracy'), 0.85, 0.7)}"
            f"{_cell(m.get('precision'), 0.85, 0.7)}"
            f"{_cell(m.get('recall'), 0.85, 0.7)}"
            f"{_cell(m.get('f1'), 0.85, 0.7)}</tr>"
        )
    return rows


def _diagnostics_corr_rows(diagnostics: dict) -> str:
    """Таблица связи качества фолда с характеристиками записи."""
    labels = [
        ("frac_positive", "frac_positive (доля окон с отказом)"),
        ("first_fault_pos_frac", "first_fault_pos_frac (положение первого отказа)"),
        ("mean_sensor_std", "mean_sensor_std (разброс сенсоров)"),
        ("dist_to_train", "dist_to_train (расстояние до обучающих записей, с самовкладом)"),
        ("dist_to_train_looe", "dist_to_train_looe (то же без самовклада)"),
    ]
    all16 = diagnostics.get("корреляции", {}).get("все_16", {}).get("f1", {})
    rows = ""
    for key, label in labels:
        stats = all16.get(key, {})
        coef = stats.get("коэффициент")
        p_value = stats.get("p_value")
        if coef is None:
            rows += f"<tr><td>{label}</td><td class='num'>—</td><td class='num'>—</td></tr>"
            continue
        cls = "ok" if abs(coef) < 0.5 else "bad"
        p_text = "—" if p_value is None else f"{p_value:.3f}"
        rows += (
            f"<tr><td>{label}</td>"
            f"<td class='num'><span class='{cls}'>{coef:.3f}</span></td>"
            f"<td class='num'>{p_text}</td></tr>"
        )
    return rows


def build_report(ctx: dict) -> str:
    """Собирает HTML-отчёт из подготовленного контекста."""
    figures = ctx.get("figures", {})
    checks = ctx.get("checks", [])
    train = ctx.get("train", {})
    test = ctx.get("test", {})
    final = ctx.get("mlp_metrics", {})
    before = ctx.get("before", {})
    comparison = ctx.get("comparison", {})
    diagnostics = ctx.get("diagnostics") or {}
    improvement = ctx.get("improvement") or {}
    loeo = ctx.get("loeo") or {}
    cv = ctx.get("cv", {})
    seed = ctx.get("seed", config.SEED)
    architecture = ctx.get("architecture", "вход → 128 → 64 → 32 → 1")

    _cls = {"OK": "ok", "ВНИМАНИЕ": "warn", "ПРОБЛЕМА": "bad"}
    check_rows = "".join(
        f"<tr><td>{c['проверка']}</td>"
        f"<td class='{_cls.get(c['результат'], 'bad')}'>{c['результат']}</td>"
        f"<td>{c['детали']}</td></tr>"
        for c in checks
    )

    exp_rows = "".join(
        f"<tr><td>{m['эксперимент']}</td>"
        f"<td class='num'>{m['объектов']}</td>"
        f"{_cell(m['accuracy'], 0.85, 0.7, 3)}"
        f"{_cell(m['precision'], 0.85, 0.7, 3)}"
        f"{_cell(m['recall'], 0.85, 0.7, 3)}"
        f"{_cell(m['f1'], 0.85, 0.7, 3)}</tr>"
        for m in ctx.get("test_by_experiment", [])
    )

    baseline_names = [
        "Итоговая модель",
        "Логистическая регрессия",
        "Постоянный ответ «отказ»",
        "Постоянный ответ «норма»",
    ]
    base_rows = _comparison_rows(comparison, baseline_names)

    summary_names = list(comparison.keys()) or baseline_names
    summary_rows = _comparison_rows(comparison, summary_names)

    # 10.4. Проверенные конфигурации.
    rows_imp = improvement.get("rows") or improvement.get("вклад_методов") or []
    winner_name = improvement.get("winner") or improvement.get("победитель")
    cfg_rows = ""
    for r in rows_imp:
        if "ошибка" in r:
            cfg_rows += (
                f"<tr><td>{r['имя']}</td><td>{r.get('дополнения', '')}</td>"
                f"<td colspan='8' class='bad'>не удалось: {r['ошибка']}</td></tr>"
            )
            continue
        mark = " class='winner'" if r["имя"] == winner_name else ""
        cfg_rows += (
            f"<tr{mark}><td>{r['имя']}</td><td>{r.get('дополнения', '')}</td>"
            f"<td class='num'>{r.get('окон_обучения', '—')}</td>"
            f"<td class='num'>{r.get('признаков', '—')}</td>"
            f"{_simple_cell(r.get('валидация_f1_0.5'), 3)}"
            f"{_simple_cell(r.get('валидация_f1'), 3)}"
            f"{_simple_cell(r.get('порог_f1'), 3)}"
            f"{_simple_cell(r.get('тест_f1'), 3)}"
            f"{_simple_cell(r.get('тест_полнота'), 3)}"
            f"<td class='num'>{r.get('прирост_f1_на_тесте', 0.0):+.3f}</td></tr>"
        )

    # 10.5. Разложение прироста.
    ladder = improvement.get("разложение_прироста") or improvement.get("ladder") or []
    ladder_rows = ""
    for step in ladder:
        m = step["метрики"]
        ladder_rows += (
            f"<tr><td>{step['этап']}</td>"
            f"{_simple_cell(m.get('accuracy'), 3)}"
            f"{_simple_cell(m.get('precision'), 3)}"
            f"{_simple_cell(m.get('recall'), 3)}"
            f"{_simple_cell(m.get('f1'), 3)}"
            f"{_simple_cell(m.get('порог', 0.5), 3)}</tr>"
        )

    verdicts = "".join(f"<li>{v}</li>" for v in ctx.get("verdicts", []))
    limits = "".join(f"<li>{v}</li>" for v in ctx.get("limitations", []))
    questions = "".join(f"<li>{v}</li>" for v in ctx.get("questions", []))
    loeo_notes = "".join(f"<p>{p}</p>" for p in ctx.get("loeo_notes", []))
    failed = "".join(f"<p>{p}</p>" for p in ctx.get("failed_methods", []))
    decomp = "".join(f"<p>{p}</p>" for p in ctx.get("decomposition", []))

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лабораторная работа №2 — Классификация состояния спутника (ADAPT)</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>Лабораторная работа №2 — Классификация состояния спутника (ADAPT)</h1>
<p class="sub">Датасет: ADAPT</p>
<table>
<tr><th>Поле</th><th>Значение</th></tr>
<tr><td>Дисциплина</td><td>Машинное обучение: классификация</td></tr>
<tr><td>Датасет</td><td>ADAPT</td></tr>
<tr><td>Инструменты</td><td>Python 3.13, Keras/TensorFlow, scikit-learn, pandas, matplotlib</td></tr>
<tr><td>Зерно</td><td>{seed}</td></tr>
</table>
<p><b>Задача.</b> По окну телеметрии спутника определить состояние системы —
норма или отказ. Это бинарная классификация; признаками служат значения
81 сенсора на 20 последовательных шагах измерения, то есть 1620 чисел на окно.
Разбиение выполняется строго по экспериментам, а качество измеряется набором
метрик на тестовой выборке (определения — в разделе 2).</p>

<h2>2. Постановка задачи</h2>
<table>
<tr><th>Параметр</th><th>Значение</th></tr>
<tr><td>Тип задачи</td><td>бинарная классификация</td></tr>
<tr><td>Целевая метка</td><td>норма (0) / отказ (1)</td></tr>
<tr><td>Признаки</td><td>1620 (81 сенсор × 20 шагов)</td></tr>
<tr><td>Разбиение</td><td>по экспериментам: обучение — {ctx.get('n_train_exp', 12)}, тест — {ctx.get('n_test_exp', 4)}</td></tr>
<tr><td>Метрики</td><td>accuracy (доля верных ответов), precision (точность по отказам), recall (полнота) и F1 (F-мера)</td></tr>
</table>
<p>Задача — бинарная классификация: каждому окну телеметрии приписывается
состояние «норма» или «отказ». Главной метрикой выбрана F1, потому что она
уравновешивает recall и precision, но приводится она только вместе с precision.
Причина в том, что постоянный ответ «отказ» даёт F1 =
{comparison.get('Постоянный ответ «отказ»', {}).get('f1', 0.0):.3f} при
precision = {comparison.get('Постоянный ответ «отказ»', {}).get('precision', 0.0):.3f}: без
рядом стоящего precision такое число выглядит приемлемым, хотя модель не
различает классы вовсе.</p>

<h2>3. Данные</h2>
<table>
<tr><th>Выборка</th><th class="num">Окон</th><th class="num">Норма</th><th class="num">Отказ</th><th class="num">Экспериментов</th></tr>
<tr><td>Обучение</td><td class="num">{train.get('окон', '—')}</td><td class="num">{train.get('норма', '—')}</td><td class="num">{train.get('отказ', '—')}</td><td class="num">{train.get('экспериментов', '—')}</td></tr>
<tr><td>Тест</td><td class="num">{test.get('окон', '—')}</td><td class="num">{test.get('норма', '—')}</td><td class="num">{test.get('отказ', '—')}</td><td class="num">{test.get('экспериментов', '—')}</td></tr>
</table>
{_figure(figures, 'class_balance')}
{_figure(figures, 'example_window')}
<p>Классы в обеих частях близки по размеру: в обучении
{train.get('отказ', '—')} окон отказа против {train.get('норма', '—')} окон нормы,
в тесте — {test.get('отказ', '—')} против {test.get('норма', '—')}. Небольшой
перекос сохраняется, поэтому accuracy остаётся содержательной, но её
следует читать вместе с recall и precision: при перекосе в сторону одного
класса accuracy легко завышается постоянным ответом.</p>

<h3>3.1. Проверки целостности данных</h3>
<table>
<tr><th>Проверка</th><th>Результат</th><th>Детали</th></tr>
{check_rows}
</table>
{report_text.table_legend()}

<h2>4. Методология</h2>
<h3>4.1. Теоретическая основа</h3>
<p><b>MLP.</b> Модель — сеть с прямой передачей сигнала:
входной слой, скрытые слои (Dense) и выходной слой. Каждый нейрон вычисляет
взвешенную сумму входов и применяет функцию активации; в скрытых слоях
используется ReLU, на выходе — sigmoid, дающая вероятность отказа.</p>
<p><b>Обратное распространение ошибки.</b> Обучение состоит из двух проходов:
прямой проход вычисляет выход сети, обратный — градиент ошибки по весам через
цепное правило. Градиент распространяется от выходного слоя к входному.</p>
<p><b>Функция потерь.</b> Функцией потерь служит cross-entropy (binary): она
сравнивает предсказанную вероятность с истинной меткой и штрафует уверенные
ошибки сильнее.</p>
<p><b>Оптимизатор Adam.</b> Веса обновляются градиентным спуском с моментом и
адаптивным шагом: w(t+1) = w(t) − α · ∂E/∂w, где α — скорость обучения,
∂E/∂w — градиент ошибки по весу.</p>
<p><b>Стандартизация по train.</b> Признаки приводятся к единому масштабу по
формуле x' = (x − x̄) / σ, где x̄ — среднее, σ — стандартное отклонение. Среднее и
разброс считаются только по train, чтобы test не влиял на подготовку.</p>
<p><b>Регуляризация.</b> Против переобучения используются Dropout
(случайное отключение нейронов при обучении), BatchNorm и early stopping
(прекращение обучения при отсутствии улучшения на val).</p>

<h3>4.2. Признаки</h3>
<p>На вход сети подаются только значения сенсоров. Каждое окно содержит 81 сенсор
на 20 последовательных шагах, то есть 1620 чисел. Соседние окна перекрываются: шаг
сетки равен 10 при длине окна 20. Метка окна соответствует состоянию системы на
последнем шаге окна. Служебные сведения об эксперименте (его номер и границы
окна) нужны только для группировки и разбиения и в признаки не входят.</p>

<h3>4.3. Пайплайн</h3>
<ol>
<li>Парсинг измерений: журнал превращается в таблицу окон.</li>
<li>Фильтрация некорректных значений сенсоров.</li>
<li>Сборка окон: 81 сенсор × 20 шагов при шаге сетки 10.</li>
<li>Разбиение (split) строго по экспериментам: train и test не пересекаются записями.</li>
<li>Стандартизация по train.</li>
<li>Обучение сети с early stopping по val.</li>
<li>Подбор threshold (порог решения) по val внутри train-выборки.</li>
<li>Измерение итоговых метрик на test.</li>
</ol>

<h2>5. Результаты</h2>
<h3>5.1. Метрики на тестовой выборке</h3>
<table>
<tr><th>Показатель</th><th class="num">Значение</th></tr>
{_metrics_rows(final)}
</table>
{report_text.table_legend()}
{report_text.metrics_glossary()}
{_figure(figures, 'confusion')}
{_figure(figures, 'pr_curve')}
{_figure(figures, 'roc')}
{_figure(figures, 'training')}

<h3>5.2. Сравнение с baseline</h3>
<p><b>Что такое baseline.</b> Baseline — простая модель, с которой сравнивают
основную, чтобы отделить вклад архитектуры от вклада самих признаков. Если сеть
не превосходит простейшее предсказание, её сложность не оправдана, поэтому
сравнение с baseline — это рамка честности, а не формальность.</p>
<p>Сравниваются четыре модели на одном и том же разбиении: итоговая нейронная
сеть, логистическая регрессия и два постоянных ответа. Постоянный ответ «отказ»
объявляет отказом каждое окно, постоянный ответ «норма» — наоборот.</p>
<table>
<tr><th>Модель</th><th class="num">accuracy</th><th class="num">precision</th>
<th class="num">recall</th><th class="num">F1</th></tr>
{base_rows}
</table>
{report_text.table_legend()}
{_figure(figures, 'baseline_comparison')}
<p><b>Вывод.</b> Постоянный ответ «отказ» — ловушка: его recall равен 1.000, а
F1 {comparison.get('Постоянный ответ «отказ»', {}).get('f1', 0.0):.3f}
выглядит высоким, но precision при этом всего
{comparison.get('Постоянный ответ «отказ»', {}).get('precision', 0.0):.3f},
потому что модель помечает отказом и норму. Именно поэтому F1 всегда
приводится рядом с precision, а постоянный ответ «норма» служит нижней планкой:
его F1 равен нулю, так как он не находит ни одного отказа.</p>

<h2>6. Сравнение</h2>
<table>
<tr><th>Модель</th><th class="num">accuracy</th><th class="num">precision</th>
<th class="num">recall</th><th class="num">F1</th></tr>
{summary_rows}
</table>
{report_text.table_legend()}
{_figure(figures, 'model_comparison')}

<h2>7. Анализ</h2>
<h3>7.1. Разбор ошибок</h3>
<table>
<tr><th>Эксперимент</th><th class="num">n_windows</th><th class="num">accuracy</th>
<th class="num">precision</th><th class="num">recall</th><th class="num">F1</th></tr>
{exp_rows}
</table>
{report_text.table_legend()}
<p>На трёх записях из четырёх отказы распознаны полностью, а запись 442 тянет
итоговый результат вниз: на ней recall равен 0.250, а F1 = 0.400, и именно она
определяет разницу между test-значением и идеалом. Итоговая метрика по test
складывается из этого разброса между записями, поэтому разбор по экспериментам
важнее усреднённого числа.</p>

<h3>7.2. Важность признаков и корреляции</h3>
<p>Permutation importance (важность перестановкой) в этой работе не считалась:
она требует устойчивой модели и представительной выборки, которых на четырёх
test-записях недостаточно. Вместо неё оценивалась связь качества фолда с
характеристиками записи: долей окон отказа, положением первого отказа, разбросом
сенсоров и расстоянием до обучающих записей. В ячейках ниже приведены
коэффициент Спирмена и p-value по всем шестнадцати записям.</p>
<table>
<tr><th>Характеристика</th><th class="num">ρ (Spearman)</th><th class="num">p-value</th></tr>
{_diagnostics_corr_rows(diagnostics)}
</table>
{report_text.table_legend()}
{_figure(figures, 'diag_frac_positive')}
{_figure(figures, 'diag_dist_to_train')}
{_figure(figures, 'diag_first_fault')}
<p>Максимальный по модулю коэффициент равен 0.350 и остаётся ниже 0.5,
поэтому заметного наклона нет: доступные характеристики записи не объясняют
разброс качества, а коэффициенты такого размера при шестнадцати наблюдениях
ожидаемы и сами по себе о связи не говорят. Отдельно стоит отметить выход
телеметрии за обучающий диапазон: у записи 443 двадцать признаков выходят за 10σ,
а у записи 445 таких признаков шестьдесят; сдвиг по постоянным на train признакам
достигает 315 в исходных единицах (у записи 443 — 3.3). Это описание того, что
телеметрия вышла за пределы, наблюдавшиеся на train, и трактовать его как
объяснение качества нельзя.</p>
<p><b>Самовклад.</b> При расстоянии «с самовкладом» запись всегда находится на
нулевом расстоянии от себя, что искусственно завышает связь: ρ = −0.350. Без
самовклада (<code>dist_to_train_looe</code>) ρ = −0.093, то есть связи нет.
Именно поэтому сравнение ведётся по варианту без самовклада.</p>
<p><b>p-value.</b> Все p > 0.05. При n = 16 мощность теста низкая: отсутствие
значимости не доказывает отсутствия связи, но и не даёт оснований её утверждать.
Диагностика показывает, что доступные характеристики не объясняют разброс;
причина — нехватка данных.</p>

<h3>7.3. Стабильность</h3>
<div class="note">Это диагностика устойчивости, а не результат модели: она не
участвует в выборе архитектуры и threshold.</div>
<p>Итоговое test-значение получено на четырёх записях и о разбросе между ними
молчит. Чтобы измерить этот разброс, помимо CV (cross-validation) по
экспериментам (пять фолдов) применена оценка LOEO (leave-one-experiment-out):
каждая из шестнадцати записей по очереди исключается, сеть обучается на остальных
пятнадцати, а threshold подбирается на одном контроле внутри обучающей части.
Исключённая запись в подборе threshold не участвует, но в тех фолдах, где
исключена другая запись, текущая попадает в train — поэтому это диагностика
процедуры, а не независимая оценка сохранённой модели.</p>
<table>
<tr><th>Показатель</th><th class="num">Среднее ± отклонение</th>
<th class="num">Минимум</th><th class="num">Максимум</th></tr>
<tr><td>F1 (tuned threshold)</td><td class="num">{loeo.get('сводка', {}).get('f1_среднее', 0.0):.3f} ± {loeo.get('сводка', {}).get('f1_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('f1_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('f1_максимум', 0.0):.3f}</td></tr>
<tr><td>F1 (threshold 0.5)</td><td class="num">{loeo.get('сводка_0.5', {}).get('f1_0.5_среднее', 0.0):.3f} ± {loeo.get('сводка_0.5', {}).get('f1_0.5_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('f1_0.5_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('f1_0.5_максимум', 0.0):.3f}</td></tr>
<tr><td>recall (tuned threshold)</td><td class="num">{loeo.get('сводка', {}).get('recall_среднее', 0.0):.3f} ± {loeo.get('сводка', {}).get('recall_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('recall_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('recall_максимум', 0.0):.3f}</td></tr>
<tr><td>recall (threshold 0.5)</td><td class="num">{loeo.get('сводка_0.5', {}).get('recall_0.5_среднее', 0.0):.3f} ± {loeo.get('сводка_0.5', {}).get('recall_0.5_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('recall_0.5_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('recall_0.5_максимум', 0.0):.3f}</td></tr>
<tr><td>precision (tuned threshold)</td><td class="num">{loeo.get('сводка', {}).get('precision_среднее', 0.0):.3f} ± {loeo.get('сводка', {}).get('precision_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('precision_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('precision_максимум', 0.0):.3f}</td></tr>
<tr><td>precision (threshold 0.5)</td><td class="num">{loeo.get('сводка_0.5', {}).get('precision_0.5_среднее', 0.0):.3f} ± {loeo.get('сводка_0.5', {}).get('precision_0.5_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('precision_0.5_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('precision_0.5_максимум', 0.0):.3f}</td></tr>
<tr><td>accuracy (tuned threshold)</td><td class="num">{loeo.get('сводка', {}).get('accuracy_среднее', 0.0):.3f} ± {loeo.get('сводка', {}).get('accuracy_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('accuracy_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка', {}).get('accuracy_максимум', 0.0):.3f}</td></tr>
<tr><td>accuracy (threshold 0.5)</td><td class="num">{loeo.get('сводка_0.5', {}).get('accuracy_0.5_среднее', 0.0):.3f} ± {loeo.get('сводка_0.5', {}).get('accuracy_0.5_отклонение', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('accuracy_0.5_минимум', 0.0):.3f}</td><td class="num">{loeo.get('сводка_0.5', {}).get('accuracy_0.5_максимум', 0.0):.3f}</td></tr>
</table>
{report_text.table_legend()}
<p>Провалов ниже 0.5 F1 — {loeo.get('сводка', {}).get('провалов_ниже_0.5', '—')} из
{loeo.get('фолдов', 16)}, полных попаданий —
{loeo.get('сводка', {}).get('полных_попаданий', '—')}. Результат распределён
неровно: промежуточных значений мало, поэтому одиночное среднее по небольшой
группе записей почти не предсказывает поведение на следующей записи.</p>
{_figure(figures, 'loeo_folds')}
{loeo_notes}

<h2>8. Ограничения</h2>
<ul>{limits}</ul>

<h2>9. Выводы</h2>
<ul>{verdicts}</ul>

<h2>10. Приложение</h2>
<h3>10.1. Схема пайплайна</h3>
{_figure(figures, 'pipeline_schema')}
<h3>10.2. Схема архитектуры</h3>
{_figure(figures, 'model_architecture')}
<p>Описание сети: {_architecture_en(architecture)}.</p>
<h3>10.3. Выходные файлы</h3>
<table>
<tr><th>Файл</th><th>Содержание</th></tr>
<tr><td>metrics_adapt.json</td><td>метрики итоговой модели, разбивка по тестовым экспериментам, CV и проверки пересборки окон</td></tr>
<tr><td>loeo_adapt.json</td><td>оценка LOEO по шестнадцати записям</td></tr>
<tr><td>summary_baselines.json</td><td>метрики итоговой модели и простых моделей на одном разбиении</td></tr>
<tr><td>improvement_adapt.json</td><td>проверенные конфигурации и разложение прироста</td></tr>
<tr><td>loeo_diagnostics.json</td><td>связь качества фолдов с характеристиками записей</td></tr>
<tr><td>split_info.json</td><td>списки обучающих и тестовых экспериментов</td></tr>
<tr><td>model_adapt.keras</td><td>обученная сеть итоговой конфигурации</td></tr>
<tr><td>reports/figures/*.png</td><td>рисунки отчёта</td></tr>
<tr><td>reports/figures/09_diag_f1_vs_frac_positive.png</td><td>связь F1 с долей окон с отказом</td></tr>
<tr><td>reports/figures/10_diag_f1_vs_dist_to_train.png</td><td>связь F1 с расстоянием до обучающих записей</td></tr>
<tr><td>reports/figures/11_diag_f1_vs_first_fault.png</td><td>связь F1 с положением первого отказа</td></tr>
<tr><td>reports/lab2_report.html</td><td>этот отчёт</td></tr>
</table>
<h3>10.4. Проверенные конфигурации</h3>
<table>
<tr><th>Имя</th><th>Дополнения</th><th class="num">n_windows</th>
<th class="num">n_features</th><th class="num">val F1 (threshold 0.5)</th>
<th class="num">val F1 (tuned threshold)</th><th class="num">threshold</th>
<th class="num">test F1</th><th class="num">test recall</th>
<th class="num">ΔF1 (test)</th></tr>
{cfg_rows}
</table>
{report_text.table_legend()}
<p>Столбцы val — то, по чему выбирался победитель; столбцы test
приведены для отчёта. Прирост отсчитывается от базовой конфигурации.
Конфигурации с другим шагом окон обучаются на большем числе окон, но проверяются
всегда на одном и том же эталонном наборе.</p>
<p>A_базовая — точка отсчёта, не набор улучшения. Улучшений — десять (B–K).</p>
<p>У конфигураций H_шаг_5 и I_смещения совпала confusion matrix на test
(recall 0.656, precision 1.000), поэтому их test-метрики совпадают; val-метрики у
них разные.</p>
<h3>10.5. Разложение прироста</h3>
<table>
<tr><th>Этап</th><th class="num">accuracy</th><th class="num">precision</th>
<th class="num">recall</th><th class="num">F1</th><th class="num">threshold</th></tr>
{ladder_rows}
</table>
{failed}
{decomp}
<h3>10.6. Вопросы, требующие уточнения</h3>
<ul>{questions}</ul>

<h2>11. Использованные методы</h2>
<table>
<tr><th>Метод</th><th>Где использовано</th></tr>
<tr><td>Обучение с учителем</td><td>Бинарная классификация нормы и отказа</td></tr>
<tr><td>MLP 128–64–32</td><td>Итоговая конфигурация сети</td></tr>
<tr><td>Dropout 0.3</td><td>Слой Dropout в скрытой части</td></tr>
<tr><td>BatchNorm</td><td>Слои нормализации после Dense-слоёв</td></tr>
<tr><td>L2 0.0001</td><td>Плотные слои сети</td></tr>
<tr><td>cross-entropy (binary)</td><td>Функция потерь</td></tr>
<tr><td>Adam</td><td>Обучение сети, скорость обучения 1e-3</td></tr>
<tr><td>early stopping</td><td>Остановка обучения по val</td></tr>
<tr><td>Стандартизация по train</td><td>Подготовка признаков</td></tr>
<tr><td>Подбор threshold по val</td><td>Выбор threshold</td></tr>
<tr><td>CV (cross-validation) по экспериментам</td><td>Пять фолдов при выборе конфигурации</td></tr>
<tr><td>LOEO (leave-one-experiment-out)</td><td>Диагностика устойчивости по шестнадцати записям</td></tr>
<tr><td>baseline: логистическая регрессия</td><td>Сравнение с итоговой сетью</td></tr>
<tr><td>baseline: постоянные ответы</td><td>Нижняя и верхняя планки по метрикам</td></tr>
<tr><td>Сравнение с baseline</td><td>Разделы 5.2 и 6</td></tr>
</table>

<footer>Отчёт собран локально: данные, вычисления и графики — без сетевых
обращений. Зерно генератора: {seed}.</footer>
</main>
</body>
</html>
"""


def write_report(html: str, path: Path | str = config.REPORT_HTML) -> Path:
    """Записывает готовый HTML в файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
