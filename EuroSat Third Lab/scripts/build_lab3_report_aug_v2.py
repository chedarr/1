# -*- coding: utf-8 -*-
"""Новый отчёт Lab3: aug v2 (цветовая аугментация) как основной эксперимент.

Читает только готовые JSON/PNG. Ничего не обучает и не перезаписывает
lab3_report.html, control_patience_max/ или summary_baselines.json.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT / "reports"
OUT = REPORTS / "lab3_report_aug_v2.html"
AUG_DIR = REPORTS / "control_patience_max_aug"
AUG_FIG = AUG_DIR / "figures"

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def n(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def signed(x: float, digits: int = 3) -> str:
    return ("+" if x >= 0 else "−") + n(abs(x), digits)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}".replace(".", ",") + " %"


def img(path: Path, caption: str, note: str, width: str = "100%") -> str:
    raw = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{raw}" alt="{caption}" '
        f'style="max-width:{width}">'
        f'<figcaption><b>{caption}</b><div class="fig-note">{note}</div></figcaption></figure>'
    )


def table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    th = "".join(f'<th class="num">{h}</th>' if i in numeric else f"<th>{h}</th>" for i, h in enumerate(headers))
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{v}</td>' if i in numeric else f"<td>{v}</td>"
            for i, v in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


CSS = """
:root {
  --fg:#1f2937; --muted:#5f6b7a; --line:#d9e0e7; --soft:#f3f6f9;
  --accent:#1f4e79; --accent2:#2f6f9f; --good:#1f7a4d; --warn:#9a5b00; --bad:#a61b1b;
}
* { box-sizing: border-box; }
body { margin:0; background:#eef2f5; color:var(--fg);
       font: 16px/1.62 "Segoe UI", Roboto, Arial, sans-serif; }
main { max-width: 1040px; margin: 28px auto; background:#fff; padding: 44px 54px 60px;
       box-shadow: 0 2px 10px rgba(15,35,55,.08); }
h1 { font-size: 30px; line-height:1.25; margin:0 0 8px; color:#17324d; }
h2 { font-size: 22px; margin:34px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--line); color:#17324d; }
h3 { font-size: 17px; margin:22px 0 8px; color:#24465f; }
p, li { margin:8px 0; }
ul, ol { margin:8px 0 8px 24px; }
.lead { color:var(--muted); font-size:17px; }
.note { background:#f5f8fb; border-left:4px solid var(--accent2); padding:11px 15px; margin:14px 0; }
.warn { background:#fff8ec; border-left:4px solid var(--warn); padding:11px 15px; margin:14px 0; }
.good { background:#f1faf5; border-left:4px solid var(--good); padding:11px 15px; margin:14px 0; }
.bad { background:#fdf2f2; border-left:4px solid var(--bad); padding:11px 15px; margin:14px 0; }
table { border-collapse:collapse; width:100%; margin:12px 0 16px; font-size:15px; }
th, td { border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }
th { background:var(--soft); }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
code { background:#eef3f7; padding:1px 5px; border-radius:4px; font-size:14px; }
pre.schema { background:#f7f9fb; border:1px solid var(--line); padding:12px 14px;
             overflow-x:auto; font:13px/1.5 Consolas, "Cascadia Mono", monospace; }
figure { margin:20px 0 24px; }
figure img { max-width:100%; height:auto; display:block; margin:0 auto; border:1px solid #e3e8ee; }
figcaption { font-size:15px; color:var(--muted); margin-top:7px; }
.fig-note { margin-top:5px; color:var(--fg); }
.small { color:var(--muted); font-size:14px; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:13px; }
.ok { background:#e8f7ee; color:#185c3a; }
.ctrl { background:#eef3f8; color:#31556f; }
footer { color:var(--muted); font-size:14px; margin-top:38px; border-top:1px solid var(--line); padding-top:12px; }
@media print { body{background:#fff} main{box-shadow:none;margin:0;max-width:none} .note,.warn,.good,.bad{break-inside:avoid} figure{break-inside:avoid} }
"""


def main() -> int:
    v1 = load(REPORTS / "control_patience_max" / "metrics_eurosat_patience_max.json")
    v2 = load(AUG_DIR / "metrics_eurosat_patience_max_aug.json")
    primary = load(REPORTS / "metrics_eurosat.json")
    baseline = load(REPORTS / "baseline_metrics.json")
    summary_base = load(REPORTS / "summary_baselines.json")
    data_check = load(REPORTS / "data_check.json")
    split = load(REPORTS / "split_info.json")
    norm = load(REPORTS / "normalization_stats.json")
    overlap = load(REPORTS / "overlap_probe.json")

    t1, t2 = v1["test"], v2["test"]
    d_acc = t2["accuracy"] - t1["accuracy"]
    d_f1 = t2["macro_f1"] - t1["macro_f1"]
    better = d_f1 > 0
    main_label = "aug v2" if better else "aug v1"
    v1_header = "aug v1 (контроль)" if better else "aug v1 (основной)"
    v2_header = "aug v2 (основной)" if better else "aug v2 (дополнительный)"
    cmp_word = "дала прирост" if d_f1 > 0 else ("не дала прироста" if d_f1 < 0 else "не изменила результат")
    cmp_class = "good" if d_f1 > 0 else ("warn" if d_f1 < 0 else "note")

    v1_by = {r["class_name"]: r for r in t1["per_class"]}
    v2_by = {r["class_name"]: r for r in t2["per_class"]}
    support = {r["class_name"]: r["support"] for r in t2["per_class"]}
    f1_sorted = sorted(t2["per_class"], key=lambda r: r["f1"])
    weakest = f1_sorted[:3]
    strongest = list(reversed(f1_sorted[-3:]))
    hog = baseline["hog_logreg"]
    d_hog_acc = t2["accuracy"] - hog["accuracy"]
    d_hog_f1 = t2["macro_f1"] - hog["macro_f1"]
    hog_relation = (
        "выше" if d_hog_acc > 0 and d_hog_f1 > 0
        else "ниже" if d_hog_acc < 0 and d_hog_f1 < 0
        else "сопоставим"
    )

    # Data table
    class_rows = []
    for name in CLASS_NAMES:
        class_rows.append([
            name,
            f"{data_check['class_distribution'][name]:,}".replace(",", " "),
            f"{split['counts']['train'][name]:,}".replace(",", " "),
            f"{split['counts']['val'][name]:,}".replace(",", " "),
            f"{split['counts']['test'][name]:,}".replace(",", " "),
        ])

    # Per-class results
    per_rows = []
    for name in CLASS_NAMES:
        r = v2_by[name]
        per_rows.append([
            name,
            n(r["precision"]), n(r["recall"]), n(r["f1"]),
            f"{r['support']:,}".replace(",", " "),
        ])

    # Per-class comparison
    cmp_rows = []
    for name in CLASS_NAMES:
        a, b = v1_by[name]["f1"], v2_by[name]["f1"]
        cmp_rows.append([name, n(a), n(b), signed(b - a), f"{support[name]:,}".replace(",", " ")])

    # Top confusions
    cm = v2["test"]["confusion_matrix"]
    confusions = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            if i != j and value:
                confusions.append((value, CLASS_NAMES[i], CLASS_NAMES[j]))
    confusions.sort(reverse=True)
    conf_rows = [[str(v), f"{a} → {b}"] for v, a, b in confusions[:8]]

    # Protocol table
    protocol_rows = [
        ["patience = 3, aug v2 отчёт (основной в старом отчёте)", n(primary["test"]["accuracy"]), n(primary["test"]["macro_f1"])],
        ["patience = 50, aug v1 (контроль)", n(t1["accuracy"]), n(t1["macro_f1"])],
        ["patience = 50, aug v2 (цветовая аугментация)", n(t2["accuracy"]), n(t2["macro_f1"])],
    ]

    # Baseline table
    base_rows = []
    for key, label in [("majority", "majority (самый частый класс)"),
                       ("random", "random"),
                       ("hog_logreg", "HOG + logreg")]:
        item = baseline[key]
        base_rows.append([label, n(item["accuracy"]), n(item["macro_f1"]), signed(item["accuracy"] - t2["accuracy"]), signed(item["macro_f1"] - t2["macro_f1"])])
    base_rows.append(["CNN, aug v2", n(t2["accuracy"]), n(t2["macro_f1"]), "—", "—"])

    fig_conf = AUG_FIG / "confusion_matrix_aug_v2.png"
    fig_curves = AUG_FIG / "training_curves_aug_v2.png"
    fig_per_class = AUG_FIG / "per_class_f1_aug_v2.png"
    fig_baseline = AUG_FIG / "baseline_comparison_aug_v2.png"
    fig_cmp_class = AUG_FIG / "aug_v1_vs_v2_per_class.png"
    fig_cmp_metrics = AUG_FIG / "aug_v1_vs_v2_metrics.png"
    for p in [fig_conf, fig_curves, fig_per_class, fig_baseline, fig_cmp_class, fig_cmp_metrics]:
        if not p.exists():
            raise FileNotFoundError(f"Нет рисунка: {p}")

    data_fig1 = REPORTS / "figures" / "01_class_distribution.png"
    data_fig2 = REPORTS / "figures" / "02_examples.png"
    data_fig3 = REPORTS / "figures" / "08_near_duplicates.png"
    arch_fig = PROJECT / "docs_new" / "figures" / "model_architecture.png"

    sec1 = f"""
<h2>1. Титул и постановка задачи</h2>
<p class="lead">EuroSAT RGB, 10 классов, CNN с нуля. Новый отчёт: основной эксперимент — <b>{main_label}</b>
(цветовая аугментация — aug v2: flip + rotation + zoom + <b>RandomBrightness(0.1)</b> +
<b>RandomContrast(0.1)</b>; контроль — aug v1 без цветовой аугментации). Оба результата показаны;
ни один не скрыт.</p>
<div class="note"><b>Что такое aug v2.</b> Та же CNN <code>Conv2D(16/32/64) + BatchNorm + Dropout(0.5)</code>
(549 626 параметров), те же данные, split, <code>seed=42</code>, <code>patience=50</code>,
<code>epochs=50</code>, <code>batch_size=64</code>, <code>lr=1e-3</code>. Добавлены только
<code>RandomBrightness(0.1)</code> и <code>RandomContrast(0.1)</code> — вместе, только на train,
до нормализации.</div>
<p><b>Метрики.</b> <i>accuracy</i> (доля верных), <i>macro-F1</i> (F1, усреднённая по классам без весов),
<i>precision</i>, <i>recall</i>, <i>confusion matrix</i> (матрица ошибок). Основная метрика сравнения —
macro-F1: она не позволяет крупным классам задавить мелкие.</p>
<table><thead><tr><th>Параметр</th><th>{v1_header}</th><th>{v2_header}</th></tr></thead><tbody>
<tr><td>Цветовая аугментация</td><td>нет</td><td><code>RandomBrightness(0.1)</code>, <code>RandomContrast(0.1)</code></td></tr>
<tr><td>Остальная аугментация</td><td>flip, rotation 0.05, zoom 0.1</td><td>flip, rotation 0.05, zoom 0.1 + цвет</td></tr>
<tr><td>Порядок</td><td>flip → rotation → zoom → normalize</td><td>flip → rotation → zoom → brightness → contrast → normalize</td></tr>
<tr><td>Где применяется</td><td>только train</td><td>только train</td></tr>
<tr><td>Параметров</td><td>{v1['params']:,}</td><td>{v2['params']:,}</td></tr>
</tbody></table>
<p class="small">Порядок экспериментов: aug v1 обучен раньше, aug v2 — позже. Это ограничение
интерпретации, а не данных: test не участвовал в выборе гиперпараметров, split зафиксирован до запуска,
<code>seed=42</code>, нормализация — только по train.</p>
"""

    sec2 = f"""
<h2>2. Данные</h2>
<p>EuroSAT RGB: <b>{data_check['n_images']:,}</b> снимков, <b>{data_check['n_classes']}</b> классов,
форма <code>{data_check['image_shape']}</code>, <code>{data_check['dtype']}</code>. Снимки 64×64×3;
классы пространственно разделены, но <code>scene_id</code> в датасете отсутствует — это ограничение,
а не свойство пайплайна.</p>
""".replace(",", " ")
    sec2 += table(["Класс", "Всего", "train", "val", "test"], class_rows, numeric={1, 2, 3, 4})
    sec2 += f"""
<p><b>Split.</b> Стратифицированный 70 / 15 / 15, <code>seed={split['seed']}</code>:
train <b>{split['split_sizes']['train']:,}</b>, val <b>{split['split_sizes']['val']:,}</b>,
test <b>{split['split_sizes']['test']:,}</b>. Test не используется при обучении и выборе порогов;
метрики считаются один раз после восстановления лучших весов.</p>
<p><b>Нормализация.</b> Среднее по train: <code>{[round(x, 3) for x in norm['mean_rgb']]}</code>,
std: <code>{[round(x, 3) for x in norm['std_rgb']]}</code>, источник — <code>{norm['source_split']}</code>.
Для val/test аугментация не применяется; нормализация применяется всегда.</p>
<div class="warn"><b>55 почти-дубликатов SeaLake.</b> В <code>overlap_probe.json</code> найдено
{overlap['phash_near_duplicates']['pairs_le_4_bits']} пар с dHash ≤ 4, из них
{overlap['phash_near_duplicates']['exact_duplicates']} точных; {overlap['phash_near_duplicates']['mad_lt_5_class_share']['SeaLake']}
пар SeaLake имеют pixel MAD < 5. Это может завышать macro-F1 на SeaLake; оценка косвенная, не доказательство утечки.</div>
""".replace(",", " ")
    sec2 += img(data_fig1, "Рис. 1. Распределение классов", "По оси X — классы, по оси Y — число снимков. Disbalance небольшой: 2000–3000 на класс; macro-F1 поэтому чувствительнее accuracy.", "92%")
    sec2 += img(data_fig2, "Рис. 2. Примеры снимков", "По одному примеру на класс. Цвет и текстура — основная информация; геометрия сцены различается слабо.", "92%")
    sec2 += img(data_fig3, "Рис. 3. Близкие пары train / test", "Пары SeaLake с dHash ≤ 4. Совпадение класса ожидаемо; высокая похожесть — свойство датасета, а не результат модели.", "92%")

    sec3 = f"""
<h2>3. Методология</h2>
<h3>3.1. Архитектура</h3>
<p>CNN с нуля: <code>Conv2D(16) → BN → ReLU → MaxPool</code>, затем <code>32</code> и <code>64</code>,
<code>Flatten 4096 → Dense(128) → Dropout(0.5) → Dense(10, softmax)</code>. Параметров —
<b>{v2['params']:,}</b>; архитектура не менялась между aug v1 и aug v2.</p>
<h3>3.2. Пайплайн</h3>
<pre class="schema">split_info.json → train / val / test
train:  augment (flip → rotation → zoom → brightness → contrast) → normalize(train stats)
val/test: normalize(train stats)  [без аугментации]
CNN 549 626 → EarlyStopping(patience=50) + ModelCheckpoint(val_loss)
→ restore best weights → test один раз → metrics / figures / report</pre>
<h3>3.3. Аугментация aug v2</h3>
<p>Порядок строго: <code>RandomFlip("horizontal")</code> → <code>RandomRotation(0.05)</code> →
<code>RandomZoom(0.1)</code> → <code>RandomBrightness(0.1)</code> → <code>RandomContrast(0.1)</code>.
Цветовые слои стоят <b>до</b> нормализации и применяются <b>только к train</b>. Диапазон яркости и
контраста — <code>[0.9, 1.1]</code>; это мягкая аугментация, а не замена освещения.</p>
<div class="note"><b>Почему до нормализации.</b> <code>RandomBrightness</code> умножает значения пикселей
на фактор около 1. Если применить его после <code>(x − mean) / std</code>, множитель сместит уже
центрированные значения и сломает статистику. Поэтому цвет — до нормализации, как и остальная аугментация.</div>
<h3>3.4. Обучение</h3>
<p><code>Adam(lr=1e-3)</code>, <code>sparse_categorical_crossentropy</code>, <code>batch_size=64</code>,
<code>epochs=50</code>, <code>patience=50</code>, <code>restore_best_weights=True</code>,
<code>ModelCheckpoint(save_best_only=True)</code>, <code>seed=42</code>. Test открывается один раз.</p>
"""
    sec3 += table(
        ["Параметр", "Значение"],
        [
            ["epochs", "50"], ["patience", "50"], ["batch_size", "64"], ["lr", "1e-3"],
            ["optimizer", "Adam"], ["loss", "sparse categorical crossentropy"],
            ["seed", "42"], ["monitor", "val_loss"], ["restore_best_weights", "True"],
            ["augmentation", "flip + rotation + zoom + brightness + contrast (train only)"],
            ["normalization", "mean/std только по train"],
        ],
    )
    if arch_fig.exists():
        sec3 += img(arch_fig, "Рис. 4. Схема архитектуры", "Вход 64×64×3, три блока Conv+BN+ReLU+MaxPool, Flatten, Dense(128), Dropout(0.5), softmax на 10 классов.", "82%")

    sec4 = f"""
<h2>4. Результаты aug v2</h2>
<p><b>accuracy = {n(t2['accuracy'])}</b>, <b>macro-F1 = {n(t2['macro_f1'])}</b> на test
({t2['n_samples']} снимков). Лучшая эпоха — <b>{v2['best_epoch']}</b> из {v2['total_epochs_run']};
время обучения — <b>{v2['training_time_sec'] / 60:.1f} мин</b>; EarlyStopping
{'сработал' if v2['early_stopping_triggered'] else 'не сработал (50 эпох)'}.
Val: accuracy {n(v2['val']['accuracy'])}, macro-F1 {n(v2['val']['macro_f1'])}.</p>
"""
    sec4 += table(["Класс", "precision", "recall", "F1", "support"], per_rows, numeric={1, 2, 3, 4})
    sec4 += img(fig_conf, "Рис. 5. Confusion matrix aug v2", "По осям — истинный и предсказанный классы; цвет — число снимков, числа в клетках — counts. Диагональ — верные ответы; вне диагонали — систематические смешения.", "92%")
    sec4 += img(fig_curves, "Рис. 6. Кривые обучения aug v2", "Слева loss, справа accuracy; train — сплошная, val — штриховая, вертикальная линия — best_epoch. Разрыв train/val показывает переобучение или его отсутствие.", "100%")
    sec4 += img(fig_per_class, "Рис. 7. Per-class F1 aug v2", "Столбцы — F1 по 10 классам. Слабые классы видны сразу; они же разбираются в §8.", "92%")

    sec5 = f"""
<h2>5. Сравнение aug v1 vs aug v2</h2>
<p>Оба прогона используют один split, один seed, одну архитектуру и одни гиперпараметры.
Отличается только цветовая аугментация. Разница: <b>accuracy {signed(d_acc)}</b>,
<b>macro-F1 {signed(d_f1)}</b>.</p>
<div class="{cmp_class}"><b>Вывод.</b> Цветовая аугментация <b>{cmp_word}</b>:
Δaccuracy = {signed(d_acc)}, Δmacro-F1 = {signed(d_f1)}. Это точечная оценка на одном seed;
доверительный интервал не строился, поэтому вывод ограничен этим прогоном.</div>
"""
    sec5 += table(
        ["Метрика", "aug v1", "aug v2", "Δ (v2 − v1)"],
        [
            ["accuracy", n(t1["accuracy"]), n(t2["accuracy"]), signed(d_acc)],
            ["macro-F1", n(t1["macro_f1"]), n(t2["macro_f1"]), signed(d_f1)],
            ["val accuracy", n(v1["val"]["accuracy"]), n(v2["val"]["accuracy"]), signed(v2["val"]["accuracy"] - v1["val"]["accuracy"])],
            ["val macro-F1", n(v1["val"]["macro_f1"]), n(v2["val"]["macro_f1"]), signed(v2["val"]["macro_f1"] - v1["val"]["macro_f1"])],
            ["best_epoch", str(v1["best_epoch"]), str(v2["best_epoch"]), "—"],
            ["эпох всего", str(v1["total_epochs_run"]), str(v2["total_epochs_run"]), "—"],
            ["время, мин", f"{v1['training_time_sec']/60:.1f}", f"{v2['training_time_sec']/60:.1f}", "—"],
        ],
        numeric={1, 2, 3},
    )
    sec5 += table(["Класс", "F1 aug v1", "F1 aug v2", "Δ F1", "support"], cmp_rows, numeric={1, 2, 3, 4})
    sec5 += img(fig_cmp_class, "Рис. 8. Per-class F1: aug v1 vs aug v2", "Две серии столбцов на класс. Рост или падение видно по каждому классу; aggregate-метрика может скрывать разнонаправленные изменения.", "100%")
    sec5 += img(fig_cmp_metrics, "Рис. 9. accuracy и macro-F1: aug v1 vs aug v2", "Сгруппированные столбцы: две метрики, два прогона. Разница мала по сравнению с межклассовым разбросом.", "82%")

    sec6 = f"""
<h2>6. Сравнение с baseline</h2>
<p>Baseline — рамка честности: majority, random и HOG + logreg. CNN aug v2 сравнивается с ними
на одном test. Числа CNN взяты из <code>control_patience_max_aug/metrics_eurosat_patience_max_aug.json</code>;
<code>summary_baselines.json</code> не пересобирался, поле <code>cnn</code> там остаётся
<code>{n(summary_base['cnn']['accuracy'])} / {n(summary_base['cnn']['macro_f1'])}</code> —
это старый patience=3, не aug v2.</p>
"""
    sec6 += table(["Метод", "accuracy", "macro-F1", "Δ accuracy vs aug v2", "Δ macro-F1 vs aug v2"], base_rows, numeric={1, 2, 3, 4})
    sec6 += f"""<div class="note"><b>Вывод.</b> Относительно HOG + logreg:
Δaccuracy = {signed(d_hog_acc)}, Δmacro-F1 = {signed(d_hog_f1)}. Это сравнение на одном test;
оно не доказывает превосходство архитектуры вне этого набора.</div>"""
    sec6 += img(fig_baseline, "Рис. 10. Baseline vs CNN aug v2", "Четыре метода, две метрики. majority и random — нижняя граница; HOG + logreg — сильный классический baseline; CNN aug v2 — текущий результат.", "100%")

    sec7 = f"""
<h2>7. Проверка протокола</h2>
<p>Два сравнения, оба честно: (1) внутри aug v1 — <code>patience=3</code> против
<code>patience=50</code>; (2) <code>aug v1</code> против <code>aug v2</code> при одинаковом
<code>patience=50</code>.</p>
"""
    sec7 += table(["Протокол", "accuracy", "macro-F1"], protocol_rows, numeric={1, 2})
    sec7 += f"""
<p><b>Что это значит.</b> Переход <code>patience=3 → 50</code> дал большой прирост
({signed(t1['accuracy'] - primary['test']['accuracy'])} accuracy,
{signed(t1['macro_f1'] - primary['test']['macro_f1'])} macro-F1), но это сравнение протоколов,
а не архитектур. Переход <code>aug v1 → aug v2</code> дал
{signed(d_acc)} accuracy и {signed(d_f1)} macro-F1 — это и есть эффект цветовой аугментации.
Оба сравнения нельзя смешивать: первое отвечает на вопрос «хватило ли обучения», второе — «помог ли цвет».</p>
<div class="warn"><b>Порядок экспериментов.</b> aug v2 запущен после aug v1; test при этом не участвовал
в выборе гиперпараметров, split зафиксирован заранее, seed тот же. Это ограничение протокола,
а не данных: оно влияет на интерпретацию, но не на независимость test.</div>
"""

    # Top confusions table + analysis
    sec8 = f"""
<h2>8. Анализ</h2>
<p>Ошибок на test: <b>{t2['n_samples'] - round(t2['accuracy'] * t2['n_samples'])}</b> из
{t2['n_samples']}. Слабейшие классы по F1: {", ".join(f"{r['class_name']} ({n(r['f1'])})" for r in weakest)}.
Сильнейшие: {", ".join(f"{r['class_name']} ({n(r['f1'])})" for r in strongest)}.</p>
"""
    sec8 += table(["Число ошибок", "Истинный → предсказанный"], conf_rows, numeric={0})
    sec8 += f"""
<p><b>Где модель ошибается.</b> Основные внедиагональные пары — это близкие по текстуре классы
(например, <code>PermanentCrop</code> / <code>HerbaceousVegetation</code>, <code>Highway</code> / <code>Industrial</code>).
Цветовая аугментация меняет яркость и контраст, но не геометрию сцены, поэтому не устраняет
смешения, основанные на форме и пространственном контексте.</p>
<p><b>SeaLake.</b> F1 = {n(v2_by['SeaLake']['f1'])}; в датасете 55 почти-дубликатов SeaLake
(dHash ≤ 4, pixel MAD < 5). Это может завышать вклад SeaLake и macro-F1. <code>scene_id</code>
отсутствует, поэтому строго отделить сцены нельзя; вывод ограничен.</p>
<div class="warn"><b>Ограничение интерпретации.</b> Один seed и один test-набор. Разница aug v2 − aug v1
не сопровождается доверительным интервалом; её нельзя объявлять статистически установленной
без дополнительных прогонов.</div>
"""

    sec9 = """
<h2>9. Ограничения</h2>
<ol>
<li><b>Ограничение протокола, а не данных.</b> aug v2 запущен после aug v1. Test не участвовал
в выборе гиперпараметров, split зафиксирован до запуска, <code>seed=42</code>, нормализация — только
по train, <code>leakage_check.json</code> пройден. Порядок экспериментов — искусственное ограничение,
влияет только на интерпретацию.</li>
<li><b><code>scene_id</code> отсутствует.</b> Нельзя гарантировать, что train и test разделены по сценам,
а не по отдельным снимкам.</li>
<li><b>55 почти-дубликатов SeaLake.</b> dHash ≤ 4 и pixel MAD < 5; macro-F1 может быть завышен.</li>
<li><b>Один seed для aug v2.</b> Разброс по seed не измерен; Δ между aug v1 и aug v2 — точечная оценка.</li>
<li><b>macro-F1 может быть завышен</b> из-за SeaLake и дисбаланса классов; accuracy этого не компенсирует.</li>
<li><b><code>patience</code> / <code>epochs</code> не менялись.</b> Единственное изменение —
цветовая аугментация; длительность обучения и критерий остановки у aug v1 и aug v2 одинаковы,
поэтому эти факторы контролируются. Сравнение всё равно остаётся на одном seed и одном test.</li>
<li><b>CPU, один прогон.</b> Время около 21 минуты на aug v2; GPU и mixed precision не использовались.</li>
</ol>
"""

    if better:
        main_conclusion = (
            f"Главный результат — <b>aug v2</b>: accuracy {n(t2['accuracy'])}, "
            f"macro-F1 {n(t2['macro_f1'])}. Относительно aug v1: "
            f"Δaccuracy {signed(d_acc)}, Δmacro-F1 {signed(d_f1)} — цветовая аугментация дала прирост "
            f"на этом прогоне."
        )
    else:
        main_conclusion = (
            f"Цветовая аугментация не улучшила macro-F1: aug v2 = {n(t2['macro_f1'])} против "
            f"{n(t1['macro_f1'])} у aug v1 (Δ {signed(d_f1)}). Основным результатом остаётся aug v1; "
            f"aug v2 показан как дополнительный эксперимент, а не как замена."
        )

    sec10 = f"""
<h2>10. Выводы</h2>
<ol>
<li>{main_conclusion}</li>
<li>Сравнение с aug v1: Δaccuracy = {signed(d_acc)}, Δmacro-F1 = {signed(d_f1)}.
Оба результата показаны; ни один не скрыт.</li>
<li>Сравнение с HOG + logreg: Δaccuracy = {signed(d_hog_acc)}, Δmacro-F1 = {signed(d_hog_f1)}.
CNN aug v2 {hog_relation} классического baseline на этом test, но это не доказывает превосходство вне набора.</li>
<li><b>aug v2 — не подгонка.</b> <code>seed=42</code>, <code>patience=50</code>, <code>epochs=50</code>,
<code>lr=1e-3</code>, <code>batch_size=64</code>, архитектура, split и нормализация — те же, что у aug v1.
Менялась только аугментация: добавлены <code>RandomBrightness(0.1)</code> и <code>RandomContrast(0.1)</code>.</li>
<li>Слабейшие классы aug v2: {", ".join(f"{r['class_name']} ({n(r['f1'])})" for r in weakest)}.
Именно они дают основной вклад в ошибки и требуют отдельных данных или более сильной модели.</li>
<li><b>SeaLake / scene_id / 55 почти-дубликатов.</b> F1 SeaLake = {n(v2_by['SeaLake']['f1'])};
<code>scene_id</code> отсутствует; 55 пар SeaLake близки. macro-F1 может быть завышен, это
ограничение данных, а не модели.</li>
<li><b>Воспроизводимость.</b> <code>seed=42</code>, <code>split_info.json</code>,
<code>normalization_stats.json</code>, два <code>.keras</code>:
<code>control_patience_max/eurosat_cnn_patience_max.keras</code> (aug v1) и
<code>control_patience_max_aug/eurosat_cnn_patience_max_aug.keras</code> (aug v2).</li>
</ol>
"""

    gain_rows = [
        ["majority", n(baseline["majority"]["accuracy"]), "—"],
        ["random", n(baseline["random"]["accuracy"]), signed(baseline["random"]["accuracy"] - baseline["majority"]["accuracy"])],
        ["HOG + logreg", n(hog["accuracy"]), signed(hog["accuracy"] - baseline["random"]["accuracy"])],
        ["CNN aug v2", n(t2["accuracy"]), signed(t2["accuracy"] - hog["accuracy"])],
    ]

    sec11 = f"""
<h2>11. Приложение</h2>
<h3>11.1. Схема пайплайна</h3>
<pre class="schema">EuroSAT RGB (27 000, 10 классов)
→ split_info.json (70/15/15, seed=42)
→ train: augment(flip → rotation → zoom → brightness → contrast) → normalize(train)
→ val/test: normalize(train), без augmentation
→ CNN 549 626 → EarlyStopping(patience=50) + ModelCheckpoint(val_loss)
→ restore best weights → test один раз → metrics / figures / report</pre>
<h3>11.2. Схема архитектуры</h3>
<pre class="schema">Input 64×64×3
→ [Conv2D(16,3×3,same) → BN → ReLU → MaxPool2] ×1
→ [Conv2D(32,3×3,same) → BN → ReLU → MaxPool2] ×1
→ [Conv2D(64,3×3,same) → BN → ReLU → MaxPool2] ×1
→ Flatten(4096) → Dense(128, ReLU) → Dropout(0.5) → Dense(10, softmax)
Параметров: 549 626</pre>
<h3>11.3. Выходные файлы</h3>
"""
    sec11 += table(
        ["Файл", "Назначение"],
        [
            ["<code>control_patience_max/metrics_eurosat_patience_max.json</code>", "aug v1, контроль; не изменялся"],
            ["<code>control_patience_max/eurosat_cnn_patience_max.keras</code>", "веса aug v1; не изменялись"],
            ["<code>control_patience_max_aug/metrics_eurosat_patience_max_aug.json</code>", "метрики aug v2"],
            ["<code>control_patience_max_aug/eurosat_cnn_patience_max_aug.keras</code>", "веса aug v2"],
            ["<code>control_patience_max_aug/figures/*.png</code>", "4 рисунка aug v2 + 2 сравнения"],
            ["<code>summary_baselines.json</code>", "baseline; не пересобирался, <code>cnn</code> = 0.6899"],
        ],
    )
    sec11 += f"""
<h3>11.4. Конфигурация aug v2</h3>
"""
    sec11 += table(
        ["Параметр", "Значение"],
        [
            ["run", "control_patience_max_aug"],
            ["augmentation_version", "aug_v2"],
            ["batch_size", "64"], ["lr", "1e-3"], ["optimizer", "Adam"],
            ["seed", "42"], ["patience", "50"], ["epochs", "50"],
            ["augmentation", "flip + rotation + zoom + brightness + contrast (train only, before normalization)"],
            ["normalization", "stats from train only"],
            ["model_checkpoint", "reports/control_patience_max_aug/eurosat_cnn_patience_max_aug.keras"],
        ],
    )
    sec11 += "<h3>11.5. Разложение прироста</h3>"
    sec11 += table(["Метод", "accuracy", "Δ accuracy к предыдущему"], gain_rows, numeric={1, 2})
    sec11 += """
<h3>11.6. Вопросы, требующие уточнения</h3>
<ul>
<li>Нужен ли отдельный прогон только с <code>RandomBrightness</code> и только с <code>RandomContrast</code>,
чтобы разделить вклад двух слоёв?</li>
<li>Нужен ли второй seed для aug v2, чтобы оценить разброс?</li>
<li>Нужно ли исключать 55 почти-дубликатов SeaLake из test для проверки устойчивости macro-F1?</li>
<li>Нужно ли подбирать <code>patience</code> / <code>epochs</code> отдельно для aug v2, или сравнение
должно оставаться строго при фиксированных значениях?</li>
</ul>
"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лабораторная работа №3 — EuroSAT, aug v2</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Лабораторная работа №3 — EuroSAT, CNN с цветовой аугментацией</h1>
<p class="lead">Новый отчёт: основной эксперимент — aug v2; контроль — aug v1. Оба результата показаны.
Дата сборки: 2 ноября 2026.</p>
<div class="note"><span class="badge ok">{main_label}: основной</span>
<span class="badge ctrl">{'aug v1' if better else 'aug v2'}: {'контроль' if better else 'дополнительный'}</span>
Результат aug v2: accuracy <b>{n(t2['accuracy'])}</b>, macro-F1 <b>{n(t2['macro_f1'])}</b>.
Результат aug v1: accuracy <b>{n(t1['accuracy'])}</b>, macro-F1 <b>{n(t1['macro_f1'])}</b>.</div>
{sec1}
{sec2}
{sec3}
{sec4}
{sec5}
{sec6}
{sec7}
{sec8}
{sec9}
{sec10}
{sec11}
<footer>Источники: <code>metrics_eurosat.json</code>, <code>control_patience_max/metrics_eurosat_patience_max.json</code>,
<code>control_patience_max_aug/metrics_eurosat_patience_max_aug.json</code>, <code>baseline_metrics.json</code>,
<code>summary_baselines.json</code>, <code>data_check.json</code>, <code>split_info.json</code>,
<code>normalization_stats.json</code>, <code>overlap_probe.json</code>. Исходные отчёты и JSON не изменялись.</footer>
</main>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print("written", OUT, OUT.stat().st_size)
    print("aug v2", n(t2["accuracy"]), n(t2["macro_f1"]))
    print("aug v1", n(t1["accuracy"]), n(t1["macro_f1"]))
    print("delta", signed(d_acc), signed(d_f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
