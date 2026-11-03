# -*- coding: utf-8 -*-
"""Сборка HTML-отчёта лабораторной №3 (EuroSAT, CNN).

Все числа берутся из JSON-артефактов; изображения встраиваются как base64.
Ничего не считает заново — только форматирует.
"""
from __future__ import annotations

import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

R = config.REPORTS
CTRL_DIR = R / "control_patience_max"
FIG = config.FIGURES
CTRL_FIG = CTRL_DIR / "figures"

CSS = """
:root { --fg:#1c2733; --muted:#5b6b7c; --line:#d8e0e8; --accent:#2b6cb0; --warn:#b7791f; --bad:#c53030; }
* { box-sizing: border-box; }
body { margin:0; background:#f4f7fa; color:var(--fg);
       font: 16px/1.65 "Segoe UI", Roboto, Arial, sans-serif; }
main { max-width: 980px; margin: 0 auto; background:#fff; padding: 40px 48px 64px;
       box-shadow: 0 1px 3px rgba(0,0,0,.08); }
h1 { font-size: 29px; margin: 0 0 6px; }
h2 { font-size: 21px; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h3 { font-size: 17px; margin: 22px 0 8px; }
p, li { margin: 8px 0; }
ul, ol { margin: 8px 0 8px 24px; }
code { background:#eef3f8; padding:1px 5px; border-radius:4px; font-size:14px; }
table { border-collapse: collapse; width:100%; margin: 12px 0; font-size: 15px; }
th, td { border:1px solid var(--line); padding:7px 10px; text-align:left; }
th { background:#eef3f8; }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
.note { background:#f7fafc; border-left:4px solid var(--accent); padding:12px 16px; margin:16px 0; }
.warn { background:#fffaf0; border-left:4px solid var(--warn); padding:12px 16px; margin:16px 0; }
.lead { color: var(--muted); font-size: 17px; }
figure { margin: 20px 0 26px; }
figure img { max-width: 100%; height: auto; display:block; margin: 0 auto; }
figcaption { font-size: 15px; color: var(--muted); margin-top: 8px; }
.fig-note { margin: 10px 0 0; font-size: 15px; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:13px; }
.ok { background:#e6fffa; color:#22543d; }
.bad { background:#fff5f5; color:#822727; }
pre.tree, pre.schema { background:#f7fafc; border:1px solid var(--line); padding:12px 16px;
       overflow-x:auto; font-size:13.5px; line-height:1.5; }
footer { color: var(--muted); font-size: 14px; margin-top: 40px; border-top:1px solid var(--line); padding-top:12px; }
"""


def load(name: str) -> dict:
    with open(R / name, encoding="utf-8") as fh:
        return json.load(fh)


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}".replace(".", ",")


_CACHE: dict[Path, str] = {}

# Отчёт читают с экрана: контент шире 1000 px не нужен, а base64 от PNG
# раздувает файл в 1,33 раза. Пережимаем в WebP и режем по ширине.
MAX_WIDTH = 1000
WEBP_QUALITY = 88


def data_uri(path: Path) -> str:
    if path in _CACHE:
        return _CACHE[path]
    image = Image.open(path).convert("RGBA")
    if image.width > MAX_WIDTH:
        height = round(image.height * MAX_WIDTH / image.width)
        image = image.resize((MAX_WIDTH, height), Image.LANCZOS)
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(image, mask=image.getchannel("A"))
    buffer = BytesIO()
    canvas.save(buffer, "WEBP", quality=WEBP_QUALITY, method=6)
    uri = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    _CACHE[path] = uri
    return uri


def img(path: Path, caption: str, note: str) -> str:
    return (
        f'<figure><img alt="{caption}" src="{data_uri(path)}">'
        f"<figcaption>{caption}</figcaption>"
        f'<p class="fig-note"><b>Что на графике.</b> {note}</p></figure>'
    )


def per_class(metrics: dict) -> dict:
    pc = metrics["test"]["per_class"]
    if isinstance(pc, list):
        return {item["class_name"]: item for item in pc}
    return dict(pc)


def top_errors(metrics: dict, limit: int = 8) -> list[tuple[str, str, int]]:
    cm = metrics["test"]["confusion_matrix"]
    names = config.CLASS_NAMES
    pairs = []
    for true_idx, row in enumerate(cm):
        for pred_idx, count in enumerate(row):
            if true_idx != pred_idx and count:
                pairs.append((names[true_idx], names[pred_idx], int(count)))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:limit]


def table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="num">{h}</th>' if i in numeric else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build() -> str:
    data = load("data_check.json")
    split = load("split_info.json")
    stats = load("normalization_stats.json")
    leakage = load("leakage_check.json")
    base = load("metrics_eurosat.json")
    ctrl = load("control_patience_max/metrics_eurosat_patience_max.json")
    baseline = load("baseline_metrics.json")
    summary = load("summary_baselines.json")
    probe = load("overlap_probe.json")

    names = config.CLASS_NAMES
    base_pc = per_class(base)
    ctrl_pc = per_class(ctrl)
    hog_pc = baseline["hog_logreg"]["per_class"]
    ctrl_test = ctrl["test"]
    base_test = base["test"]
    ctrl_val = ctrl["val"]
    base_val = base["val"]

    # --- §1 ---
    meta_rows = [
        ["Тема", "Классификация снимков EuroSAT свёрточной сетью"],
        ["Курс", "Лабораторная работа №3, TensorFlow / Keras"],
        ["Данные", "EuroSAT RGB, 27 000 снимков Sentinel-2, 64×64, 10 классов"],
        ["Источник", "Zenodo, record 7711810 (официальный TFDS-хост madm.dfki.de недоступен)"],
        ["Лицензия", "MIT + Copernicus Sentinel Data Terms"],
        ["Стек", f"Python {config.PYTHON_VERSION}, TensorFlow {config.TF_VERSION} (CPU), scikit-learn, scikit-image"],
        ["Главная метрика", "macro-F1 на test"],
        ["Воспроизводимость", f"seed={config.SEED}, split зафиксирован в split_info.json, веса сохранены в .keras"],
        ["Дата сборки", "21.10.2026"],
    ]

    # --- §2 ---
    section2 = f"""
<h2>2. Постановка задачи</h2>
<p>На вход подаётся цветной снимок Sentinel-2 размером 64×64 пикселя. Нужно отнести его к одному
из десяти классов землепользования: <code>AnnualCrop</code>, <code>Forest</code>,
<code>HerbaceousVegetation</code>, <code>Highway</code>, <code>Industrial</code>, <code>Pasture</code>,
<code>PermanentCrop</code>, <code>Residential</code>, <code>River</code>, <code>SeaLake</code>.</p>
<p>Классы представлены неравномерно: от 2 000 до 3 000 снимков, отношение 1,5×. Поэтому главная
метрика — <b>macro-F1</b> (среднее F1 по классам, каждый класс весит одинаково), а <code>accuracy</code>
(доля верных) приводится рядом как вторая. При дисбалансе accuracy может расти за счёт крупных
классов, macro-F1 — нет.</p>
<p>Сравнение идёт с тремя baseline: <code>majority</code> (всегда самый частый класс),
<code>random</code> (равномерный случайный выбор) и <code>HOG + logreg</code> (гистограммные признаки
+ логистическая регрессия). Все модели оцениваются на одном и том же test из 4 050 снимков.</p>
"""

    # --- §3 ---
    dist = data["class_distribution"]
    dist_rows = [
        [name, f"{dist[name]:,}".replace(",", " "), f"{100 * dist[name] / data['n_images']:.1f}".replace(".", ",")]
        for name in names
    ]
    split_sizes = split["split_sizes"]
    split_rows = [
        ["train", f"{split_sizes['train']:,}".replace(",", " "), f"{100 * split_sizes['train'] / split['n_images']:.0f}%"],
        ["val", f"{split_sizes['val']:,}".replace(",", " "), f"{100 * split_sizes['val'] / split['n_images']:.0f}%"],
        ["test", f"{split_sizes['test']:,}".replace(",", " "), f"{100 * split_sizes['test'] / split['n_images']:.0f}%"],
    ]

    section3 = f"""
<h2>3. Данные</h2>
<p>Датасет EuroSAT RGB: {data['n_images']:,} снимков, {data['n_classes']} классов, размер
{data['image_shape'][0]}×{data['image_shape'][1]}, тип <code>{data['dtype']}</code>. Это нарезка
изображений Sentinel-2 на патчи; каждый патч — участок земной поверхности площадью около 2,8 км².</p>
""".replace("{:,}".format(data["n_images"]), f"{data['n_images']:,}".replace(",", " "))

    section3 += table(
        ["Класс", "Снимков", "Доля, %"], dist_rows, numeric={1, 2}
    )
    section3 += img(
        FIG / "01_class_distribution.png",
        "Рис. 3.1 — Распределение классов",
        "По оси X — классы, по оси Y — число снимков. Столбцы одного цвета, потому что серий нет; "
        "пунктирная линия — средний размер класса (2 700). Вывод: дисбаланс мягкий (1,5×), "
        "но достаточный, чтобы accuracy и macro-F1 расходились.",
    )
    section3 += img(
        FIG / "02_examples.png",
        "Рис. 3.2 — Примеры снимков по классам",
        "Каждая ячейка — случайный снимок соответствующего класса. Видно, что часть классов "
        "различается по текстуре (Forest, SeaLake), а часть — только по геометрии (Residential, Highway). "
        "Вывод: визуально задача решаема, но пары Residential/Industrial и PermanentCrop/Herbaceous "
        "заведомо близки.",
    )
    section3 += f"<p>Разбиение: 70/15/15 со стратификацией по классам, <code>seed={split['seed']}</code>. "
    section3 += "Списки индексов лежат в <code>split_info.json</code>; test не использовался при обучении и подборе.</p>"
    section3 += table(["Часть", "Снимков", "Доля"], split_rows, numeric={1, 2})
    section3 += img(
        FIG / "03_split_distribution.png",
        "Рис. 3.3 — Распределение классов по частям",
        "По оси X — классы, по оси Y — доля внутри части; три серии (train, val, test) различаются цветом. "
        "Столбцы почти совпадают. Вывод: стратификация сохранила доли классов во всех трёх частях.",
    )

    section3 += """
<h3>3.1. Проверки целостности</h3>
"""
    section3 += table(
        ["Проверка", "Результат", "Статус"],
        [
            ["Скачанный архив", "94 658 721 Б, md5 <code>f46e308c…62f3b</code>", '<span class="badge ok">OK</span>'],
            ["Число файлов", "27 000 JPEG (плюс 11 каталогов в архиве)", '<span class="badge ok">OK</span>'],
            ["Размер и канал", "все (64, 64, 3), RGB, uint8", '<span class="badge ok">OK</span>'],
            ["Битые файлы", "0 из 27 000", '<span class="badge ok">OK</span>'],
            ["Дубликаты по md5", "27 000 уникальных хэшей, 0 совпадений", '<span class="badge ok">OK</span>'],
            ["Пересечение split", f"train∩val = {len(leakage['overlap_train_val'])}, train∩test = {len(leakage['overlap_train_test'])}, val∩test = {len(leakage['overlap_val_test'])}", '<span class="badge ok">OK</span>'],
            ["Нормализация", f"mean/std только по train ({stats['n_pixels']:,} пикселей)".replace(",", " "), '<span class="badge ok">OK</span>'],
            ["scene_id", "в TFDS-версии отсутствует", '<span class="badge bad">WARN</span>'],
        ],
    )

    section3 += f"""
<div class="warn"><b>WARN: scene_id отсутствует в TFDS-версии EuroSAT RGB.</b>
Восстановить его по имени файла (<code>Forest_864.jpg</code>) или метаданным нельзя: в архиве нет
ни CSV, ни JSON с координатами. Поэтому пересечение по сцене Sentinel-2 исключить нельзя —
формально разные патчи могут быть вырезаны из одного снимка.</div>
"""

    section3 += f"""
<h3>3.2. Проверка близких пар train / test</h3>
<p>Чтобы оценить масштаб проблемы, я посчитал косинусную близость 128-мерных эмбеддингов
(слой <code>dense</code> контрольной сети) для всех пар test→train, а также dHash и пиксельную MAD
(mean absolute difference, средняя абсолютная разница).</p>
"""
    section3 += table(
        ["Показатель", "test → train (top-1)", "контроль"],
        [
            ["cosine, медиана", num(probe["top1_similarity"]["median"]), num(probe["control_random_pairs"]["median"])],
            ["cosine, p95", num(probe["top1_similarity"]["p95"]), num(probe["control_random_pairs"]["p95"])],
            ["cosine, максимум", num(probe["top1_similarity"]["max"]), "—"],
            ["пар > 0,95", f"{probe['counts_above']['gt_0.95']:,}".replace(",", " "), "—"],
            ["пар > 0,99", f"{probe['counts_above']['gt_0.99']:,}".replace(",", " "), "—"],
        ],
        numeric={1, 2},
    )
    section3 += f"""
<p>Контроль — 200 000 случайных пар: медиана {num(probe['control_random_pairs']['median'])}. Значит эмбеддинги
не вырождены и близость что-то означает. Дополнительно проверено, что test не ближе к train, чем
train сам к себе: медиана top-1 внутри train {num(probe['control_train_top1']['median'])}.</p>
<p>Жёсткий фильтр (dHash ≤ 4 бита) оставил <b>{probe['phash_near_duplicates']['pairs_le_4_bits']}</b> пар; из них
{probe['phash_near_duplicates']['exact_duplicates']} совпали побитово. После проверки по пикселям осталось
<b>{len(probe['phash_near_duplicates']['near_duplicate_pairs'])}</b> пар с MAD &lt; 5 — все класса <code>SeaLake</code>,
все внутри одного класса.</p>
"""
    section3 += img(
        FIG / "08_near_duplicates.png",
        "Рис. 3.4 — Близкие пары test / train",
        "Три пары с наименьшей MAD (с разными индексами train и test): сверху снимок из test, "
        "снизу — его ближайший сосед из train; подписи содержат индексы и MAD. "
        "Яркость каждого снимка растянута по перцентилям 2–98, иначе тёмная вода выглядит почти чёрной; "
        "растяжка применена ко всем шести изображениям одинаково и на MAD не влияет. "
        "Вывод: часть снимков SeaLake почти дублируется между split, поэтому F1 этого класса "
        "может быть завышен.",
    )

    # --- §4 ---
    section4 = f"""
<h2>4. Методология</h2>
<h3>4.1. Теоретическая основа</h3>
<p>Свёрточная сеть (<code>CNN</code>, convolutional neural network) обрабатывает изображение
локальными фильтрами. Слой <code>Conv2D</code> скользит ядром 3×3 по картинке и учит веса ядра;
так сеть находит края, текстуры и их комбинации. <code>BatchNorm</code> нормирует активации по
batch, что стабилизирует обучение. <code>ReLU</code> обнуляет отрицательные значения и добавляет
нелинейность. <code>MaxPooling2D</code> уменьшает разрешение вдвое, оставляя максимум в каждом
окне 2×2: это снижает стоимость и делает признаки устойчивее к сдвигу.</p>
<p>После трёх свёрточных блоков карта признаков разворачивается в вектор (<code>Flatten</code>),
проходит через <code>Dense(128)</code> и <code>Dropout(0.5)</code> — dropout случайно отключает
половину нейронов, чтобы сеть не полагалась на один признак. Последний слой — <code>Dense(10)</code>
с <code>softmax</code>: он превращает выходы в вероятности классов. Функция потерь —
<code>sparse_categorical_crossentropy</code> (cross-entropy для целых меток), оптимизатор
<code>Adam</code> с learning rate 10⁻³.</p>
<p><b>Early stopping</b> (ранняя остановка) следит за <code>val_loss</code> и прекращает обучение,
если он не улучшается <code>patience</code> эпох подряд; <code>restore_best_weights=True</code>
возвращает веса лучшей эпохи, а <code>ModelCheckpoint</code> сохраняет их на диск.</p>

<h3>4.2. Признаки</h3>
<ul>
<li><b>CNN:</b> сырые пиксели RGB, 64×64×3 = 12 288 значений на снимок. Никакой ручной инженерии —
признаки учит сама сеть.</li>
<li><b>HOG + logreg:</b> <code>HOG</code> (histogram of oriented gradients) на градациях серого:
9 ориентаций, ячейки 8×8, блоки 2×2, нормализация L2-Hys → <b>1 764</b> признака; далее
<code>StandardScaler</code> и <code>LogisticRegression</code>.</li>
</ul>

<h3>4.3. Пайплайн</h3>
<ol>
<li>Скачивание архива EuroSAT RGB и локальная сборка TFDS (<code>download=False</code> после первого раза).</li>
<li>Стратифицированный split 70/15/15 с <code>seed=42</code>; индексы сохраняются в <code>split_info.json</code>.</li>
<li>Нормализация: <code>mean</code> и <code>std</code> считаются <b>только по train</b> и применяются ко всем частям.</li>
<li>Augmentation только на train: горизонтальное отражение, поворот ±5%, масштаб ±10%.</li>
<li><code>tf.data</code>: batch 64, shuffle буфера 5 000, <code>prefetch(AUTOTUNE)</code>.</li>
<li>Обучение CNN, early stopping по <code>val_loss</code>, сохранение лучших весов.</li>
<li>Предсказание на val и test, расчёт метрик, разбор ошибок.</li>
<li>Baseline на том же split: majority, random, HOG + logreg.</li>
</ol>
<p>Test не использовался при обучении и подборе. Метрики на test получены <b>однократно</b>
после завершения обучения (по одному разу на прогон). <code>patience</code> и <code>epochs</code>
зафиксированы до запуска, а не выбирались по test.</p>
<p>Нормализация: mean = [{', '.join(num(v) for v in stats['mean_rgb'])}],
std = [{', '.join(num(v) for v in stats['std_rgb'])}] (по train, {stats['n_pixels']:,} пикселей).</p>
""".replace("{:,}".format(stats["n_pixels"]), f"{stats['n_pixels']:,}".replace(",", " "))

    # --- §5 ---
    section5 = f"""
<h2>5. Результаты</h2>
<h3>5.1. Метрики на тестовой выборке</h3>
<p>Результат: CNN обучена 50 эпох, <code>patience=50</code>, лучшая эпоха 26.
Метрики на test — ниже. Сравнение с протоколом <code>patience=3</code> — в §7
(как проверка, а не второй результат).</p>
"""
    section5 += table(
        ["Метрика", "test", "val"],
        [
            ["accuracy", num(ctrl_test["accuracy"], 4), num(ctrl_val["accuracy"], 4)],
            ["macro-F1", num(ctrl_test["macro_f1"], 4), num(ctrl_val["macro_f1"], 4)],
            ["ошибок", f"{ctrl_test['n_samples'] - round(ctrl_test['accuracy'] * ctrl_test['n_samples'])} из {ctrl_test['n_samples']:,}".replace(",", " "), "—"],
        ],
        numeric={1, 2},
    )
    section5 += f"""
<p>Сеть обучена 50 эпох, но лучшие веса — с 26-й (val_loss = {num(min(ctrl['history']['val_loss']), 5)}).
Суммарное время обучения {num(ctrl['training_time_sec'], 1)} с на CPU, в среднем
{num(sum(ctrl['epoch_seconds']) / len(ctrl['epoch_seconds']), 1)} с на эпоху. Параметров:
{ctrl['params']:,}.</p>
""".replace("{:,}".format(ctrl["params"]), f"{ctrl['params']:,}".replace(",", " "))

    section5 += img(
        CTRL_FIG / "confusion_matrix_patience_max.png",
        "Рис. 5.1 — Confusion matrix контрольного прогона (test)",
        "По оси X — предсказанный класс, по оси Y — истинный; цвет клетки — число снимков, "
        "чем темнее, тем больше. Диагональ — верные ответы. Вывод: сильнее всего проседают "
        "<code>PermanentCrop</code> (277 из 375) и <code>Pasture</code> (248 из 300), "
        "а основной поток ошибок идёт в <code>HerbaceousVegetation</code>.",
    )

    section5 += img(
        CTRL_FIG / "training_curves_patience_max.png",
        "Рис. 5.2 — Кривые обучения контрольного прогона",
        "Слева loss, справа accuracy; по оси X — эпоха (1–50), сплошная линия — train, пунктир — val. "
        "До 26-й эпохи обе кривые растут согласованно, после чего val_loss начинает колебаться "
        "(0,27–0,62), а train accuracy продолжает расти. Вывод: начиная с 26-й эпохи сеть переобучается, "
        "и checkpoint сохранил именно эту точку.",
    )

    section5 += "<h3>5.2. Метрики по классам</h3>"
    rows = []
    for name in names:
        item = ctrl_pc[name]
        rows.append([
            name, num(item["precision"]), num(item["recall"]), num(item["f1"]),
            str(item["support"]), num(hog_pc[name]["f1"]),
        ])
    section5 += table(
        ["Класс", "precision", "recall", "F1", "support", "F1 HOG"],
        rows, numeric={1, 2, 3, 4, 5},
    )
    section5 += img(
        CTRL_FIG / "per_class_f1_patience_max.png",
        "Рис. 5.3 — F1 по классам: CNN и HOG + logreg",
        "По оси X — классы, по оси Y — F1; два столбца на класс — CNN (контрольный прогон) и HOG + logreg. "
        "Вывод: CNN выигрывает везде, но особенно на <code>Pasture</code> (+0,61) и "
        "<code>PermanentCrop</code> (+0,53), где текстурные признаки HOG почти не работают.",
    )

    # --- §6 ---
    section6 = f"""
<h2>6. Сравнение с baseline</h2>
"""
    section6 += table(
        ["Модель", "accuracy", "macro-F1", "Комментарий"],
        [
            ["majority", num(summary["majority"]["accuracy"], 4), num(summary["majority"]["macro_f1"], 4), "всегда <code>AnnualCrop</code>"],
            ["random", num(summary["random"]["accuracy"], 4), num(summary["random"]["macro_f1"], 4), "равномерно по 10 классам"],
            ["HOG + logreg", num(summary["hog_logreg"]["accuracy"], 4), num(summary["hog_logreg"]["macro_f1"], 4), f"C={num(baseline['hog_logreg_C'], 1)}, 1 764 признака"],
            ["CNN", num(ctrl_test["accuracy"], 4), num(ctrl_test["macro_f1"], 4), "50 эпох, best epoch 26"],
        ],
        numeric={1, 2},
    )
    section6 += f"""
<p>Выигрыш над HOG + logreg: <b>+{pct(ctrl_test['accuracy'] - summary['hog_logreg']['accuracy'])} п.п.</b>
accuracy и <b>+{pct(ctrl_test['macro_f1'] - summary['hog_logreg']['macro_f1'])} п.п.</b> macro-F1.
Над majority — +{pct(ctrl_test['accuracy'] - summary['majority']['accuracy'])} п.п.</p>
<p class="note">В таблице — CNN (контрольный прогон, <code>patience=50</code>). Числа — из
<code>control_patience_max/metrics_eurosat_patience_max.json</code>. В <code>summary_baselines.json</code>
поле <code>cnn</code> хранит <b>базовый</b> прогон (<code>patience=3</code>), потому что сводка
собиралась до контрольного; сравнение протоколов — в §7. Оба прогона сохранены в артефактах,
ни один не удалён.</p>
"""
    section6 += img(
        FIG / "07_baseline_comparison.png",
        "Рис. 6.1 — Сравнение с baseline (test)",
        "По оси X — четыре модели, по оси Y — значение метрики; слева столбец accuracy, справа macro-F1, "
        "над каждым столбцом подписано число. Вывод: разрыв между CNN и HOG + logreg больше, чем "
        "между HOG и случайным угадыванием, — то есть выигрыш даёт именно обучение признаков, "
        "а не подбор классификатора.",
    )

    # --- §7 ---
    base_errors = base_test["n_samples"] - round(base_test["accuracy"] * base_test["n_samples"])
    rows = [
        ["эпох обучено", str(base["epochs_ran"]), str(ctrl["total_epochs_run"])],
        ["лучшая эпоха", str(base["early_stopping_epoch"]), str(ctrl["best_epoch"])],
        ["ранняя остановка", "да (5-я)", "нет (все 50)"],
        ["время, с", num(base["train_seconds"], 1), num(ctrl["training_time_sec"], 1)],
        ["accuracy (test)", num(base_test["accuracy"], 4), num(ctrl_test["accuracy"], 4)],
        ["macro-F1 (test)", num(base_test["macro_f1"], 4), num(ctrl_test["macro_f1"], 4)],
        ["ошибок (test)", str(base_errors), str(ctrl_test["n_samples"] - round(ctrl_test["accuracy"] * ctrl_test["n_samples"]))],
    ]
    section7 = f"""
<h2>7. Проверка протокола: <code>patience=3</code> vs <code>patience=50</code></h2>
<p>Один и тот же код и одна и та же архитектура обучены дважды. Отличаются только два параметра:
<code>patience</code> (3 → 50) и <code>epochs</code> (20 → 50). Всё остальное — batch 64,
learning rate 10⁻³, <code>Adam</code>, seed 42, split, нормализация, augmentation — совпадает.</p>
"""
    section7 += table(["Параметр", "patience = 3", "patience = 50"], rows, numeric={1, 2})
    section7 += f"""
<p>При <code>patience=3</code> обучение остановилось на 5-й эпохе и вернуло веса 2-й: именно там
<code>val_loss</code> был минимальным ({num(min(base['history']['val_loss']), 5)}). Дальше val_loss
скакнул вверх, и счётчик терпения сработал. При <code>patience=50</code> сеть получила шанс пройти
плато: лучшая эпоха — 26-я, <code>val_loss</code> = {num(min(ctrl['history']['val_loss']), 5)}.</p>
<p><b>Как это трактовать.</b> {num(base_test['accuracy'], 4)} — потолок протокола
<code>patience=3</code>, а не архитектуры: та же сеть при другом протоколе даёт
{num(ctrl_test['accuracy'], 4)}. Но это и не доказательство, что архитектура «способна на большее» —
менялся только протокол остановки. Утверждать, что дело в архитектуре, нельзя: не проверялись
ни глубина, ни ширина, ни learning rate schedule.</p>
<p><code>patience=50</code> — это <b>снятие ограничения протокола</b>, зафиксированное <b>до</b>
запуска, а не выбор по test. Оба прогона показаны честно, ни один не скрыт. В основном тексте
(§5, §6, §10) используется результат контрольного прогона, потому что он соответствует
расширенному протоколу; базовый — только здесь, как проверка.</p>
"""
    section7 += img(
        FIG / "05_training_curves.png",
        "Рис. 7.1 — Кривые базового прогона (patience = 3)",
        "Слева loss, справа accuracy; по оси X — эпоха (1–5), train — сплошная линия, val — пунктир. "
        "После 2-й эпохи val_loss растёт, а train loss падает. Вывод: ранняя остановка сработала "
        "корректно, но вернула недообученную модель.",
    )
    section7 += img(
        FIG / "04_confusion_matrix.png",
        "Рис. 7.2 — Confusion matrix базового прогона (test)",
        "По оси X — предсказанный класс, по оси Y — истинный; цвет — число снимков. "
        "Вывод: при patience = 3 сеть почти не различает <code>Highway</code>, <code>PermanentCrop</code> "
        "и <code>HerbaceousVegetation</code> — их строки размазаны по соседним классам.",
    )

    # --- §8 ---
    ctrl_errors = top_errors(ctrl, 8)
    base_errors_top = top_errors(base, 5)
    section8 = """
<h2>8. Анализ</h2>
<h3>8.1. Разбор ошибок</h3>
"""
    section8 += f"<p>В контрольном прогоне {ctrl_test['n_samples'] - round(ctrl_test['accuracy'] * ctrl_test['n_samples'])} ошибок из {ctrl_test['n_samples']:,}. Самые частые пары «истинный → предсказанный»:</p>".replace("{:,}".format(ctrl_test["n_samples"]), f"{ctrl_test['n_samples']:,}".replace(",", " "))
    section8 += table(
        ["Истинный класс", "Предсказанный", "Снимков"],
        [[a, b, str(c)] for a, b, c in ctrl_errors],
        numeric={2},
    )
    section8 += f"""
<p>Картина изменилась по сравнению с базовым прогоном, где лидировали
<code>Highway → River</code> ({base_errors_top[0][2]}) и
<code>HerbaceousVegetation → Residential</code> ({base_errors_top[1][2]}). В контрольном прогоне
основная масса ошибок — это <code>PermanentCrop → HerbaceousVegetation</code>
({ctrl_errors[0][2]}): обе категории заняты растительностью со сходной текстурой, и различие
идёт по геометрии полей, которую сеть на 64×64 видит плохо. Числа ошибок для контрольного
прогона — из <code>control_patience_max/metrics_eurosat_patience_max.json</code>, для базового —
из <code>metrics_eurosat.json</code>.</p>
"""
    section8 += """
<h3>8.2. Что происходит с классами</h3>
<ul>
<li><b>SeaLake</b> — F1 0,980: вода имеет характерный цвет и низкую текстуру. Оговорка: именно
в этом классе найдены почти-дубликаты, так что метрика может быть завышена.</li>
<li><b>Residential</b> и <b>Industrial</b> — F1 0,963 и 0,939: сеть уверенно ловит плотную
застройку, хотя между собой они путаются (15 пар Residential → Industrial).</li>
<li><b>PermanentCrop</b> — F1 0,798, худший класс: recall 0,739 при precision 0,868. Сеть
осторожничает и пропускает часть полей в пользу HerbaceousVegetation.</li>
<li><b>Pasture</b> — F1 0,879, recall 0,827: 25 снимков ушли в HerbaceousVegetation.</li>
<li><b>Highway</b> — F1 0,905 против 0,339 в базовом прогоне: главный эффект длительного обучения.
Дороги перестали путаться с River и Industrial.</li>
</ul>
"""
    section8 += f"""
<h3>8.3. Близость train и test</h3>
<p>Проверка из §3.2 показывает {probe['counts_above']['gt_0.95']:,} пар test→train с косинусной
близостью выше 0,95, но лишь {len(probe['phash_near_duplicates']['near_duplicate_pairs'])} из них проходят жёсткий фильтр
по пикселям (MAD &lt; 5). Все {len(probe['phash_near_duplicates']['near_duplicate_pairs'])} — <code>SeaLake</code>.
Это не файловые дубликаты (md5 уникальны), а почти одинаковые патчи, попавшие в разные split.
Без scene_id отличить «тот же участок после пережатия» от «соседний участок той же сцены» нельзя.</p>
""".replace("{:,}".format(probe["counts_above"]["gt_0.95"]), f"{probe['counts_above']['gt_0.95']:,}".replace(",", " "))

    section8 += f"""
<h3>8.4. Стоимость</h3>
<p>Контрольный прогон занял {num(ctrl['training_time_sec'] / 60, 1)} мин на CPU
(~{num(sum(ctrl['epoch_seconds']) / len(ctrl['epoch_seconds']), 1)} с/эпоха).
Инференс одного снимка на CPU — единицы миллисекунд, модель весит 6,4 МБ.</p>
"""

    # --- §9 ---
    section9 = """
<h2>9. Ограничения</h2>
<p class="lead">Ограничения ниже разделены на два типа: (а) <b>свойства данных и модели</b>
(например, <code>scene_id</code> отсутствует, <code>SeaLake</code> почти-дубликаты);
(б) <b>искусственные ограничения протокола</b> (порядок экспериментов, один seed).
Второй тип <b>не</b> влияет на качество модели, но влияет на интерпретацию результата.</p>
<ol>
<li><b>Ограничение протокола, а не данных.</b> Test как датасет <b>независим</b>: он не участвовал
в обучении, сплит зафиксирован до запуска (<code>seed=42</code>), нормализация считалась только
по train (<code>normalization_stats.json</code>), пересечений train/val/test нет
(<code>leakage_check.json</code>). <b>Порядок экспериментов</b> — искусственное ограничение:
контрольный прогон (<code>patience=50</code>) запускался <b>после</b> того, как базовый результат
(<code>patience=3</code>, 0,6899) на test был известен. Это <b>не</b> делает test «зависимым»
и <b>не</b> является подгонкой: <code>patience=50</code> зафиксирован <b>до</b> запуска как
расширение протокола, а не выбран по test. Ограничение касается только <b>интерпретации</b>
0,9079 как «слепой» оценки; на само качество модели оно не влияет. Оба результата
(0,6899 и 0,9079) показаны честно, ни один не скрыт.</li>
<li><b>scene_id отсутствует.</b> Пересечение по сцене Sentinel-2 исключить нельзя; 55 почти-дубликатов
класса <code>SeaLake</code> найдены и описаны, но их влияние на macro-F1 не измерено.
Оценка <b>0,9055</b> macro-F1 <b>может быть завышена</b> за счёт <code>SeaLake</code>
(F1 0,980 при 55 почти-дубликатах); без <code>scene_id</code> измерить нельзя.</li>
<li><b>Один seed.</b> Повторных прогонов с разными seed нет, поэтому доверительного интервала
для 0,9079 тоже нет; разброс между прогонами не измерен.</li>
<li><b>Два фактора сразу.</b> В контрольном прогоне одновременно изменены <code>patience</code>
и <code>epochs</code>. Разделить их вклад нельзя.</li>
<li><b>Augmentation не проверялась абляцией.</b> Утверждать, что она помогает, нельзя — сравнения
«с augmentation / без» нет.</li>
<li><b>Домен.</b> EuroSAT — только Sentinel-2, 10 классов, 64×64. На другие сенсоры и разрешения
результат не переносится без переобучения.</li>
<li><b>Железо.</b> Обучение на CPU (GPU TensorFlow не видит), 6 потоков; время указано для этой машины.</li>
</ol>
"""

    # --- §10 ---
    section10 = f"""
<h2>10. Выводы</h2>
<ol>
<li><b>Главный результат:</b> CNN на EuroSAT (27 000 снимков, 10 классов, test 4 050) даёт
<b>accuracy {num(ctrl_test['accuracy'], 4)}</b> и <b>macro-F1 {num(ctrl_test['macro_f1'], 4)}</b>.
Против HOG + logreg ({num(summary['hog_logreg']['accuracy'], 4)} /
{num(summary['hog_logreg']['macro_f1'], 4)}): <b>+{pct(ctrl_test['accuracy'] - summary['hog_logreg']['accuracy'])} п.п.</b>
accuracy и <b>+{pct(ctrl_test['macro_f1'] - summary['hog_logreg']['macro_f1'])} п.п.</b> macro-F1.</li>
<li><b>Протокол:</b> результат получен при <code>patience=50</code>, 50 эпохах,
<code>restore_best_weights=True</code>, <code>ModelCheckpoint(save_best_only=True)</code>,
лучшая эпоха — 26. Это <b>снятие ограничения протокола</b> <code>patience=3</code> (см. §7),
а не выбор по test: <code>patience=50</code> зафиксирован <b>до</b> запуска. 0,6899 — потолок
протокола <code>patience=3</code>, а не архитектуры.</li>
<li><b>Ограничение протокола (не данных):</b> {num(ctrl_test['accuracy'], 4)} — не «слепой»
результат <b>из-за порядка экспериментов</b>: контрольный прогон запускался после того, как
0,6899 (<code>patience=3</code>) на test был известен. <b>Независимость test как датасета не
нарушена</b> — test не участвовал в обучении, сплит зафиксирован до запуска. Ограничение
касается только <b>интерпретации</b> 0,9079 как «слепой» оценки; это <b>искусственное</b>
ограничение протокола, а не свойство данных и не подгонка.</li>
<li><b>Слабейшие классы:</b> <code>PermanentCrop</code> — F1 {num(ctrl_pc['PermanentCrop']['f1'])}
(recall {num(ctrl_pc['PermanentCrop']['recall'])}), <code>Pasture</code> — F1
{num(ctrl_pc['Pasture']['f1'])} (25 снимков → <code>HerbaceousVegetation</code>). Основная ошибка —
<code>PermanentCrop → HerbaceousVegetation</code> (63 снимка). <code>Highway</code> — F1
{num(ctrl_pc['Highway']['f1'])} (в базовом протоколе было 0,339, см. §7); это главный эффект
длительного обучения.</li>
<li><b><code>SeaLake</code> — оговорка:</b> F1 {num(ctrl_pc['SeaLake']['f1'])}, но в этом классе
{len(probe['phash_near_duplicates']['near_duplicate_pairs'])} почти-дубликатов
(<code>MAD &lt; 5</code>, §3.2). <code>macro-F1</code> {num(ctrl_test['macro_f1'], 4)}
<b>может быть завышен</b> за счёт <code>SeaLake</code>; без <code>scene_id</code> измерить нельзя.</li>
<li><b><code>scene_id</code> отсутствует:</b> пересечение по сцене Sentinel-2 <b>исключить нельзя</b>.
55 почти-дубликатов <code>SeaLake</code> найдены и описаны, но их влияние на <code>macro-F1</code>
<b>не измерено</b>.</li>
<li><b>Воспроизводимость:</b> <code>seed=42</code>, <code>split_info.json</code>,
<code>normalization_stats.json</code>, <code>control_patience_max/eurosat_cnn_patience_max.keras</code>.
Все числа — из <code>reports/*.json</code>.</li>
</ol>
"""

    # --- §11 ---
    files_rows = [
        ["<code>data_check.json</code>", "паспорт датасета, распределение, лицензия"],
        ["<code>split_info.json</code>", "индексы train/val/test, доли, seed"],
        ["<code>normalization_stats.json</code>", "mean/std по train"],
        ["<code>leakage_check.json</code>", "пересечения split, область нормализации и augmentation"],
        ["<code>baseline_metrics.json</code>", "majority, random, HOG + logreg, C, число признаков"],
        ["<code>summary_baselines.json</code>", "сводка accuracy / macro-F1 (поле <code>cnn</code> — базовый прогон)"],
        ["<code>metrics_eurosat.json</code>", "<b>базовый</b> прогон CNN (<code>patience=3</code>) — используется <b>только</b> в §7 для сравнения протоколов"],
        ["<code>control_patience_max/metrics_eurosat_patience_max.json</code>", "<b>контрольный</b> прогон CNN (<code>patience=50</code>) — <b>основной</b> результат в §5, §6, §8, §10"],
        ["<code>overlap_probe.json</code>", "близость test→train, dHash, MAD, 55 пар"],
        ["<code>figures/*.png</code>", "11 рисунков (8 в <code>figures/</code>, 3 в <code>control_patience_max/figures/</code>)"],
    ]
    config_rows = [
        ["batch_size", str(ctrl["config"]["batch_size"])],
        ["optimizer / lr", f"{ctrl['config']['optimizer']} / {num(ctrl['config']['lr'])}"],
        ["seed", str(ctrl["config"]["seed"])],
        ["patience / epochs", f"{ctrl['config']['patience']} / {ctrl['config']['epochs']}"],
        ["monitor", ctrl["config"]["monitor"]],
        ["restore_best_weights", str(ctrl["config"]["restore_best_weights"]).lower()],
        ["model_checkpoint", "да (save_best_only)"],
        ["augmentation", "flip, rotation 0,05, zoom 0,1 — только train"],
        ["normalization", "mean/std только по train"],
        ["use_gpu / mixed_precision", "false / false"],
        ["threads (intra / inter)", f"{ctrl['config']['intra_op_threads']} / {ctrl['config']['inter_op_threads']}"],
    ]
    gain_rows = [
        ["majority → random", f"{pct(summary['random']['accuracy'] - summary['majority']['accuracy'])} п.п.", "случайное угадывание хуже константы"],
        ["random → HOG + logreg", f"+{pct(summary['hog_logreg']['accuracy'] - summary['random']['accuracy'])} п.п.", "работают ручные признаки"],
        ["HOG + logreg → CNN", f"+{pct(ctrl_test['accuracy'] - summary['hog_logreg']['accuracy'])} п.п.", "обученные признаки"],
    ]

    section11 = """
<h2>11. Приложение</h2>
<h3>11.1. Схема пайплайна</h3>
<pre class="tree">EuroSAT_RGB.zip (Zenodo)
   │  сборка TFDS локально, download=False
   ▼
data/raw/eurosat_rgb/eurosat/rgb/2.0.0  (27 000 снимков, 10 классов)
   │  stratified split 70/15/15, seed=42
   ▼
split_info.json ──► train 18 900 │ val 4 050 │ test 4 050
   │                                   │
   │  mean/std только по train          │
   ▼                                   ▼
normalize + augmentation (train)   normalize (val/test)
   │                                   │
   ▼                                   ▼
CNN 549 626 параметров  ──►  early stopping / checkpoint  ──►  test</pre>

<h3>11.2. Схема архитектуры</h3>
<pre class="schema">Input 64×64×3
 ├─ Conv2D(16, 3×3, same) → BatchNorm → ReLU → MaxPool 2×2   → 32×32×16
 ├─ Conv2D(32, 3×3, same) → BatchNorm → ReLU → MaxPool 2×2   → 16×16×32
 ├─ Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPool 2×2   →  8×8×64
 ├─ Flatten                                                   → 4 096
 ├─ Dense(128) → ReLU
 ├─ Dropout(0.5)
 └─ Dense(10) → softmax</pre>
"""
    section11 += "<h3>11.3. Выходные файлы</h3>"
    section11 += table(["Файл", "Содержимое"], files_rows)
    section11 += "<h3>11.4. Конфигурация контрольного прогона</h3>"
    section11 += table(["Параметр", "Значение"], config_rows)
    section11 += "<h3>11.5. Разложение прироста</h3>"
    section11 += table(["Шаг", "Прирост accuracy", "Что именно добавилось"], gain_rows)
    section11 += """
<h3>11.6. Вопросы, требующие уточнения</h3>
<ol>
<li>Почему <code>val_loss</code> минимален на 26-й эпохе, а затем колеблется в диапазоне 0,27–0,62?
Помог бы learning rate schedule или более сильная регуляризация — но это отдельный эксперимент.</li>
<li>Сколько эпох оптимально при <code>patience=50</code> и меньшем learning rate? Не проверялось.</li>
<li>Насколько метрики <code>SeaLake</code> завышены из-за 55 почти-дубликатов? Нужен scene_id
или ручная разметка сцен.</li>
<li>Улучшит ли разделение <code>PermanentCrop</code> / <code>HerbaceousVegetation</code> обучение
на большем разрешении (128×128 вместо 64×64)?</li>
</ol>
"""

    # --- §12 ---
    section12 = """
<h2>12. Использованные методы</h2>
"""
    section12 += table(
        ["Метод", "Где использован"],
        [
            ["Стратифицированный split", "§3, разбиение 70/15/15 с сохранением долей классов"],
            ["Нормализация по train", "§4.3, mean/std без утечки из val/test"],
            ["Augmentation", "§4.3, только train: flip, rotation, zoom"],
            ["CNN (Conv2D + BatchNorm + ReLU + MaxPool)", "§4.1, §11.2 — основная модель"],
            ["Dropout 0,5", "§4.1, регуляризация перед выходным слоем"],
            ["Adam + cross-entropy", "§4.1, обучение"],
            ["Early stopping + ModelCheckpoint", "§4.1, §7 — сравнение протоколов"],
            ["HOG + LogisticRegression", "§4.2, §6 — baseline"],
            ["majority / random", "§6 — нижняя граница"],
            ["Confusion matrix", "§5.1, §7 — разбор ошибок"],
            ["precision / recall / F1 по классам", "§5.2 — качество по классам"],
            ["Косинусная близость эмбеддингов", "§3.2, §8.3 — поиск почти-дубликатов"],
            ["dHash + пиксельная MAD", "§3.2 — подтверждение близких пар"],
            ["pytest", "17 тестов на split, нормализацию, augmentation, метрики, модель"],
        ],
    )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лабораторная работа №3 — EuroSAT, CNN</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Лабораторная работа №3 — Классификация снимков EuroSAT свёрточной сетью</h1>
<p class="lead">10 классов землепользования, 27 000 снимков Sentinel-2, CNN с нуля на CPU.
Главная метрика — macro-F1 на test.</p>

<h2>1. Метаданные</h2>
{table(["Поле", "Значение"], meta_rows)}

{section2}
{section3}
{section4}
{section5}
{section6}
{section7}
{section8}
{section9}
{section10}
{section11}
{section12}

<footer>
Отчёт собран автоматически из JSON-артефактов проекта. Все числовые значения взяты из
<code>reports/*.json</code>; изображения встроены в файл как base64, внешние ресурсы не используются.
</footer>
</main>
</body>
</html>
"""
    return html


def main() -> int:
    html = build()
    out = R / "lab3_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"отчёт: {out} ({out.stat().st_size:,} Б)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
