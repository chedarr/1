# Лабораторная №3 — CNN на EuroSAT

Классификация спутниковых снимков Sentinel-2 (RGB) по 10 классам
землепользования. Этап обработки данных: загрузка, проверка целостности,
стратифицированный split, нормализация по train, аугментация только на train.

## Данные

* TFDS `eurosat/rgb`, 27 000 снимков, `(64, 64, 3)`, `uint8`, 10 классов.
* Кэш TFDS — локально в `data/raw/eurosat_rgb/` (не в `~/.cache`).
* Однократное скачивание: `python scripts/download_eurosat.py`.
* Все дальнейшие запуски — `download=False`, сеть не используется.

## Окружение

* Windows, Python 3.13, TensorFlow 2.20.0 (CPU-сборка, `is_cuda_build: False`).
* `USE_GPU=False`, `MIXED_PRECISION=False`, потоки CPU: 6 / 1.
* `seed=42`.

## Структура

```
src/            config.py, loader.py, split.py, normalize.py, augment.py,
                metrics.py, cnn.py
scripts/        download_eurosat.py, prepare_tfds_local.py, check_loaded.py,
                check_integrity.py, build_artifacts.py, run_baselines.py,
                train_cnn.py, make_model_figures.py
data/raw/       кэш TFDS
reports/        JSON-артефакты, eurosat_cnn.keras, figures/
docs/           data_processing_notes.md, model_training_notes.md
tests/          тесты сплита, нормализации, аугментации, метрик, CNN
```

## Запуск

```powershell
python scripts\download_eurosat.py     # однократно
python scripts\build_artifacts.py      # data_check, split, normalization, leakage
python scripts\run_baselines.py        # majority, random, HOG + logreg
python scripts\train_cnn.py            # CNN, metrics_eurosat.json, .keras
python scripts\make_model_figures.py   # рисунки 04–07
python scripts\train_cnn_patience_max.py   # контрольный прогон, 50 эпох
python scripts\make_control_figures.py     # рисунки контрольного прогона
python scripts\overlap_probe.py            # близость test→train, near-duplicate пары
python scripts\make_overlap_figure.py      # рисунок 08
python scripts\build_lab3_report.py        # HTML-отчёт
python scripts\build_lab3_likbez.py        # HTML-ликбез
python -m pytest tests -q
```

## Результаты (test, 4 050 снимков)

| Модель | accuracy | macro-F1 |
|---|---|---|
| majority | 0.1111 | 0.0200 |
| random | 0.0970 | 0.0968 |
| HOG + logreg | 0.5570 | 0.5397 |
| CNN (основной, `patience=3`) | 0.6899 | 0.6703 |
| CNN (`patience=max`, контрольный) | 0.9079 | 0.9055 |

Основной прогон остановился на 5-й эпохе (`patience=3`), поэтому модель
недообучена. Тот же протокол с `patience=50` и 50 эпохами (`best_epoch=26`)
даёт 0.9079: 0.6899 — следствие остановки, а не свойство архитектуры. При этом
это и не доказательство, что архитектура «способна на большее»: менялся только
протокол остановки. Подробности — в `docs/model_training_notes.md`.

## Артефакты

* `reports/data_check.json` — размер, классы, распределение, лицензия.
* `reports/split_info.json` — split по индексам снимков, `seed=42`.
* `reports/normalization_stats.json` — `mean`/`std` по каналам, только train.
* `reports/leakage_check.json` — пересечения выборок и контроль утечек.
* `reports/baseline_metrics.json` — baseline `majority`, `random`,
  `HOG + logreg` (per-class, confusion matrix, выбранный `C`).
* `reports/metrics_eurosat.json` — метрики CNN на val/test, history,
  per-class, confusion matrix, индексы ошибок.
* `reports/summary_baselines.json` — accuracy и macro-F1 всех моделей.
* `reports/eurosat_cnn.keras` — веса CNN.
* `reports/control_patience_max/` — контрольный прогон: JSON, `.keras`,
  `figures/` (не заменяет базовые артефакты).
* `reports/overlap_probe.json` — близость test→train по эмбеддингам,
  dHash, MAD и 55 near-duplicate пар.
* `reports/figures/` — распределение классов, примеры снимков, split,
  confusion matrix, кривые обучения, per-class F1, сравнение с baseline,
  near-duplicate пары.
* `reports/lab3_report.html` — самодостаточный HTML-отчёт (изображения base64).
* `reports/lab3_likbez.html`, `docs/lab3_likbez.md` — ликбез по работе.
