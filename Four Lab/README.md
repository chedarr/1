# Лабораторная работа №4 — многомерный прогноз GOCE с помощью LSTM

Прогноз напряжения батареи (`PHT10180`, BAT_VOL_R) на 24 часа вперёд по 24 часам истории
пяти каналов GOCE с шагом 10 минут.

## Данные

`data/raw/goce/` — 5 parquet (`PHT10080`, `PHT10100`, `PHT10180`, `PHT10260`, `PHT10300`)
и `metadata.csv`. После inner join по `timestamp` — 239 616 строк, 231 194 остаются после
удаления строк с NaN (3,515 %). Длинных NaN-серий (больше 6 шагов) — 3, самая длинная
7 270 шагов (50,5 суток). Разрывы не склеиваются: окна, пересекающие разрыв, выбрасываются.

## Постановка

- Вход: окно `lookback = 144` шага × 5 каналов (`value_mean`).
- Выход: `horizon = 144` значений целевой (direct multi-step).
- Split: хронологический 70/15/15 по строкам ряда, без перемешивания.
- Нормализация: `StandardScaler` только по train.
- Метрики: MAE, RMSE, R², skill score относительно persistence.
- Baseline: persistence (naive), constant, linear (Ridge на 720 признаках).

## Окружение

Python 3.13.15; TensorFlow 2.20.0 (CPU, GPU не используется), pandas 3.0.6, numpy 2.5.3,
scikit-learn 1.9.1, matplotlib 3.11.2, pyarrow 25.0.1, Pillow, pytest 9.1.1.
`seed = 42`, `USE_GPU = False`, `MIXED_PRECISION = False`, 6 потоков CPU.
Версии зафиксированы в `requirements.txt`.

## Запуск

```powershell
python scripts/check_project.py
python scripts/prepare_data.py
python scripts/run_baselines.py
python scripts/run_experiment.py --name A --save-model
python scripts/run_experiment.py --name B
python scripts/run_experiment.py --name C
python scripts/run_experiment.py --name D
python scripts/run_experiment.py --name E
python scripts/run_experiment.py --name F
python scripts/run_experiment.py --name G
python scripts/run_experiment.py --name E0
python scripts/run_experiment.py --name lb72
python scripts/run_experiment.py --name lb288
python scripts/run_experiment.py --name hz72
python scripts/run_experiment.py --name hz28
python scripts/summarize.py
python scripts/make_figures.py
python scripts/build_report.py
pytest -q
```

## Структура

```
src/            config, loader, features, split, pipeline, model, metrics, baselines, plots, html_report
scripts/        prepare_data, run_baselines, run_experiment, summarize, make_figures, check_project,
                check_leakage, build_report
data/raw/goce/  5 parquet + metadata.csv
data/interim/   data_summary.json
data/processed/ X_*.npy, y_*.npy, scaler.pkl, split_info.json
reports/        lab4_report.html, summary_baselines.json, metrics_experiments.json,
                ablation_results.json, leakage_check.json, experiments/*, figures/*
docs/           lab4_likbez.md
tests/          pytest
```

## Результаты

Сводка — в `reports/lab4_report.html`; числа — только из JSON в `reports/`.
