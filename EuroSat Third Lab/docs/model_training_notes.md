# Этап модели: CNN и baseline на EuroSAT

Документ фиксирует протокол обучения, архитектуру, метрики и ограничения.
Все числа взяты из артефактов `reports/metrics_eurosat.json`,
`reports/baseline_metrics.json`, `reports/summary_baselines.json`.

## 1. Данные и протокол

* Выборки берутся из `reports/split_info.json` по индексам: train 18 900,
  val 4 050, test 4 050 (стратифицированный split 70/15/15, `seed=42`).
* Нормализация — статистики из `reports/normalization_stats.json`,
  посчитанные только по train: mean `[86.711, 97.047, 102.688]`,
  std `[51.973, 35.123, 29.682]`.
* Augmentation применяется только к train-батчам
  (`src/augment.py`, слой внутри `tf.data`), val и test идут без искажений.
* `batch_size=64`, оптимизатор Adam (`learning_rate=1e-3`),
  loss `sparse_categorical_crossentropy`, до 20 эпох.
* Early stopping: `monitor="val_loss"`, `patience=3`,
  `restore_best_weights=True`. Метрика остановки — loss, а не accuracy:
  это задано протоколом этапа.

## 2. Архитектура

| Блок | Слои | Выход |
|---|---|---|
| вход | — | 64 × 64 × 3 |
| 1 | Conv2D(16, 3×3, same) → BatchNorm → ReLU → MaxPool(2) | 32 × 32 × 16 |
| 2 | Conv2D(32, 3×3, same) → BatchNorm → ReLU → MaxPool(2) | 16 × 16 × 32 |
| 3 | Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPool(2) | 8 × 8 × 64 |
| классификатор | Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, softmax) | 10 |

Всего параметров: **549 626**. Свёрточные слои без bias (его роль берёт
BatchNorm). Модель обучается с нуля, без transfer learning.

## 3. Ход обучения

| epoch | train loss | train acc | val loss | val acc |
|---|---|---|---|---|
| 1 | 1.4157 | 0.4896 | 1.1493 | 0.6015 |
| 2 | 1.0929 | 0.6035 | **0.9217** | 0.6988 |
| 3 | 0.9527 | 0.6598 | 1.2365 | 0.6153 |
| 4 | 0.9040 | 0.6722 | 0.9468 | 0.6864 |
| 5 | 0.8387 | 0.7020 | 0.9315 | 0.6943 |

Early stopping сработал на 5-й эпохе (val loss не улучшался 3 эпохи после
2-й). Веса восстановлены на лучшую эпоху — **2-ю**. Время обучения на CPU
(Ryzen 5 3600, 6 потоков): 132.7 с.

## 4. Метрики на test

| Модель | accuracy | macro-F1 |
|---|---|---|
| majority | 0.1111 | 0.0200 |
| random | 0.0970 | 0.0968 |
| HOG + logreg | 0.5570 | 0.5397 |
| **CNN** | **0.6899** | **0.6703** |

На val у CNN accuracy 0.6988 и macro-F1 0.6776 — то есть test и val
согласованы, разрыв мал.

### Per-class метрики CNN (test)

| Класс | precision | recall | F1 | support |
|---|---|---|---|---|
| AnnualCrop | 0.901 | 0.684 | 0.778 | 450 |
| Forest | 0.906 | 0.791 | 0.845 | 450 |
| HerbaceousVegetation | 0.787 | 0.393 | 0.524 | 450 |
| Highway | 0.418 | 0.285 | 0.339 | 375 |
| Industrial | 0.659 | 0.973 | 0.786 | 375 |
| Pasture | 0.632 | 0.780 | 0.699 | 300 |
| PermanentCrop | 0.676 | 0.384 | 0.490 | 375 |
| Residential | 0.659 | 0.904 | 0.762 | 450 |
| River | 0.479 | 0.819 | 0.604 | 375 |
| SeaLake | 0.888 | 0.864 | 0.876 | 450 |

### Per-class F1: CNN против HOG + logreg

| Класс | CNN | HOG + logreg | разница |
|---|---|---|---|
| AnnualCrop | 0.778 | 0.518 | +0.260 |
| Forest | 0.845 | 0.758 | +0.087 |
| HerbaceousVegetation | 0.524 | 0.408 | +0.116 |
| Highway | 0.339 | 0.558 | −0.219 |
| Industrial | 0.786 | 0.575 | +0.211 |
| Pasture | 0.699 | 0.270 | +0.429 |
| PermanentCrop | 0.490 | 0.272 | +0.218 |
| Residential | 0.762 | 0.643 | +0.119 |
| River | 0.604 | 0.489 | +0.115 |
| SeaLake | 0.876 | 0.905 | −0.029 |

## 5. Ошибки: что видно по confusion matrix

Матрица ошибок — `reports/figures/04_confusion_matrix.png` (10 × 10, `Blues`).

* `Highway` — самый слабый класс: 126 из 375 уходит в `River`, 56 в
  `Industrial`, 55 в `Residential`. Причина видна по данным: шоссе и реки —
  вытянутые линейные структуры, а у нас всего 3 свёрточных блока и
  разрешение 64 × 64; модель уверенно путает «дорогу» с «водой».
* `PermanentCrop` распадается на `Highway` (67), `Industrial` (41),
  `HerbaceousVegetation` (40) и `Residential` (39): класс визуально
  неоднороден (поля, сады, теплицы).
* `HerbaceousVegetation` теряет 99 снимков в `Residential` и 50 в `Pasture` —
  зелёные и серо-зелёные текстуры при 64 × 64 похожи.
* Сильные стороны: `SeaLake` (F1 0.876), `Forest` (0.845), `Industrial`
  (recall 0.973), `Residential` (recall 0.904).

## 6. Baseline

* `majority` — самый частый класс train (AnnualCrop, 2100 из 18 900);
  accuracy 0.1111, macro-F1 0.0200.
* `random` — равномерные случайные метки, `default_rng(42)`;
  accuracy 0.0970, macro-F1 0.0968.
* `HOG + logreg` — HOG по grayscale (orientations=9, ячейки 8 × 8,
  блоки 2 × 2, `L2-Hys`) даёт **1764** признака; `StandardScaler` +
  `LogisticRegression`. Гиперпараметр `C` выбран по val macro-F1:
  `C=0.1` → 0.5479, `C=1.0` → 0.5119, взят **C=0.1**.
  Test: accuracy 0.5570, macro-F1 0.5397. Обучение — только на train,
  test не использовался ни для выбора `C`, ни для калибровки.

## 7. Выводы и ограничения

1. CNN обходит сильный baseline HOG + logreg на **+13.3 п.п.** accuracy и
   **+13.1 п.п.** macro-F1; majority и random предсказуемо проваливаются.
2. Прирост неравномерен: максимум — `Pasture` (+0.429 F1) и `AnnualCrop`
   (+0.260), а на `Highway` CNN **хуже** HOG (−0.219) и на `SeaLake`
   чуть хуже (−0.029).
3. **Главное ограничение честно:** ранняя остановка произошла на 5-й эпохе,
   train accuracy в этот момент ещё росла (0.4896 → 0.7020 без плато).
   Модель недообучена: 3 свёрточных блока по 16/32/64 фильтра — очень
   маленькая сеть, а `patience=3` по шумному `val_loss` обрывает обучение
   раньше, чем оно успевает выйти на плато. Результат 0.69 — это результат
   протокола, а не потолка архитектуры. Это проверено контрольным прогоном
   в §9: он дал **0.9079 / 0.9055**.
4. ROC-AUC не считается: ТЗ этапа требует accuracy, macro-F1, per-class
   precision/recall/F1 и confusion matrix.
5. Один прогон, `seed=42`; доверительные интервалы и повторные запуски не
   заявлены.

## 8. Артефакты и воспроизведение

```powershell
python scripts\run_baselines.py    # majority, random, HOG + logreg
python scripts\train_cnn.py        # CNN + metrics_eurosat.json + eurosat_cnn.keras
python scripts\make_model_figures.py
python scripts\train_cnn_patience_max.py   # контрольный прогон (50 эпох)
python scripts\make_control_figures.py
```

* `reports/metrics_eurosat.json` — метрики test/val, per-class, confusion
  matrix, history, `epochs_ran`, `early_stopping_epoch`, индексы первых
  30 ошибок.
* `reports/baseline_metrics.json` — три baseline, `hog_logreg_C`,
  `hog_logreg_val_macro_f1`.
* `reports/summary_baselines.json` — accuracy/macro-F1 всех четырёх моделей.
* `reports/eurosat_cnn.keras` — веса обученной модели.
* `reports/figures/04_confusion_matrix.png`, `05_training_curves.png`,
  `06_per_class_f1.png`, `07_baseline_comparison.png`.

## 9. Контрольный прогон: `patience=max`

Отдельный эксперимент: меняются **только** `patience` (3 → 50) и `epochs`
(20 → 50). Архитектура, `batch_size=64`, `lr=1e-3`, `Adam`, `seed=42`,
split, нормализация и `augmentation` — те же. `EarlyStopping` с
`patience=50` при `epochs=50` не срабатывает; `restore_best_weights=True`
и `ModelCheckpoint(save_best_only=True)` возвращают веса эпохи с лучшим
`val_loss`. Тест открывался один раз, после обучения.

### Результат

| Прогон | accuracy | macro-F1 | epochs | patience | best_epoch |
|---|---|---|---|---|---|
| основной | 0.6899 | 0.6703 | 5 | 3 | 2 |
| контрольный | **0.9079** | **0.9055** | 50 | 50 | 26 |

Прирост: **+0.2180 accuracy** и **+0.2352 macro-F1**. `val`: 0.9096 / 0.9065 —
test и val согласованы. Время обучения 1271.3 с (~21 мин на CPU),
`early_stopping_triggered=False`.

`val_loss` вышел на минимум 0.2520 на 26-й эпохе, после чего колебался
(0.27–0.62) при монотонном росте train accuracy до 0.9089 к 50-й эпохе.
То есть 26-я эпоха — не случайная точка: и `val_loss`, и `val_accuracy`
на ней лучшие.

### Per-class метрики контрольного прогона

| Класс | precision | recall | F1 | support |
|---|---|---|---|---|
| AnnualCrop | 0.939 | 0.887 | 0.912 | 450 |
| Forest | 0.945 | 0.962 | 0.954 | 450 |
| HerbaceousVegetation | 0.747 | 0.940 | 0.833 | 450 |
| Highway | 0.932 | 0.880 | 0.905 | 375 |
| Industrial | 0.910 | 0.971 | 0.939 | 375 |
| Pasture | 0.939 | 0.827 | 0.879 | 300 |
| PermanentCrop | 0.868 | 0.739 | 0.798 | 375 |
| Residential | 0.964 | 0.962 | 0.963 | 450 |
| River | 0.891 | 0.891 | 0.891 | 375 |
| SeaLake | 0.991 | 0.969 | 0.980 | 450 |

Слабейший класс теперь `PermanentCrop` (F1 0.798), а не `Highway` (0.905):
именно `Highway` дал максимальный прирост — с 0.339 до 0.905 F1. Остаточные
ошибки: `PermanentCrop` → `HerbaceousVegetation` (63 из 375) и `Pasture` →
`HerbaceousVegetation` (25), `HerbaceousVegetation` → `PermanentCrop` (16).

### Вывод

0.6899 — **следствие остановки, а не свойство архитектуры**: `patience=3` по
шумному `val_loss` обрывал обучение на 5-й эпохе, когда train accuracy была 0.70
и продолжала расти. При снятии ограничения та же сеть `16/32/64` выходит на
0.9079. Это не доказывает, что архитектура «способна на большее»: менялся только
протокол остановки. Контрольный прогон — дополнительный; основной результат не
заменён, оба сохранены отдельно.

### Артефакты контрольного прогона

* `reports/control_patience_max/metrics_eurosat_patience_max.json` —
  accuracy, macro-F1, per-class, confusion matrix, `history` (50 эпох),
  `best_epoch=26`, `total_epochs_run=50`, `early_stopping_triggered=false`,
  `training_time_sec`.
* `reports/control_patience_max/eurosat_cnn_patience_max.keras` — веса
  лучшей эпохи.
* `reports/control_patience_max/figures/` — `confusion_matrix_patience_max.png`,
  `training_curves_patience_max.png` (вертикальная линия на `best_epoch`),
  `per_class_f1_patience_max.png`.
* Базовые `reports/metrics_eurosat.json`, `reports/eurosat_cnn.keras` и
  `reports/figures/04–07` не изменены (md5 совпадают).
