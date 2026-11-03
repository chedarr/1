# -*- coding: utf-8 -*-
"""Однократное скачивание TFDS eurosat/rgb в локальный кэш проекта.

После этого скрипта все остальные модули используют download=False.
"""
import sys
from pathlib import Path

import tensorflow_datasets as tfds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW, TFDS_NAME


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"кэш TFDS: {DATA_RAW}")
    builder = tfds.builder(TFDS_NAME, data_dir=str(DATA_RAW))
    builder.download_and_prepare(download_dir=str(DATA_RAW))
    info = builder.info
    print(f"название: {info.name}")
    print(f"версия: {info.version}")
    print(f"снимков в train: {info.splits['train'].num_examples}")
    print(f"классов: {info.features['label'].num_classes}")
    print(f"имена классов: {info.features['label'].names}")
    print(f"форма снимка: {info.features['image'].shape}")
    print(f"dtype: {info.features['image'].dtype}")
    print(f"лицензия: {info.license}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
