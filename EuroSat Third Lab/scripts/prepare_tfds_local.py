# -*- coding: utf-8 -*-
"""Локальная сборка TFDS eurosat/rgb из уже скачанного архива.

Официальный билдер качает EuroSAT.zip с madm.dfki.de, который недоступен.
Архив Zenodo (EuroSAT_RGB.zip) распаковывается локально, после чего
_split_generators подменяется на локальный путь — downloader не вызывается.
"""
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow_datasets as tfds  # noqa: E402

from src import config  # noqa: E402


def main() -> int:
    zip_path = config.DATA_RAW / "EuroSAT_RGB.zip"
    extract_dir = config.DATA_RAW / "EuroSAT_RGB"

    if not zip_path.exists():
        print(f"[ERROR] нет архива: {zip_path}")
        return 1

    already = list(extract_dir.glob("*/*.jpg")) if extract_dir.exists() else []
    if len(already) < 1000:
        print(f"[..] распаковка {zip_path.name} -> {config.DATA_RAW}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(config.DATA_RAW)
        print(f"[OK] распаковано, jpg: {len(list(extract_dir.glob('*/*.jpg')))}")
    else:
        print(f"[OK] уже распаковано, jpg: {len(already)}")

    builder = tfds.builder(config.TFDS_NAME, data_dir=str(config.DATA_RAW))

    def _local_splits(dl_manager):
        return [
            tfds.core.SplitGenerator(
                name="train",
                gen_kwargs={"path": str(extract_dir), "selection": "rgb"},
            )
        ]

    builder._split_generators = _local_splits
    builder.download_and_prepare(download_dir=str(config.DATA_RAW))

    info = builder.info
    print("name:", info.name)
    print("version:", info.version)
    print("splits:", {k: v.num_examples for k, v in info.splits.items()})
    print("classes:", info.features["label"].names)
    print("shape:", info.features["image"].shape)
    print("dtype:", info.features["image"].dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
