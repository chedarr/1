# -*- coding: utf-8 -*-
"""Проверка целостности EuroSAT: архив, JPEG, дубликаты, TFDS."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

ZIP = config.DATA_RAW / "EuroSAT_RGB.zip"
EXTRACT = config.DATA_RAW / "EuroSAT_RGB"
EXPECTED_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    result: dict = {"zip": {}, "jpeg": {}, "duplicates": {}, "tfds": {}}

    result["zip"] = {
        "size": ZIP.stat().st_size,
        "md5": md5_of(ZIP),
        "md5_expected": EXPECTED_MD5,
    }

    files = sorted(EXTRACT.glob("*/*.jpg"))
    per_class = Counter(p.parent.name for p in files)
    bad = []
    sizes = Counter()
    hashes: dict[str, list[str]] = {}
    for p in files:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if im.size != (64, 64) or im.mode != "RGB":
                    bad.append([p.name, list(im.size), im.mode])
        except Exception as exc:  # noqa: BLE001
            bad.append([p.name, type(exc).__name__, str(exc)[:80]])
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(p.name)
        sizes[p.stat().st_size] += 1

    dup_groups = [v for v in hashes.values() if len(v) > 1]
    result["jpeg"] = {
        "n_files": len(files),
        "per_class": dict(per_class),
        "bad": bad[:20],
        "n_bad": len(bad),
        "unique_hashes": len(hashes),
    }
    result["duplicates"] = {
        "n_groups": len(dup_groups),
        "n_extra": sum(len(g) - 1 for g in dup_groups),
        "examples": dup_groups[:5],
    }

    import tensorflow_datasets as tfds

    builder = tfds.builder(config.TFDS_NAME, data_dir=str(config.DATA_RAW))
    result["tfds"] = {
        "name": builder.info.name,
        "version": str(builder.info.version),
        "num_examples": builder.info.splits["train"].num_examples,
        "n_classes": builder.info.features["label"].num_classes,
    }

    print(json.dumps(result, ensure_ascii=False, indent=1))
    ok = (
        result["zip"]["md5"] == EXPECTED_MD5
        and len(files) == 27000
        and len(bad) == 0
        and len(dup_groups) == 0
        and result["tfds"]["num_examples"] == 27000
        and result["tfds"]["n_classes"] == 10
    )
    print("ИТОГ:", "ЦЕЛОСТНОСТЬ OK" if ok else "ЕСТЬ ПРОБЛЕМЫ")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
