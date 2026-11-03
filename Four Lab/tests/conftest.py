# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def frame() -> pd.DataFrame:
    from src import config

    index = pd.date_range("2020-01-01", periods=40, freq="10min")
    values = {name: [1.0 + i * 0.01 for i in range(40)] for name in config.FEATURE_COLUMNS}
    values[config.TARGET_FEATURE][10] = float("nan")
    values[config.TARGET_FEATURE][11] = float("nan")
    return pd.DataFrame(values, index=index)
