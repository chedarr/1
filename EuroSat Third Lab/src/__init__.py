# -*- coding: utf-8 -*-
"""Пакет обработки данных EuroSAT."""
import random

import numpy as np

from src.config import SEED, configure_runtime

random.seed(SEED)
np.random.seed(SEED)

__all__ = ["configure_runtime", "SEED"]
