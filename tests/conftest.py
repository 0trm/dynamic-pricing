import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.models.train_demand_model import train_ensemble


@pytest.fixture(scope='session')
def data() -> pd.DataFrame:
    return pd.read_csv(config.SYNTHETIC_DATA_PATH)


@pytest.fixture(scope='session')
def small_model(data):
    """Ensemble trained on three matches: enough series to exercise every path, fast enough for CI."""
    return train_ensemble(data[data['match_id'].isin([1, 2, 3])])
