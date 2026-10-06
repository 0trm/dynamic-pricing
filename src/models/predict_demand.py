"""DemandModel: unified prediction surface over Prophet + XGBoost residuals.

Series with sales history (a match-zone already on sale) get Prophet's
forecast plus the XGBoost residual. Series without one (a match that just
went on sale) get the pooled XGBoost model trained directly on sales: per-series
Prophet has nothing to say about a match it has never seen.

Used by simulate.py, optimize.py, and the Streamlit app. The trained
artifacts are produced by `train_demand_model.py` via `DemandModel.save()`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config

logger = logging.getLogger(__name__)


@dataclass
class DemandModel:
    prophet_bundle: dict
    feature_pipeline: object
    xgb: object
    pooled_xgb: object | None = None  # XGBoost on sales, for series with no history (cold start)
    price_support: dict[str, tuple[int, int]] | None = None  # zone -> (lo, hi) prices seen in training

    @classmethod
    def load(cls, models_dir: str = config.MODELS_DIR) -> 'DemandModel':
        with open(os.path.join(models_dir, os.path.basename(config.PRICE_SUPPORT_PATH))) as f:
            support = {zone: tuple(band) for zone, band in json.load(f).items()}
        artifact = lambda path: joblib.load(os.path.join(models_dir, os.path.basename(path)))
        return cls(
            prophet_bundle=artifact(config.PROPHET_MODELS_PATH),
            feature_pipeline=artifact(config.FEATURE_PIPELINE_PATH),
            xgb=artifact(config.XGB_RESIDUAL_MODEL_PATH),
            pooled_xgb=artifact(config.POOLED_XGB_PATH),
            price_support=support,
        )

    def save(self, models_dir: str = config.MODELS_DIR) -> None:
        os.makedirs(models_dir, exist_ok=True)
        artifact = lambda path: os.path.join(models_dir, os.path.basename(path))
        joblib.dump(self.prophet_bundle, artifact(config.PROPHET_MODELS_PATH))
        joblib.dump(self.feature_pipeline, artifact(config.FEATURE_PIPELINE_PATH))
        joblib.dump(self.xgb, artifact(config.XGB_RESIDUAL_MODEL_PATH))
        joblib.dump(self.pooled_xgb, artifact(config.POOLED_XGB_PATH))
        with open(artifact(config.PRICE_SUPPORT_PATH), 'w') as f:
            json.dump(self.price_support, f, indent=2)

    def _match_date(self, match_id: int) -> pd.Timestamp:
        ref = pd.Timestamp(self.prophet_bundle['reference_match_date'])
        step = self.prophet_bundle['days_between_matches']
        return ref + pd.Timedelta(days=step * (int(match_id) - 1))

    def _prophet_forecast(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prophet yhat per row, and a mask of rows whose series has a Prophet model."""
        regressors = self.prophet_bundle['regressors']
        series_models = self.prophet_bundle['series_models']
        out = np.zeros(len(features), dtype=float)
        has_history = np.zeros(len(features), dtype=bool)

        for (match_id, seat_zone), idx in features.groupby(['match_id', 'seat_zone']).groups.items():
            model = series_models.get((int(match_id), str(seat_zone)))
            if model is None:
                continue
            rows = features.loc[idx]
            ds = self._match_date(match_id) - pd.to_timedelta(rows['days_until_match'].astype(int), unit='D')
            prophet_input = pd.DataFrame({'ds': ds.values})
            for reg in regressors:
                prophet_input[reg] = rows[reg].values
            positions = features.index.get_indexer(idx)
            out[positions] = model.predict(prophet_input)['yhat'].clip(lower=0).values
            has_history[positions] = True

        return out, has_history

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if features.empty:
            return np.array([])
        features = features.reset_index(drop=True)
        prophet_yhat, has_history = self._prophet_forecast(features)
        X = self.feature_pipeline.transform(features)
        prediction = prophet_yhat + self.xgb.predict(X)
        if not has_history.all():
            if self.pooled_xgb is None:
                logger.warning("%d rows have no Prophet series and no pooled model; using the residual alone.",
                               (~has_history).sum())
            else:
                prediction[~has_history] = self.pooled_xgb.predict(X[~has_history])
        return np.clip(prediction, a_min=0, a_max=None)
