"""DemandModel: unified prediction surface over Prophet + XGBoost residuals.

Used by simulate.py, optimize.py, and the Streamlit app. The trained
artifacts (`prophet_models.joblib`, `feature_pipeline.joblib`,
`xgb_residual_model.joblib`) are produced by `train_demand_model.py`.
"""

from __future__ import annotations

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

    @classmethod
    def load(
        cls,
        prophet_path: str = config.PROPHET_MODELS_PATH,
        pipeline_path: str = config.FEATURE_PIPELINE_PATH,
        xgb_path: str = config.XGB_RESIDUAL_MODEL_PATH,
    ) -> 'DemandModel':
        return cls(
            prophet_bundle=joblib.load(prophet_path),
            feature_pipeline=joblib.load(pipeline_path),
            xgb=joblib.load(xgb_path),
        )

    def _match_date(self, match_id: int) -> pd.Timestamp:
        ref = pd.Timestamp(self.prophet_bundle['reference_match_date'])
        step = self.prophet_bundle['days_between_matches']
        return ref + pd.Timedelta(days=step * (int(match_id) - 1))

    def _prophet_forecast(self, features: pd.DataFrame) -> np.ndarray:
        regressors = self.prophet_bundle['regressors']
        series_models = self.prophet_bundle['series_models']
        out = np.zeros(len(features), dtype=float)

        for (match_id, seat_zone), idx in features.groupby(['match_id', 'seat_zone']).groups.items():
            key = (int(match_id), str(seat_zone))
            model = series_models.get(key)
            if model is None:
                # Series unseen in training (e.g. a skipped zone). Fall back to series-mean of 0;
                # XGBoost residual still captures most of the signal.
                logger.warning("No Prophet model for series %s. Using yhat=0.", key)
                continue
            rows = features.loc[idx]
            ds = self._match_date(match_id) - pd.to_timedelta(rows['days_until_match'].astype(int), unit='D')
            prophet_input = pd.DataFrame({'ds': ds.values})
            for reg in regressors:
                prophet_input[reg] = rows[reg].values
            yhat = model.predict(prophet_input)['yhat'].clip(lower=0).values
            out[features.index.get_indexer(idx)] = yhat

        return out

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if features.empty:
            return np.array([])
        features = features.reset_index(drop=True)
        prophet_yhat = self._prophet_forecast(features)
        X = self.feature_pipeline.transform(features)
        residual = self.xgb.predict(X)
        prediction = prophet_yhat + residual
        return np.clip(prediction, a_min=0, a_max=None)
