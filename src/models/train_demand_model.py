"""Train the Prophet + XGBoost residual-fitting ensemble.

Stage A: one Prophet model per (match_id, seat_zone) series captures the
temporal core (trend, weekly seasonality, holidays, weekday effect).

Stage B: XGBoost is fit on Prophet's in-sample residuals using the full
feature set (price, demand signals, external factors). Final prediction
is Prophet yhat + XGBoost residual.
"""

import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config
from src.features.build_features import build_feature_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

PROPHET_REGRESSORS = ['is_weekday', 'is_holiday']


def match_date_for(match_id: int) -> pd.Timestamp:
    return pd.Timestamp(config.REFERENCE_MATCH_DATE) + pd.Timedelta(
        days=config.DAYS_BETWEEN_MATCHES * (int(match_id) - 1)
    )


def add_ds_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ds'] = df['match_id'].map(match_date_for) - pd.to_timedelta(df['days_until_match'], unit='D')
    return df


def fit_prophet_per_series(df: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Fit one Prophet model per (match_id, seat_zone). Return models and in-sample yhat aligned to df."""
    prophet_models: dict[tuple[int, str], Prophet] = {}
    yhat = pd.Series(index=df.index, dtype=float)

    for (match_id, seat_zone), series in df.groupby(['match_id', 'seat_zone']):
        # Prophet expects columns: ds, y. Regressors added via add_regressor.
        prophet_df = series[['ds', config.TARGET_COLUMN] + PROPHET_REGRESSORS].rename(
            columns={config.TARGET_COLUMN: 'y'}
        ).sort_values('ds')

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        for reg in PROPHET_REGRESSORS:
            model.add_regressor(reg)
        model.fit(prophet_df)

        forecast = model.predict(prophet_df[['ds'] + PROPHET_REGRESSORS])
        yhat.loc[series.index] = forecast['yhat'].clip(lower=0).values
        prophet_models[(int(match_id), str(seat_zone))] = model

    return prophet_models, yhat


def train_ensemble(df: pd.DataFrame) -> tuple[dict, object, XGBRegressor]:
    """Fit the full ensemble on `df`. Returns (prophet_bundle, feature_pipeline, xgb)."""
    df = add_ds_column(df)
    logging.info("Stage A: fitting Prophet on %d series", df.groupby(['match_id', 'seat_zone']).ngroups)
    prophet_models, prophet_yhat = fit_prophet_per_series(df)
    residuals = df[config.TARGET_COLUMN] - prophet_yhat
    logging.info("Prophet in-sample residual stats: mean=%.2f std=%.2f", residuals.mean(), residuals.std())

    logging.info("Stage B: fitting feature pipeline + XGBoost on residuals")
    feature_pipeline = build_feature_pipeline()
    X_raw = df.drop(columns=[config.TARGET_COLUMN, 'ds'])
    X = feature_pipeline.fit_transform(X_raw)

    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X, residuals.values)

    prophet_bundle = {
        'series_models': prophet_models,
        'regressors': PROPHET_REGRESSORS,
        'reference_match_date': config.REFERENCE_MATCH_DATE,
        'days_between_matches': config.DAYS_BETWEEN_MATCHES,
    }
    return prophet_bundle, feature_pipeline, xgb


def main() -> None:
    logging.info("Loading synthetic data from %s", config.SYNTHETIC_DATA_PATH)
    df = pd.read_csv(config.SYNTHETIC_DATA_PATH)
    prophet_bundle, feature_pipeline, xgb = train_ensemble(df)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(prophet_bundle, config.PROPHET_MODELS_PATH)
    joblib.dump(feature_pipeline, config.FEATURE_PIPELINE_PATH)
    joblib.dump(xgb, config.XGB_RESIDUAL_MODEL_PATH)
    logging.info("Saved Prophet bundle, feature pipeline, and XGBoost residual model to %s", config.MODELS_DIR)


if __name__ == '__main__':
    main()
