"""Honest holdout evaluation: retrain the ensemble on a train split, predict on a held-out tail.

Holds out the last 14 days of each (match_id, seat_zone) series. The ensemble
is trained from scratch on the remaining 77 days per series to avoid leakage.

1. Forecast accuracy: WAPE / R^2 / MAE / RMSE against a DummyRegressor mean baseline.
2. Decision quality: for every holdout row, the optimizer's recommended price is
   scored against the data generator's true price response
   (`make_dataset.price_effect`). Because no other demand driver depends on
   price, revenue at price p is proportional to p * price_effect(p / base), so
   we can report the share of the best achievable revenue each recommendation
   earns. Zone capacity never binds in the synthetic data, so it is ignored.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config
from src.data.make_dataset import price_effect
from src.decision_engine.constants import ZONE_BASE_PRICES
from src.decision_engine.optimize import OptimizationEngine
from src.models.predict_demand import DemandModel
from src.models.train_demand_model import price_support, train_ensemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HOLDOUT_DAYS = 14


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), 1e-9))


def summarize(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'model': label,
        'WAPE': wape(y_true, y_pred),
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def true_revenue_share(prices: np.ndarray, zones: pd.Series) -> np.ndarray:
    """Share of the best achievable revenue each price earns under the generator's true demand curve."""
    ratio = prices / zones.map(ZONE_BASE_PRICES).to_numpy()
    grid = np.linspace(0.01, 3.0, 3000)
    best = (grid * price_effect(grid)).max()
    return ratio * price_effect(ratio) / best


def main() -> None:
    logging.info("Loading synthetic data from %s", config.SYNTHETIC_DATA_PATH)
    df = pd.read_csv(config.SYNTHETIC_DATA_PATH)

    is_holdout = df['days_until_match'] < HOLDOUT_DAYS
    train_df, test_df = df[~is_holdout].copy(), df[is_holdout].copy()
    logging.info("Train rows: %d, holdout rows: %d", len(train_df), len(test_df))

    y_test = test_df[config.TARGET_COLUMN].values
    features_test = test_df.drop(columns=[config.TARGET_COLUMN])

    logging.info("Retraining ensemble on train split for unbiased evaluation...")
    prophet_bundle, feature_pipeline, xgb = train_ensemble(train_df)
    eval_model = DemandModel(
        prophet_bundle=prophet_bundle,
        feature_pipeline=feature_pipeline,
        xgb=xgb,
        price_support=price_support(train_df),
    )
    y_pred = eval_model.predict(features_test)

    baseline = DummyRegressor(strategy='mean')
    baseline.fit(train_df.drop(columns=[config.TARGET_COLUMN]), train_df[config.TARGET_COLUMN])
    y_baseline = baseline.predict(features_test)

    rows = [
        summarize('Ensemble (Prophet + XGBoost)', y_test, y_pred),
        summarize('Baseline (DummyRegressor mean)', y_test, y_baseline),
    ]
    results = pd.DataFrame(rows)
    print("\n--- Holdout Evaluation (last 14 days per series, no leakage) ---")
    print(results.to_string(index=False, formatters={
        'WAPE': '{:.1%}'.format,
        'R2': '{:.3f}'.format,
        'MAE': '{:.1f}'.format,
        'RMSE': '{:.1f}'.format,
    }))
    lift = (rows[1]['WAPE'] - rows[0]['WAPE']) / max(rows[1]['WAPE'], 1e-9)
    print(f"\nEnsemble WAPE is {lift:.0%} lower than baseline.")
    print("-----------------------------------------------------------------\n")

    logging.info("Scoring the optimizer's recommendations on %d holdout rows...", len(features_test))
    engine = OptimizationEngine(model=eval_model)
    recommended = engine.recommend_prices(features_test)
    zones = features_test['seat_zone'].reset_index(drop=True)
    share = true_revenue_share(recommended, zones)
    ratio = recommended / zones.map(ZONE_BASE_PRICES).to_numpy()
    bands = zones.map(eval_model.price_support)
    at_edge = np.array([p <= lo + 5 or p >= hi - 5 for p, (lo, hi) in zip(recommended, bands)])
    print("--- Decision quality (holdout rows, vs. the generator's true demand curve) ---")
    print(f"Share of best achievable revenue: {share.mean():.0%} (mean), {np.median(share):.0%} (median)")
    print(f"Median recommended price / base price: {np.median(ratio):.2f}x (true optimum: 1.25x)")
    print(f"Recommendations within one step of the band edge: {at_edge.mean():.0%}")
    by_zone = pd.DataFrame({'zone': zones, 'share': share}).groupby('zone')['share'].mean()
    print("By zone: " + ", ".join(f"{z} {v:.0%}" for z, v in by_zone.items()))
    print("------------------------------------------------------------------------------\n")


if __name__ == '__main__':
    main()
