"""Revenue-maximizing price via grid search over the DemandModel."""

import logging
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.decision_engine.constants import (
    PRICE_SEARCH_RANGE_RATIO,
    SAMPLE_BASE_FEATURES,
    ZONE_BASE_PRICES,
)
from src.models.predict_demand import DemandModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def default_price_range_for(seat_zone: str) -> tuple[int, int]:
    """Bounds derived from the zone's base price and the training distribution band."""
    base = ZONE_BASE_PRICES[seat_zone]
    lo, hi = PRICE_SEARCH_RANGE_RATIO
    return int(round(base * lo)), int(round(base * hi))


class OptimizationEngine:
    def __init__(self, model: DemandModel | None = None):
        self.model = model or DemandModel.load()

    def revenue_curve(
        self,
        base_features: pd.DataFrame,
        price_range: tuple[int, int] | None = None,
        step: int = 5,
    ) -> pd.DataFrame:
        """Vectorized: build one batch of candidate-price rows, predict once, return prices × revenue table."""
        base_row = base_features.iloc[0].to_dict()
        if price_range is None:
            price_range = default_price_range_for(base_row['seat_zone'])
        lo, hi = price_range
        prices = np.arange(lo, hi + 1, step)

        batch = pd.DataFrame([{**base_row, 'ticket_price': float(p)} for p in prices])
        sales = self.model.predict(batch)
        sales = np.maximum(0, np.round(sales)).astype(int)
        revenue = prices * sales
        return pd.DataFrame({'price': prices, 'predicted_sales': sales, 'projected_revenue': revenue})

    def run_optimization(
        self,
        base_features: pd.DataFrame,
        price_range: tuple[int, int] | None = None,
        step: int = 5,
    ) -> tuple[float, float]:
        curve = self.revenue_curve(base_features, price_range, step)
        best = curve.loc[curve['projected_revenue'].idxmax()]
        return float(best['price']), float(best['projected_revenue'])


if __name__ == '__main__':
    engine = OptimizationEngine()
    features_df = pd.DataFrame([SAMPLE_BASE_FEATURES])
    optimal_price, max_revenue = engine.run_optimization(base_features=features_df)
    print("\n--- Optimization Result ---")
    print(f"Optimal Price Recommendation: €{optimal_price:.2f}")
    print(f"Maximum Estimated Revenue: €{max_revenue:,.2f}")
    print("---------------------------\n")
