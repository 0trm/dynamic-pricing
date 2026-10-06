"""Revenue-maximizing price via grid search over the DemandModel."""

import logging
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.decision_engine.constants import SAMPLE_BASE_FEATURES
from src.models.predict_demand import DemandModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OptimizationEngine:
    def __init__(self, model: DemandModel | None = None):
        self.model = model or DemandModel.load()

    def price_range_for(self, seat_zone: str) -> tuple[int, int]:
        """Prices the zone was actually sold at in training (p5-p95); the model is not trusted outside them."""
        return self.model.price_support[seat_zone]

    def revenue_curve(
        self,
        base_features: pd.DataFrame,
        price_range: tuple[int, int] | None = None,
        step: int = 5,
    ) -> pd.DataFrame:
        """Vectorized: build one batch of candidate-price rows, predict once, return prices × revenue table."""
        base_row = base_features.iloc[0].to_dict()
        if price_range is None:
            price_range = self.price_range_for(base_row['seat_zone'])
        lo, hi = price_range
        prices = np.arange(lo, hi + 1, step)

        batch = pd.DataFrame([{**base_row, 'ticket_price': float(p)} for p in prices])
        sales = self.model.predict(batch)
        # Revenue from unrounded sales: in low-volume zones (VIP sells 0-3 a day) rounding
        # flattens the curve to zero and the argmax falls back to the lowest price.
        revenue = prices * sales
        return pd.DataFrame({'price': prices, 'predicted_sales': np.round(sales).astype(int), 'projected_revenue': revenue})

    def recommend_prices(self, features: pd.DataFrame, step: int = 5) -> np.ndarray:
        """Revenue-maximizing price for every row, in one prediction batch (used by evaluate.py)."""
        rows = features.reset_index(drop=True)
        candidates = []
        for i, zone in rows['seat_zone'].items():
            lo, hi = self.price_range_for(zone)
            candidates.append(pd.DataFrame({'row': i, 'ticket_price': np.arange(lo, hi + 1, step, dtype=float)}))
        grid = pd.concat(candidates, ignore_index=True)
        batch = rows.drop(columns=['ticket_price']).loc[grid['row']].reset_index(drop=True)
        batch['ticket_price'] = grid['ticket_price']
        batch = batch[rows.columns]
        grid['revenue'] = grid['ticket_price'] * self.model.predict(batch)
        best = grid.loc[grid.groupby('row')['revenue'].idxmax()]
        return best.sort_values('row')['ticket_price'].to_numpy()

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
