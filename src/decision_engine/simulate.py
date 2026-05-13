"""Simulate the impact of a hypothetical price on sales and revenue."""

import logging
import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.decision_engine.constants import SAMPLE_BASE_FEATURES
from src.models.predict_demand import DemandModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SimulationEngine:
    def __init__(self, model: DemandModel | None = None):
        self.model = model or DemandModel.load()

    def run_simulation(self, price: float, base_features: pd.DataFrame) -> dict:
        features = base_features.copy()
        features['ticket_price'] = price

        predicted_sales = int(max(0, round(self.model.predict(features)[0])))
        projected_revenue = price * predicted_sales

        return {
            'simulated_price': price,
            'predicted_sales': predicted_sales,
            'projected_revenue': projected_revenue,
        }


if __name__ == '__main__':
    engine = SimulationEngine()
    features_df = pd.DataFrame([SAMPLE_BASE_FEATURES])
    result = engine.run_simulation(price=120.00, base_features=features_df)
    print("\n--- Simulation Result ---")
    print(f"Simulated Price: €{result['simulated_price']:.2f}")
    print(f"Predicted Sales: {result['predicted_sales']} tickets")
    print(f"Projected Revenue: €{result['projected_revenue']:.2f}")
    print("-------------------------\n")
