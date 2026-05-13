"""Price elasticity sanity check.

A demand model that ignores price would yield a meaningless optimizer
("just pick the price cap"). This script samples a handful of real
training rows and verifies that predicted sales are non-increasing as
price rises across the full search range.

Exits non-zero if more than a small fraction of rows violate monotonic
non-increase. Counted violations and the worst offender are printed.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import config
from src.decision_engine.optimize import OptimizationEngine
from src.models.predict_demand import DemandModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_SAMPLES = 20
PRICE_STEP = 5
# A row is meaningful if predicted sales actually move with price.
# "Flat" predictions are as bad as non-monotonic ones: the optimizer
# would just pick the price cap. Both failure modes need to be caught.
TOLERATED_PER_STEP_INCREASE = 2          # tickets (integer-rounding wiggle)
MIN_RELATIVE_SPREAD = 0.20               # (max - min) / max sales must exceed this
MAX_VIOLATION_FRACTION = 0.25            # at most 25% of sampled rows may violate either rule
MIN_NONZERO_SAMPLES = 10                 # ignore series with predicted sales ~0 everywhere


def evaluate_row(engine: OptimizationEngine, row: pd.DataFrame) -> dict:
    # Zone-aware default range from OptimizationEngine.
    curve = engine.revenue_curve(row, price_range=None, step=PRICE_STEP)
    sales = curve['predicted_sales'].values
    deltas = np.diff(sales)
    worst_increase = int(deltas.max()) if len(deltas) else 0
    spread = int(sales.max() - sales.min())
    relative_spread = spread / max(sales.max(), 1)
    monotonic_fail = worst_increase > TOLERATED_PER_STEP_INCREASE
    flat_fail = sales.max() >= MIN_NONZERO_SAMPLES and relative_spread < MIN_RELATIVE_SPREAD
    return {
        'min_sales': int(sales.min()),
        'max_sales': int(sales.max()),
        'spread': spread,
        'relative_spread': round(relative_spread, 2),
        'worst_per_step_increase': worst_increase,
        'violates': monotonic_fail or flat_fail,
        'optimal_price': float(curve.loc[curve['projected_revenue'].idxmax(), 'price']),
    }


def main() -> int:
    df = pd.read_csv(config.SYNTHETIC_DATA_PATH)
    df = df.drop(columns=[config.TARGET_COLUMN])

    rng = np.random.default_rng(seed=0)
    sample_idx = rng.choice(df.index, size=min(NUM_SAMPLES, len(df)), replace=False)
    sample = df.loc[sample_idx].reset_index(drop=True)

    engine = OptimizationEngine(model=DemandModel.load())

    results = []
    for i in range(len(sample)):
        row = sample.iloc[[i]].reset_index(drop=True)
        out = evaluate_row(engine, row)
        out.update({'match_id': int(row['match_id'].iloc[0]),
                    'seat_zone': row['seat_zone'].iloc[0],
                    'days_until_match': int(row['days_until_match'].iloc[0])})
        results.append(out)

    report = pd.DataFrame(results)
    violation_rate = report['violates'].mean()
    print("\n--- Elasticity sanity check ---")
    print(report.to_string(index=False))
    print(f"\nSampled rows: {len(report)}")
    print(f"Mean sales spread (max - min across zone-specific range): {report['spread'].mean():.1f} tickets")
    print(f"Median optimal price: €{report['optimal_price'].median():.0f}")
    print(f"Violation rate: {violation_rate:.0%}  (tolerated: <{MAX_VIOLATION_FRACTION:.0%})")
    print("-------------------------------\n")

    if violation_rate > MAX_VIOLATION_FRACTION:
        logging.error(
            "Elasticity check FAILED: too many rows have non-monotonic price-to-sales response. "
            "The optimizer's recommendations cannot be trusted."
        )
        return 1
    logging.info("Elasticity check PASSED.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
