import numpy as np
import pytest

import config
from src.data.make_dataset import price_effect
from src.decision_engine.constants import ZONE_BASE_PRICES
from src.decision_engine.optimize import OptimizationEngine
from src.models.evaluate import true_revenue_share
from src.models.train_demand_model import price_support


def test_true_optimum_is_one_and_a_quarter_times_base():
    ratios = np.linspace(0.5, 2.5, 2001)
    assert ratios[np.argmax(ratios * price_effect(ratios))] == pytest.approx(1.25, abs=1e-3)


def test_price_support_is_inside_observed_prices(data):
    for zone, (lo, hi) in price_support(data).items():
        prices = data.loc[data['seat_zone'] == zone, 'ticket_price']
        assert prices.min() <= lo < hi <= prices.max() + 1


def test_recommendations_stay_inside_the_band(data, small_model):
    rows = data[data['match_id'].isin([1, 2, 3])].drop(columns=[config.TARGET_COLUMN]).sample(30, random_state=0)
    prices = OptimizationEngine(model=small_model).recommend_prices(rows)
    for price, zone in zip(prices, rows['seat_zone']):
        lo, hi = small_model.price_support[zone]
        assert lo <= price <= hi


def test_batch_recommendation_matches_single_row_curve(data, small_model):
    engine = OptimizationEngine(model=small_model)
    row = data[data['match_id'] == 2].drop(columns=[config.TARGET_COLUMN]).iloc[[10]].reset_index(drop=True)
    curve = engine.revenue_curve(row)
    assert engine.recommend_prices(row)[0] == curve.loc[curve['projected_revenue'].idxmax(), 'price']


def test_unseen_match_uses_the_pooled_model(data, small_model):
    unseen = data[data['match_id'] == 9].drop(columns=[config.TARGET_COLUMN])
    with_fallback = small_model.predict(unseen)
    pooled = np.clip(small_model.pooled_xgb.predict(small_model.feature_pipeline.transform(unseen)), 0, None)
    np.testing.assert_allclose(with_fallback, pooled)


def test_true_revenue_share_is_one_at_the_optimum(data):
    zones = data['seat_zone'].drop_duplicates().reset_index(drop=True)
    base = zones.map(ZONE_BASE_PRICES).to_numpy()
    np.testing.assert_allclose(true_revenue_share(1.25 * base, zones), 1.0, atol=1e-3)


def test_save_and_load_round_trip(tmp_path, data, small_model):
    small_model.save(str(tmp_path))
    loaded = type(small_model).load(str(tmp_path))
    rows = data[data['match_id'] == 1].drop(columns=[config.TARGET_COLUMN]).head(20)
    np.testing.assert_allclose(loaded.predict(rows), small_model.predict(rows))
    assert loaded.price_support == small_model.price_support
