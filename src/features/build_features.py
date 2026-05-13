"""Feature pipeline factory.

Single source of truth for which columns the model sees and how they are
encoded. `train_demand_model.py` fits and persists the pipeline; nothing
else should call `fit` on a fresh instance.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that pass through `feature_pipeline.transform` raw.
# - `match_id`: series identity (Prophet already handles it); including it
#   would let XGBoost memorize series instead of learning generalizable features.
# - `web_conversion_rate`: defined in the data generator as sales/web_visits,
#   which is target leakage at decision time. At the moment we propose a
#   new price, the conversion rate at that price is what we are predicting.
COLUMNS_TO_DROP = ['match_id', 'web_conversion_rate']

NUMERICAL_FEATURES = [
    'days_until_match',
    'ticket_price',
    'ea_opponent_strength',
    'web_visits',
    'social_media_sentiment',
    'google_trends_index',
    'internal_search_trends',
    'flights_to_barcelona_index',
    'competitor_avg_price',
    'zone_seats_availability',
    'ticket_availability_pct',
    'team_position',
]

CATEGORICAL_FEATURES = [
    'seat_zone',
    'opponent_tier',
    'weather_forecast',
]


def build_feature_pipeline() -> Pipeline:
    """Return an unfitted ColumnTransformer that drops match_id, scales numerics, one-hot encodes categoricals."""
    return ColumnTransformer(
        transformers=[
            ('drop', 'drop', COLUMNS_TO_DROP),
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ],
        remainder='passthrough',  # boolean flags (is_holiday, is_weekday, etc.) pass through
    )
