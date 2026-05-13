# Constants used by the decision engine and shared with the data generator.

# Base prices per zone. The data generator centers prices around these
# values; the optimizer derives its search bounds from them so the grid
# search stays inside the training distribution per zone.
ZONE_BASE_PRICES = {
    'VIP': 250,
    'Lateral': 120,
    'Corner': 90,
    'Gol Nord': 75,
    'Gol Sud': 75,
}

# Bounds (as multiples of base price) within which the historical data
# has meaningful coverage. Outside this band, XGBoost extrapolates and
# Prophet's residual model can't be trusted.
PRICE_SEARCH_RANGE_RATIO = (0.5, 2.5)


# A sample of base features for a single data point.
# Used by simulate.py / optimize.py as a hypothetical scenario.
# Schema mirrors the columns produced by src/data/make_dataset.py.
SAMPLE_BASE_FEATURES = {
    'match_id': 2,
    'days_until_match': 15,
    'seat_zone': 'Lateral',
    'opponent_tier': 'A',
    'ea_opponent_strength': 86,
    'is_weekday': False,
    'is_international': False,
    'top_player_injured': False,
    'league_winner_known': False,
    'team_position': 2,
    'is_holiday': False,
    'popular_concert_in_city': False,
    'weather_forecast': 'Sunny',
    'flights_to_barcelona_index': 110,
    'google_trends_index': 85,
    'internal_search_trends': 3500,
    'web_visits': 80000,
    'web_conversion_rate': 0.045,
    'social_media_sentiment': 0.8,
    'competitor_avg_price': 155.0,
    'zone_seats_availability': 4500,
    'ticket_availability_pct': 0.56,
}
