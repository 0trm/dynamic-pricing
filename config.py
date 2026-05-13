import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data
DATA_DIR = os.path.join(BASE_DIR, 'data')
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, '03_synthetic', 'synthetic_match_data.csv')

# Models
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROPHET_MODELS_PATH = os.path.join(MODELS_DIR, 'prophet_models.joblib')
XGB_RESIDUAL_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_residual_model.joblib')
FEATURE_PIPELINE_PATH = os.path.join(MODELS_DIR, 'feature_pipeline.joblib')

# Reference date used to map (match_id, days_until_match) -> a calendar date
# so Prophet sees a meaningful seasonal axis. Deterministic and embedded
# in the persisted prophet_models bundle so predictions stay aligned.
REFERENCE_MATCH_DATE = '2025-01-01'
DAYS_BETWEEN_MATCHES = 30

TARGET_COLUMN = 'zone_historical_sales'
