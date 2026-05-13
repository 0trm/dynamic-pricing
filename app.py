"""Streamlit HiTL page: pick a match/zone, see the recommended price, approve it."""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src.decision_engine.optimize import OptimizationEngine, default_price_range_for
from src.decision_engine.simulate import SimulationEngine
from src.models.predict_demand import DemandModel

PROPOSALS_PATH = Path(config.BASE_DIR) / 'proposals.jsonl'

ARTIFACTS = [
    config.PROPHET_MODELS_PATH,
    config.XGB_RESIDUAL_MODEL_PATH,
    config.FEATURE_PIPELINE_PATH,
]

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")
st.title("Dynamic Pricing Engine")
st.caption("Demand forecast + price optimization with a human-in-the-loop approval step.")


def _missing_artifacts() -> bool:
    return not all(os.path.exists(p) for p in ARTIFACTS) or not os.path.exists(config.SYNTHETIC_DATA_PATH)


@st.cache_resource(show_spinner=False)
def bootstrap_pipeline():
    """Generate data + train the ensemble on first run (Streamlit Cloud has no joblibs)."""
    if not _missing_artifacts():
        return
    from src.data import make_dataset
    from src.models import train_demand_model
    with st.spinner("First-time setup: generating synthetic data and training the ensemble (~30s)..."):
        if not os.path.exists(config.SYNTHETIC_DATA_PATH):
            make_dataset.main()
        train_demand_model.main()


@st.cache_resource
def load_engines():
    model = DemandModel.load()
    return SimulationEngine(model=model), OptimizationEngine(model=model)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(config.SYNTHETIC_DATA_PATH)


bootstrap_pipeline()

try:
    sim_engine, opt_engine = load_engines()
except FileNotFoundError:
    st.error("Trained models not found. Run `make all` first.")
    st.stop()

df = load_data()

with st.sidebar:
    st.header("Match selector")
    match_id = st.selectbox("Match ID", sorted(df['match_id'].unique()))
    match_df = df[df['match_id'] == match_id]
    seat_zone = st.selectbox("Seat zone", sorted(match_df['seat_zone'].unique()))
    days_until_match = st.slider(
        "Days until match",
        int(match_df['days_until_match'].min()),
        int(match_df['days_until_match'].max()),
        value=15,
    )

context = match_df[
    (match_df['seat_zone'] == seat_zone) & (match_df['days_until_match'] == days_until_match)
]
if context.empty:
    st.warning("No data row for this combination.")
    st.stop()

base_row = context.iloc[0].to_dict()
base_row.pop(config.TARGET_COLUMN, None)
base_features = pd.DataFrame([base_row])
current_price = float(base_row['ticket_price'])

st.subheader(f"Match {match_id} · {seat_zone} · {days_until_match} days out")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Current price", f"€{current_price:.2f}")
col_b.metric("Opponent tier", base_row['opponent_tier'])
col_c.metric("Weather forecast", base_row['weather_forecast'])

st.divider()
st.subheader("Revenue vs. price")
price_range = default_price_range_for(seat_zone)
curve = opt_engine.revenue_curve(base_features, price_range=price_range, step=5)
optimum = curve.loc[curve['projected_revenue'].idxmax()]

chart_df = curve.set_index('price')[['projected_revenue']]
st.line_chart(chart_df, height=280)

col1, col2, col3 = st.columns(3)
col1.metric("Recommended price", f"€{optimum['price']:.2f}",
            delta=f"€{optimum['price'] - current_price:+.2f} vs current")
col2.metric("Predicted sales", f"{int(optimum['predicted_sales'])} tickets")
col3.metric("Projected revenue", f"€{optimum['projected_revenue']:,.0f}")

st.divider()
st.subheader("What-if simulation")
hypothetical_price = st.slider(
    "Try a hypothetical price",
    min_value=int(price_range[0]),
    max_value=int(price_range[1]),
    value=int(optimum['price']),
    step=5,
)
sim = sim_engine.run_simulation(price=float(hypothetical_price), base_features=base_features)
sc1, sc2 = st.columns(2)
sc1.metric("Predicted sales", f"{sim['predicted_sales']} tickets")
sc2.metric("Projected revenue", f"€{sim['projected_revenue']:,.0f}")

st.divider()
if st.button("✓ Approve recommended price", type="primary"):
    proposal = {
        'timestamp': datetime.utcnow().isoformat(),
        'match_id': int(match_id),
        'seat_zone': seat_zone,
        'days_until_match': int(days_until_match),
        'current_price': current_price,
        'recommended_price': float(optimum['price']),
        'predicted_sales': int(optimum['predicted_sales']),
        'projected_revenue': float(optimum['projected_revenue']),
    }
    with PROPOSALS_PATH.open('a') as f:
        f.write(json.dumps(proposal) + '\n')
    st.success(f"Approved. Logged to {PROPOSALS_PATH.name}.")
