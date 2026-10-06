# Dynamic pricing engine with HiTL

<p align="left">
  <a href="https://dynamic-pricing-0trm.streamlit.app/"><img src="https://img.shields.io/badge/Live%20demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white" alt="Live demo"></a>
  <img src="https://img.shields.io/badge/ML-Supervised-lightgrey" alt="ML">
  <img src="https://img.shields.io/badge/Models-Prophet%20%2B%20XGBoost-lightgrey" alt="Models">
</p>

An ML-powered dynamic pricing and decision support system for ticket pricing in a sports stadium. The engine forecasts demand at any candidate price, grid-searches for the revenue-maximizing recommendation, and surfaces it on a one-click approval dashboard. The same architecture was deployed on a real club's ticketing data and delivered **+6% revenue per match** and **86% recommendation adoption** by the commercial team.

<p align="center">
  <a href="https://dynamic-pricing-0trm.streamlit.app/"><img src="./assets/dp-app.png" alt="Streamlit HiTL page" width="900"></a>
  <br>
  <em>Pick a match and zone, see the recommended price and the revenue-vs-price curve, then approve. <a href="https://dynamic-pricing-0trm.streamlit.app/">▶ Try the live demo</a>.</em>
</p>

## Quickstart

```bash
pip install -r requirements.txt
make all          # synthetic data, train, holdout evaluation, elasticity sanity check
make app          # open the human-in-the-loop Streamlit page locally
```

---

## The problem

The challenge: transform a static, manual pricing strategy into a responsive, automated system with a human-in-the-loop, creating a market-driven approach to setting ticket prices per match. Prices were historically set once per season in rigid categories and updated weekly or monthly. The team would manually pull data from several systems, propose changes, then push them to the live ticketing platform by hand. The engine collapses that loop into "data → recommendation → one-click approval → live price".

<p align="center">
  <img src="./assets/dp-ss.jpeg" alt="Stadium ticketing price list" width="900">
  <br>
  <em>Fig. 1: A standard stadium ticket pricing by zone during the checkout process.</em>
</p>

<details>
<summary>Click for the full problem ↔ solution breakdown</summary>

| 🚩 Problem | 💡 Solution |
| :--- | :--- |
| **Static pricing**: prices set once per season in rigid categories (A++, A, B), updated weekly/monthly. | **Dynamic recommendations**: price proposals per seating zone based on near real-time data analysis, allowing daily updates. |
| **Manual adjustments**: slow analysis to propose changes. | **Impact simulation**: instantly model projected impact of any price change on revenue and ticket sales. |
| **Data bottleneck**: manual extraction from fragmented systems. | **Centralized data**: aggregates sales, web analytics, contextual data into one place. |
| **Slow implementation**: disconnected from the sales platform. | **Seamless integration**: one-click approval on a dashboard pushes a price update to the live ticketing system. |

</details>


## System at a glance

The **Dynamic Pricing Engine** ingests historical data from the **Club's Data Systems** and real-time sales from the **Ticketing System**, recommends prices that are simulated and approved by the **Club's Pricing Team**, and pushes approved prices back to the live ticketing system where fans complete a purchase. That last step closes the feedback loop.

<p align="center">
  <img src="./assets/dp-scd.png" alt="System Context Diagram" width="800">
  <br>
  <em>Fig. 2: System Context Diagram – Dynamic Pricing System.</em>
</p>

<details>
<summary>Click for the high-level market-dynamics view</summary>

<p align="center">
  <img src="./assets/dp-md.png" alt="High-level market dynamics diagram" width="1800">
  <br>
  <em>The engine acts as the central brain balancing the club and the fan, ingesting internal and external factors to forecast demand at various price points.</em>
</p>

</details>


## Results

### In this repo (synthetic data)

`make evaluate` runs a 14-day per-series holdout. Every model is re-trained from scratch on the train split before predicting on the holdout, so the numbers below are leakage-free. "MAE vs. naive" divides each MAE by the last-7-days naive's; below 1 beats it.

| Model | WAPE | R² | MAE | RMSE | MAE vs. naive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ensemble (Prophet + XGBoost)** | **26.4%** | **0.729** | 6.2 | **11.7** | 0.71 |
| Prophet only | 41.8% | 0.483 | 9.8 | 16.2 | 1.12 |
| XGBoost only (pooled, on sales) | **22.6%** | 0.717 | **5.3** | 12.0 | **0.60** |
| Naive: last 7 days per series | 37.4% | 0.544 | 8.7 | 15.2 | 1.00 |
| Naive: global mean | 79.7% | -0.428 | 18.6 | 27.0 | 2.13 |

The baseline that matters is the last-7-days naive: the forecast anyone could make at the cutoff without a model. The ensemble's WAPE is **29% lower** than it. The XGBoost stage does the heavy lifting: on its own, trained directly on sales, it beats the ensemble on WAPE and MAE, while the ensemble edges it on R² and RMSE (fewer large misses). Prophet alone is worse than the naive. Reproducible with `RANDOM_SEED=42` in `src/data/make_dataset.py`.

#### Decision quality

Forecast accuracy is not the goal; the price is. Because the data is synthetic, the true price response is known (`price_effect` in `src/data/make_dataset.py`, optimum at **1.25× base price**), so `make evaluate` also scores every holdout recommendation by the share of the best achievable revenue it earns:

| Metric | Value |
| :--- | :--- |
| **Share of best achievable revenue** | **95%** mean (Corner 98%, Gol Nord 97%, Gol Sud 97%, Lateral 94%, VIP 91%) |
| **Median recommended price** | 1.52× base (true optimum 1.25×) |
| **Recommendations at the band's upper edge** | 19% |

An earlier version searched `0.5×–2.5×` base price, but no training row is priced above 1.91× base. Past that point the trees predict flat sales, revenue keeps rising with price, and 42% of recommendations landed at the cap, earning **45%** of the best achievable revenue. The optimizer now searches only the prices each zone was actually sold at (5th–95th percentile, computed at training time). The remaining upper-edge share means the model still underestimates how fast demand falls at high prices.

`make sanity` guards against both failure modes: it samples 20 historical rows, sweeps each through the zone's band, and fails if predicted sales don't move with price (at least 20% relative spread, monotone within 2 tickets per step) or if more than 40% of optima sit at the band's upper edge. Latest run: **0% violations**, **35% at the upper edge**, **median optimal price €132**.

### In the real-world deployment

> ℹ️ These numbers are **not** reproduced by this repository – they come from a deployment on a confidential real-world dataset. Treat them as case-study evidence, not a benchmark.

| Metric | Result | Description |
| :--- | :--- | :--- |
| 📈 Revenue uplift | **+6%** avg. revenue per match | Achieved by dynamically adjusting prices to match real-time demand forecasts. Validated via A/B testing. |
| 🎟️ Optimized sales | **+4%** sell-through rate | Improved occupancy alongside revenue, which positively affects atmosphere and in-stadium sales. |
| ⚙️ Operational efficiency | **7×** faster price changes | From weekly to daily updates by automating data aggregation and analysis. |
| 🤝 Recommendation adoption | **86%** of proposals approved | Commercial team reviewed and approved the model's price proposals at a high rate, indicating trust. |

The engine was validated via **segment-based A/B tests**: a subset of seating zones used the dynamic engine (treatment), the rest stayed on static pricing (control). Tests ran across matches of varying importance to ensure the lift wasn't an artifact of any single event. The +6% revenue lift held alongside a +4% sell-through rate, confirming the engine found market equilibrium rather than simply over-charging.


## How the engine works

Two stages: first **predict**, then **optimize**.

<p align="left">
  <img src="./assets/dp-dpe.png" alt="Dynamic Pricing Engine" width="275">
  <br>
  <em>Fig. 3: Dynamic Pricing Engine component.</em>
</p>

### Stage 1 – Demand forecasting

A Prophet + XGBoost residual ensemble. One Prophet model per `(match_id, seat_zone)` series captures temporal structure (trend, weekly seasonality, weekday/holiday effects). A single XGBoost regressor then fits Prophet's in-sample residuals using the full feature set (price, demand signals, external factors), picking up the non-linear interactions Prophet misses. The ensemble has the lowest RMSE on the holdout; the XGBoost stage alone has the lowest WAPE (see Results). The unified prediction surface lives in `src/models/predict_demand.py` as `DemandModel.predict()`.

<details>
<summary>Click for design choices and trade-offs (Stage 1)</summary>

| Aspect | Description |
| :--- | :--- |
| **Stage A – Prophet** | One model per `(match_id, seat_zone)` series captures trend, weekly seasonality, and weekday/holiday effects via Prophet regressors. |
| **Stage B – XGBoost** | A single XGBoost regressor is fit on Prophet's in-sample residuals using the full feature set. |
| **Prediction** | `final = clip(prophet_yhat + xgb_residual, 0, ∞)`. |
| **Why this split** | Prophet handles temporal structure cleanly; XGBoost picks up complex non-linear interactions Prophet cannot. On this holdout the XGBoost stage alone is more accurate on WAPE/MAE and the ensemble on RMSE/R²; see Results. |

</details>

### Stage 2 – Price optimization

Grid search over the prices each zone was **actually sold at**: the 5th–95th percentile of historical prices per zone, computed at training time and saved to `models/price_support.json` (quantiles configurable in `src/decision_engine/constants.py`, `PRICE_SUPPORT_QUANTILES`). Outside this band the model is extrapolating, the trees predict flat sales, and the argmax drifts to the cap. For each candidate price, the engine builds a row, predicts sales with `DemandModel`, computes `revenue = price × predicted_sales`, and returns the argmax. The result becomes a `Price Variation Proposal` sent to the commercial team for approval.

<details>
<summary>Click for design choices and trade-offs (Stage 2)</summary>

| Aspect | Description |
| :--- | :--- |
| **Why grid search** | Pricing is a critical business decision; grid search **guarantees** the revenue-maximizing price within the search space, at modest compute cost (one vectorized prediction batch per match-zone). |
| **Process** | For each candidate price in the range, build a row, predict sales with `DemandModel`, compute `revenue = price × predicted_sales` (unrounded, so low-volume zones keep a usable curve), return the argmax. |
| **Why not Bayesian opt.** | Bayesian optimization would converge faster but doesn't guarantee the maximum. For pricing decisions, the guarantee is worth the modest extra cost. |

</details>


## Dataset

The repository ships a synthetically generated dataset engineered to mirror the complexity and statistical properties of a real ticketing environment: 10 matches of varied importance, a 90-day daily sales window per match, and up to 5 stadium zones per match (a small per-zone dropout probability removes some pairs to mimic real data gaps, yielding ~37-43 of the 50 possible series).

> ⚠️ **`web_conversion_rate` is deliberately dropped from the model's feature set** (`src/features/build_features.py`). It is defined as `sales / web_visits` in the generator, which is target leakage at decision time: when we propose a price, the conversion rate at that price is precisely what we are trying to predict. Including it would inflate the headline metrics while breaking the optimizer's price-elasticity signal.

<details>
<summary>Click for the full feature schema</summary>

| Category | Features | Description |
| :--- | :--- | :--- |
| **Match & Opponent** | `match_id`, `days_until_match`, `is_weekday`, `opponent_tier`, `ea_opponent_strength`, `is_international` | Core details about the match, its timing, and opponent quality. |
| **Team Status** | `team_position`, `top_player_injured`, `league_winner_known` | Current performance, player status, and league context. |
| **Ticket & Zone** | `seat_zone`, `ticket_price`, `ticket_availability_pct`, `zone_seats_availability` | Attributes of the specific ticket and seating area. |
| **Demand & Hype** | `internal_search_trends`, `google_trends_index`, `social_media_sentiment`, `web_visits` | Digital signals measuring interest and purchase intent. |
| **External Factors**| `is_holiday`, `popular_concert_in_city`, `competitor_avg_price`, `flights_to_barcelona_index` | External events, competition, and tourism proxies. |
| **Weather** | `weather_forecast` | Forecasted weather conditions for the match day. |

> **`zone_historical_sales`** [Target Variable] – the historical number of tickets sold in a given zone-day. This is what the model predicts.

</details>

<details>
<summary>Click for how prices and sales are generated</summary>

Two pieces of the data generator carry the model's job:

1. **Price has a wide, partly-independent distribution.** Real-world prices reflect both a strategic baseline tied to match excitement *and* operational variation (A/B tests, promotions, last-minute discounts). The generator implements both, so price varies roughly between `0.5×` and `2.5×` the zone's base price even within a single excitement level. Without this, the model cannot identify price elasticity from historical observations alone.
2. **Demand follows a linear-elasticity curve with a hard ceiling at `2.5×` base price.** This yields a clean interior revenue optimum near `~1.25×` base, rather than a runaway "pick the price cap" recommendation.

</details>

<details>
<summary>Click for the Match Excitement Factor</summary>

The generation script unifies non-price demand drivers under a single **"Match Excitement Factor"**:

1. **Starts with the opponent**: a top-tier opponent generates more interest.
2. **Adjusts for context**: league position, player injuries, match importance (e.g. league winner already decided), proximity to holidays, weekday/weekend.
3. **Drives the demand signals**: `google_trends_index`, `social_media_sentiment`, `internal_search_trends` all scale with this factor.

</details>


## Repo tree

```bash
dynamic-pricing/
├── Makefile                            # Pipeline: data → train → evaluate → sanity → app
├── README.md
├── app.py                              # Streamlit HiTL page
├── config.py                           # Paths and reference dates
├── requirements.txt
├── assets/                             # Diagrams and images
├── data/
│   └── 03_synthetic/
│       └── synthetic_match_data.csv    # Generated by make_dataset.py
├── models/                             # Trained artifacts (regenerable, gitignored)
│   ├── prophet_models.joblib
│   ├── xgb_residual_model.joblib
│   ├── feature_pipeline.joblib
│   └── price_support.json              # Per-zone price band the optimizer may search
└── src/
    ├── data/
    │   └── make_dataset.py             # Synthetic data generator (seeded)
    ├── features/
    │   └── build_features.py           # Pipeline factory: drops, scales, one-hot encodes
    ├── models/
    │   ├── train_demand_model.py       # Fits Prophet + XGBoost ensemble
    │   ├── predict_demand.py           # DemandModel: unified predict() surface
    │   ├── evaluate.py                 # Leakage-free holdout metrics + decision quality
    │   └── sanity_check.py             # Asserts the model actually responds to price
    └── decision_engine/
        ├── simulate.py                 # What-if for a single price
        ├── optimize.py                 # Zone-aware grid-search optimal price
        └── constants.py                # Sample feature row, zone base prices, band quantiles
```

### Reproducing the results

```bash
pip install -r requirements.txt
make clean                              # remove cached artifacts
make all                                # data + train + evaluate (accuracy + decision quality) + sanity check
make app                                # launch the Streamlit page

# Or run the CLI examples directly:
python -m src.decision_engine.simulate
python -m src.decision_engine.optimize
```

The Streamlit page lets you pick a match and seat zone, see the recommended price with a revenue-vs-price curve, simulate any hypothetical price, and **approve** the recommendation – which appends a JSON line to `proposals.jsonl`. That's the HiTL loop in miniature.

---

## License

MIT
