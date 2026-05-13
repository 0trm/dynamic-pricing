![My Cover](./assets/dp-cover.jpeg)

# Dynamic Pricing Engine with HITL

<p align="left">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
  <img src="https://img.shields.io/badge/ML-Supervised-lightgrey" alt="License">
  <img src="https://img.shields.io/badge/Models-Prophet%20%2B%20XGBoost-lightgrey" alt="Models">
</p>

> An ML-powered dynamic pricing and decision support system for ticket pricing in a sports stadium.
> **Objective:** evolve a manual price-decision process into a data-driven, semi-automated workflow that improves ticketing revenue and sales.

### Quickstart

```bash
pip install -r requirements.txt
make all          # synthetic data, train, holdout evaluation, elasticity sanity check
make app          # open the human-in-the-loop Streamlit page
```

### Outline

- [Holdout results](#holdout-results-on-synthetic-data)
- [Real-world impact](#real-world-impact)
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Structure](#structure)

---

## Holdout results (on synthetic data)

Produced by `make evaluate` against a 14-day per-series holdout. The ensemble is re-trained from scratch on the train split before predicting on the holdout, so the numbers below are leakage-free.

| Metric | Ensemble (Prophet + XGBoost) | Baseline (mean) |
| :--- | :--- | :--- |
| **WAPE** | **26.7%** | 79.7% |
| **R²** | **0.726** | -0.428 |
| **MAE** | **6.2** tickets | 18.6 tickets |
| **RMSE** | **11.8** tickets | 27.0 tickets |

The ensemble's WAPE is **67% lower** than the mean baseline. Numbers are reproducible with `RANDOM_SEED=42` in `src/data/make_dataset.py`.

### Elasticity sanity check

A common failure mode for demand models is to learn everything *except* the price-to-sales relationship, leaving the optimizer to recommend the price cap on every row. `make sanity` defends against this. It samples 20 historical rows, sweeps each through the optimizer's price range, and asserts that predicted sales actually move with price and do so non-trivially (relative spread >= 20%). The check fails the build if too many rows are flat or non-monotonic.

Latest run: **mean spread 11.6 tickets across the per-zone range**, **median optimal price €182** (well inside the search band), **0% violation rate**. The optimizer is making real decisions, not picking the ceiling.

## Real-world impact

> ℹ️ The numbers below come from a deployment on a confidential real-world dataset. They are **not** reproduced by this repository; the public code runs on a synthetic dataset designed to mirror the statistical structure of the original. Treat them as case-study evidence, not a benchmark.

| Metric | Result | Description |
| :--- | :--- | :--- |
| 📈 Revenue uplift | **+6%** avg. revenue per match | Achieved by dynamically adjusting prices to match real-time demand forecasts. Validated via A/B testing. |
| 🎟️ Optimized sales | **+4%** sell-through rate | Improved occupancy alongside revenue, which positively affects atmosphere and in-stadium sales. |
| ⚙️ Operational efficiency | **7×** faster price changes | From weekly to daily updates by automating data aggregation and analysis. |
| 🤝 Recommendation adoption | **86%** of proposals approved | Commercial team reviewed and approved the model's price proposals at a high rate, indicating trust. |

## Overview

The challenge: transform a static, manual pricing strategy into a responsive, automated system with a human-in-the-loop (HiTL), creating a market-driven approach to setting ticket prices per match.

<p align="center">
  <img src="./assets/dp-ss.jpeg" alt="Stadium ticketing price list" width="1000">
  <br>
  <em>Fig. 1: A standard stadium ticket pricing by zone during the checkout process.</em>
</p>

| 🚩 Problem | 💡 Solution |
| :--- | :--- |
| **Static pricing**: prices set once per season in rigid categories (A++, A, B), updated weekly/monthly. | **Dynamic recommendations**: price proposals per seating zone based on near real-time data analysis, allowing daily updates. |
| **Manual adjustments**: slow analysis to propose changes. | **Impact simulation**: instantly model projected impact of any price change on revenue and ticket sales. |
| **Data bottleneck**: manual extraction from fragmented systems. | **Centralized data**: aggregates sales, web analytics, contextual data into one place. |
| **Slow implementation**: disconnected from the sales platform. | **Seamless integration**: one-click approval on a dashboard pushes a price update to the live ticketing system. |

The diagram below illustrates the conceptual framework. The system acts as the *brain* balancing the goals of The Club and The Fan, ingesting internal and external factors to forecast demand at various price points. The **Decision Engine** then uses this forecast to recommend an optimal price.

<p align="center">
  <img src="./assets/dp-md.png" alt="High-level market dynamics diagram" width="2000">
  <br>
  <em>Fig. 2: A high-level diagram of the system's market dynamics.</em>
</p>


## Architecture

The diagram below is the system context view. The central **Dynamic Pricing Engine** ingests historical data from the **Club's Data Systems** and real-time sales data from the **Ticketing System**, recommends prices that are simulated and approved by the **Club's Pricing Team**, and pushes approved prices to the live Ticketing System where **The Fan** completes a purchase, closing the feedback loop.

<p align="center">
  <img src="./assets/dp-scd.png" alt="System Context Diagram" width="800">
  <br>
  <em>Fig. 3: System Context Diagram – Dynamic Pricing System.</em>
</p>


## Dataset

The repository ships a synthetically generated dataset engineered to mirror the complexity and statistical properties of a real ticketing environment.

The dataset simulates:

* **10 matches** of varied importance (league, cup, international).
* **A full 90-day sales window** per match, daily records.
* **Up to 5 stadium zones** per match. A small per-zone dropout probability (`ZONE_SKIP_PROBABILITY` in `src/data/make_dataset.py`) removes some zone-match pairs to mimic real data gaps; in practice this yields ~37-43 of the 50 possible series.

| Category | Features | Description |
| :--- | :--- | :--- |
| **Match & Opponent** | `match_id`, `days_until_match`, `is_weekday`, `opponent_tier`, `ea_opponent_strength`, `is_international` | Core details about the match, its timing, and opponent quality. |
| **Team Status** | `team_position`, `top_player_injured`, `league_winner_known` | Current performance, player status, and league context. |
| **Ticket & Zone** | `seat_zone`, `ticket_price`, `ticket_availability_pct`, `zone_seats_availability` | Attributes of the specific ticket and seating area. |
| **Demand & Hype** | `internal_search_trends`, `google_trends_index`, `social_media_sentiment`, `web_visits` | Digital signals measuring interest and purchase intent. |
| **External Factors**| `is_holiday`, `popular_concert_in_city`, `competitor_avg_price`, `flights_to_barcelona_index` | External events, competition, and tourism proxies. |
| **Weather** | `weather_forecast` | Forecasted weather conditions for the match day. |

> **`zone_historical_sales`** [Target Variable] – the historical number of tickets sold in a given zone-day. This is what the model predicts.

`web_conversion_rate` is present in the CSV for completeness but is **deliberately dropped from the model's feature set** (`src/features/build_features.py`). It is defined as `sales / web_visits` in the generator, which is target leakage at decision time: at the moment we propose a price, the conversion rate at that price is precisely what we are trying to predict. Including it would inflate the headline metrics while breaking the optimizer's price-elasticity signal.

### How prices and sales are generated

Two pieces of the data generator carry the model's job:

1. **Price has a wide, partly-independent distribution.** Real-world prices reflect both a strategic baseline tied to match excitement *and* operational variation (A/B tests, promotions, last-minute discounts). The generator implements both, so price varies roughly between `0.5×` and `2.5×` the zone's base price even within a single excitement level. Without this, the model cannot identify price elasticity from historical observations alone.
2. **Demand follows a linear-elasticity curve with a hard ceiling at `2.5×` base price.** This yields a clean interior revenue optimum near `~1.25×` base, rather than a runaway "pick the price cap" recommendation.

### Match Excitement Factor

The generation script unifies non-price demand drivers under a single **"Match Excitement Factor"**:

1. **Starts with the opponent**: a top-tier opponent generates more interest.
2. **Adjusts for context**: league position, player injuries, match importance (e.g. league winner already decided), proximity to holidays, weekday/weekend.
3. **Drives the demand signals**: `google_trends_index`, `social_media_sentiment`, `internal_search_trends` all scale with this factor.


## Modeling

The model is a two-stage hybrid: first **predict**, then **optimize**.

<p align="left">
  <img src="./assets/dp-dpe.png" alt="Dynamic Pricing Engine" width="275">
  <br>
  <em>Fig. 4: Dynamic Pricing Engine component.</em>
</p>

### Stage 1: 📈 Demand Forecasting

Answers: *"At a given price, how many tickets are we likely to sell?"*

| Aspect | Description |
| :--- | :--- |
| **Model** | A hybrid **ensemble**: Prophet + XGBoost in a residual-fitting setup. |
| **Stage A – Prophet** | One model per `(match_id, seat_zone)` series captures trend, weekly seasonality, and weekday/holiday effects via Prophet regressors. |
| **Stage B – XGBoost** | A single XGBoost regressor is fit on Prophet's in-sample residuals using the full feature set (price, demand signals, external factors). |
| **Prediction** | `final = clip(prophet_yhat + xgb_residual, 0, ∞)`. Implemented as a single `DemandModel.predict()` surface in `src/models/predict_demand.py`. |
| **Why this split** | Prophet handles temporal structure cleanly; XGBoost picks up complex non-linear interactions Prophet cannot. The combination beats either alone on the holdout. |

See [Holdout Results](#holdout-results-on-synthetic-data) above for metrics.

### Stage 2: ⚙️ Price Optimization

Answers: *"What is the single best price to maximize total revenue?"*

| Aspect | Description |
| :--- | :--- |
| **Approach** | Grid search over a zone-aware range of valid prices. |
| **Search bounds** | Per-zone bounds: `[0.5 × base_price, 2.5 × base_price]`. Outside this band the model would be extrapolating beyond the training distribution and the residual XGBoost can't be trusted. The band is configurable in `src/decision_engine/constants.py` (`PRICE_SEARCH_RANGE_RATIO`). |
| **Why grid search** | Pricing is a critical business decision; grid search **guarantees** the revenue-maximizing price within the search space, at modest compute cost (one vectorized prediction batch per match-zone). |
| **Process** | For each candidate price in the range, build a row, predict sales with `DemandModel`, compute `revenue = price × predicted_sales`, return the argmax. |
| **Output** | A `Price Variation Proposal` (recommended price + projected revenue + projected sales) sent to the commercial team for approval. |
| **Why not Bayesian opt.** | Bayesian optimization would converge faster but doesn't guarantee the maximum. For pricing decisions, the guarantee is worth the modest extra cost. |

### Validation (in the real-world deployment)

In production, the engine was validated via segment-based **A/B tests**: a subset of seating zones used the dynamic engine (treatment), while the rest stayed on static pricing (control). Tests were run across matches of varying importance to ensure the lift wasn't an artifact of any single event. Result: a consistent **+6% lift in average revenue per match** with a **+4% sell-through rate** – confirming the model found market equilibrium rather than simply over-charging.

*(This A/B analysis is not reproduced here – the public repo focuses on the predictive/optimization core, not the deployment infrastructure.)*


## Structure

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
│   └── feature_pipeline.joblib
└── src/
    ├── data/
    │   └── make_dataset.py             # Synthetic data generator (seeded)
    ├── features/
    │   └── build_features.py           # Pipeline factory: drops, scales, one-hot encodes
    ├── models/
    │   ├── train_demand_model.py       # Fits Prophet + XGBoost ensemble
    │   ├── predict_demand.py           # DemandModel: unified predict() surface
    │   ├── evaluate.py                 # Leakage-free holdout metrics
    │   └── sanity_check.py             # Asserts the model actually responds to price
    └── decision_engine/
        ├── simulate.py                 # What-if for a single price
        ├── optimize.py                 # Zone-aware grid-search optimal price
        └── constants.py                # Sample feature row + zone base prices
```

### Reproducing the results

```bash
pip install -r requirements.txt
make clean                              # remove cached artifacts
make all                                # data + train + evaluate + sanity check
make app                                # launch the Streamlit page

# Or run the CLI examples directly:
python -m src.decision_engine.simulate
python -m src.decision_engine.optimize
```

The Streamlit page lets you pick a match and seat zone, see the recommended price with a revenue-vs-price curve, simulate any hypothetical price, and **approve** the recommendation – which appends a JSON line to `proposals.jsonl`. That's the HiTL loop in miniature.

---

## License

MIT

<br>

*Built ~~by~~ with AI.* <br>
© 2026 trm
