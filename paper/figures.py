# %% [markdown]
# Figures and numbers for the paper.
#
# Reruns the holdout evaluation the same way `src/models/evaluate.py` does, then
# writes the four evaluation charts to `paper/figures/*.svg` and every number the
# paper quotes to `paper/figures/numbers.json`.
#
#     python paper/figures.py
#
# Set FONT_DIR to a folder with LibertinusSerif-*.otf to match the paper's font.

# %%
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.data.make_dataset import price_effect
from src.decision_engine.constants import ZONE_BASE_PRICES
from src.decision_engine.optimize import OptimizationEngine
from src.models.evaluate import HOLDOUT_DAYS, SERIES, series_lookup, true_revenue_share, wape
from src.models.train_demand_model import train_ensemble

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('src.models.predict_demand').setLevel(logging.ERROR)

OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Zone labels in the paper (the synthetic data keeps the original names).
ZONE_LABELS = {'VIP': 'VIP', 'Lateral': 'Sideline', 'Corner': 'Corner', 'Gol Nord': 'North end', 'Gol Sud': 'South end'}
UNCONSTRAINED = (0.5, 2.5)  # the earlier search band, as multiples of base price

# %% Style: print figure, paper font, one accent hue plus a contrast hue (validated pair)
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK_2, GRID = '#1a1a1a', '#6b6a66', '#e4e3df'
COL_W, FULL_W = 3.35, 7.0

font_dir = os.environ.get('FONT_DIR')
if font_dir:
    for f in Path(font_dir).glob('LibertinusSerif-*.otf'):
        font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    'font.family': 'Libertinus Serif' if font_dir else 'serif',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.edgecolor': INK_2,
    'axes.labelcolor': INK,
    'xtick.color': INK_2,
    'ytick.color': INK_2,
    'text.color': INK,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'lines.linewidth': 1.5,
    'legend.frameon': False,
    'svg.fonttype': 'path',
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# %% Holdout split and the evaluation model, exactly as evaluate.py
df = pd.read_csv(config.SYNTHETIC_DATA_PATH)
is_holdout = df['days_until_match'] < HOLDOUT_DAYS
train_df, test_df = df[~is_holdout].copy(), df[is_holdout].copy()
features_test = test_df.drop(columns=[config.TARGET_COLUMN]).reset_index(drop=True)
y_test = test_df[config.TARGET_COLUMN].to_numpy()

model = train_ensemble(train_df)
y_pred = model.predict(features_test)
recent = train_df[train_df['days_until_match'] < HOLDOUT_DAYS + 7]
y_last7 = series_lookup(test_df, recent.groupby(SERIES)[config.TARGET_COLUMN].mean())

# %% Forecast accuracy (Table 2)
y_prophet = np.clip(model._prophet_forecast(features_test)[0], 0, None)
y_pooled = np.clip(model.pooled_xgb.predict(model.feature_pipeline.transform(features_test)), 0, None)
y_mean = np.full_like(y_test, train_df[config.TARGET_COLUMN].mean(), dtype=float)


def metrics(y_hat: np.ndarray) -> dict:
    err = y_test - y_hat
    r2 = 1 - (err ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    return {'wape': wape(y_test, y_hat), 'r2': float(r2), 'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt((err ** 2).mean()))}


accuracy = {
    'ensemble': metrics(y_pred),
    'prophet': metrics(y_prophet),
    'xgb_pooled': metrics(y_pooled),
    'naive_last7': metrics(y_last7),
    'naive_mean': metrics(y_mean),
}

# %% Decision quality: support-constrained search vs the earlier unconstrained band
zones = features_test['seat_zone']
base = zones.map(ZONE_BASE_PRICES).to_numpy()
engine = OptimizationEngine(model=model)
rec = engine.recommend_prices(features_test)
share = true_revenue_share(rec, zones)
bands = zones.map(model.price_support)
at_edge = np.array([p >= hi - 5 for p, (lo, hi) in zip(rec, bands)])

wide_support = {z: (int(b * UNCONSTRAINED[0]), int(b * UNCONSTRAINED[1])) for z, b in ZONE_BASE_PRICES.items()}
engine_wide = OptimizationEngine(model=replace(model, price_support=wide_support))
rec_wide = engine_wide.recommend_prices(features_test)
share_wide = true_revenue_share(rec_wide, zones)
at_cap_wide = np.array([p >= wide_support[z][1] - 5 for p, z in zip(rec_wide, zones)])

by_zone = pd.DataFrame({'zone': zones, 'constrained': share, 'unconstrained': share_wide}).groupby('zone').mean()
max_train_ratio = float((train_df['ticket_price'] / train_df['seat_zone'].map(ZONE_BASE_PRICES)).max())

decision = {
    'share_mean': float(share.mean()),
    'share_by_zone': {ZONE_LABELS[z]: float(v) for z, v in by_zone['constrained'].items()},
    'median_ratio': float(np.median(rec / base)),
    'upper_edge': float(at_edge.mean()),
    'unconstrained_share_mean': float(share_wide.mean()),
    'unconstrained_at_cap': float(at_cap_wide.mean()),
    'unconstrained_share_by_zone': {ZONE_LABELS[z]: float(v) for z, v in by_zone['unconstrained'].items()},
    'max_train_price_ratio': max_train_ratio,
}

# %% Figure: price response for one zone, averaged over its holdout rows
ZONE = 'Lateral'
rows = features_test[features_test['seat_zone'] == ZONE].reset_index(drop=True)
b = ZONE_BASE_PRICES[ZONE]
ratios = np.round(np.arange(UNCONSTRAINED[0], UNCONSTRAINED[1] + 1e-9, 0.025), 3)
batch = rows.loc[rows.index.repeat(len(ratios))].reset_index(drop=True)
batch['ticket_price'] = np.tile(ratios * b, len(rows))
pred_rev = (batch['ticket_price'] * model.predict(batch)).to_numpy().reshape(len(rows), len(ratios)).mean(axis=0)
true_rev = ratios * price_effect(ratios)
lo, hi = (v / b for v in model.price_support[ZONE])
in_band = (ratios >= lo) & (ratios <= hi)
pick_band = ratios[in_band][pred_rev[in_band].argmax()]
pick_wide = ratios[pred_rev.argmax()]

fig, ax = plt.subplots(figsize=(COL_W, 2.3))
ax.axvspan(lo, hi, color=BLUE, alpha=0.08, lw=0)
ax.text(lo + 0.03, 1.2, 'prices seen\nin training', ha='left', va='top', fontsize=6.5, color=INK_2, linespacing=1.1)
ax.plot(ratios, true_rev / true_rev.max(), color=INK_2, ls='--', lw=1.2, label='True revenue')
ax.plot(ratios, pred_rev / pred_rev.max(), color=BLUE, label='Model-predicted revenue')
marks = [
    (1.25, INK_2, INK, f'true\noptimum\n1.25×', 'right', 0.04),
    (pick_band, BLUE, BLUE, f'constrained\npick\n{pick_band:.2f}×', 'left', 0.04),
    (pick_wide, ORANGE, ORANGE, f'unconstrained\npick\n{pick_wide:.2f}×', 'right', 0.45),
]
for x, line_c, text_c, text, side, y in marks:
    ax.axvline(x, color=line_c, lw=0.8)
    ax.text(x + (0.03 if side == 'left' else -0.03), y, text, ha=side, va='bottom', fontsize=6.5, color=text_c, linespacing=1.1)
ax.set_xlim(UNCONSTRAINED)
ax.set_ylim(0, 1.25)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel('Price as a multiple of the zone base price')
ax.set_ylabel('Revenue (scaled to each curve’s max)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, 'price_response')
decision['example_zone'] = ZONE_LABELS[ZONE]
decision['example_pick_constrained'] = float(pick_band)
decision['example_pick_unconstrained'] = float(pick_wide)

# %% Figure: forecast vs actual for the series at the 25th / 50th / 75th percentile of holdout WAPE
test_keyed = test_df[SERIES + ['days_until_match', config.TARGET_COLUMN]].reset_index(drop=True).assign(pred=y_pred, naive=y_last7)
series_wape = test_keyed.groupby(SERIES).apply(lambda g: wape(g[config.TARGET_COLUMN].to_numpy(), g['pred'].to_numpy()), include_groups=False).sort_values()
picks = [series_wape.index[int(round(q * (len(series_wape) - 1)))] for q in (0.25, 0.5, 0.75)]

fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.9), sharey=False)
for ax, (match_id, zone), q in zip(axes, picks, ('25th', '50th', '75th')):
    hist = df[(df['match_id'] == match_id) & (df['seat_zone'] == zone)].sort_values('days_until_match', ascending=False)
    hold = test_keyed[(test_keyed['match_id'] == match_id) & (test_keyed['seat_zone'] == zone)].sort_values('days_until_match', ascending=False)
    ax.axvspan(HOLDOUT_DAYS - 0.5, -0.5, color=BLUE, alpha=0.08, lw=0)
    ax.plot(hist['days_until_match'], hist[config.TARGET_COLUMN], color=INK_2, lw=1.0, label='Actual sales')
    ax.plot(hold['days_until_match'], hold['naive'], color=INK_2, ls=':', lw=1.2, label='Naive (last 7 days)')
    ax.plot(hold['days_until_match'], hold['pred'], color=BLUE, label='Ensemble forecast')
    ax.set_xlim(60, -0.5)
    ax.set_ylim(bottom=0)
    ax.set_title(f'Match {match_id}, {ZONE_LABELS[zone]} · WAPE {series_wape[(match_id, zone)]:.0%} ({q} pct.)', loc='left')
    ax.set_xlabel('Days until match')
axes[0].set_ylabel('Tickets sold per day')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
fig.tight_layout(w_pad=1.2)
save(fig, 'forecast_examples')

# %% Figure: decision quality by zone, constrained vs unconstrained search
order = ['Corner', 'North end', 'South end', 'Sideline', 'VIP']
cons = [decision['share_by_zone'][z] for z in order] + [decision['share_mean']]
wide = [decision['unconstrained_share_by_zone'][z] for z in order] + [decision['unconstrained_share_mean']]
labels = order + ['All zones']
y = np.arange(len(labels))[::-1]
h = 0.36

fig, ax = plt.subplots(figsize=(COL_W, 2.3))
ax.barh(y + h / 2 + 0.01, cons, height=h, color=BLUE, label='Observed-price band')
ax.barh(y - h / 2 - 0.01, wide, height=h, color=ORANGE, label=f'Unconstrained ({UNCONSTRAINED[0]}×–{UNCONSTRAINED[1]}×)')
for yi, c, w in zip(y, cons, wide):
    ax.text(c + 0.01, yi + h / 2, f'{c:.0%}', va='center', fontsize=6.5)
    ax.text(w + 0.01, yi - h / 2, f'{w:.0%}', va='center', fontsize=6.5)
ax.set_yticks(y, labels)
ax.axhline(0.5, color=INK_2, lw=0.5)
ax.set_xlim(0, 1.1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0%', '25%', '50%', '75%', '100%'])
ax.grid(axis='y', visible=False)
ax.set_xlabel('Share of best achievable revenue (holdout mean)')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.16), ncol=2)
save(fig, 'decision_quality')

# %% Cold start: leave one match out, with and without the pooled fallback
cold_rows, cold_shares = [], []
for match_id in sorted(df['match_id'].unique()):
    held = df['match_id'] == match_id
    m = train_ensemble(df[~held])
    feats = df[held].drop(columns=[config.TARGET_COLUMN]).reset_index(drop=True)
    y_true = df.loc[held, config.TARGET_COLUMN].to_numpy()
    cold_rows.append({
        'match_id': int(match_id),
        'fallback': wape(y_true, m.predict(feats)),
        'residual_only': wape(y_true, replace(m, pooled_xgb=None).predict(feats)),
    })
    cold_shares.append(true_revenue_share(OptimizationEngine(model=m).recommend_prices(feats), feats['seat_zone']))
cold = pd.DataFrame(cold_rows).sort_values('fallback', ascending=False).reset_index(drop=True)
cold_start = {
    'fallback_median': float(cold['fallback'].median()),
    'fallback_min': float(cold['fallback'].min()),
    'fallback_max': float(cold['fallback'].max()),
    'residual_only_median': float(cold['residual_only'].median()),
    'unseen_share_mean': float(np.concatenate(cold_shares).mean()),
    'per_match': cold_rows,
}

fig, ax = plt.subplots(figsize=(COL_W, 2.1))
yy = np.arange(len(cold))
ax.hlines(yy, cold['fallback'], cold['residual_only'], color=GRID, lw=2)
ax.scatter(cold['residual_only'], yy, s=22, color=ORANGE, zorder=3, label='Without fallback', edgecolor='white', linewidth=0.6)
ax.scatter(cold['fallback'], yy, s=22, color=BLUE, zorder=3, label='With pooled XGBoost fallback', edgecolor='white', linewidth=0.6)
ax.set_yticks(yy, [f'Match {m}' for m in cold['match_id']])
ax.set_xlim(0, 1.05)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0%', '25%', '50%', '75%', '100%'])
ax.grid(axis='y', visible=False)
ax.set_xlabel('WAPE on a match never seen in training')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.17), ncol=2)
save(fig, 'cold_start')

# %% Analyst panel mockup: one match on one day, every zone's proposal from the evaluation model
PANEL_MATCH, PANEL_DAY, PANEL_ZONE = 5, 10, 'Lateral'
day_rows = features_test[(features_test['match_id'] == PANEL_MATCH) & (features_test['days_until_match'] == PANEL_DAY)]
assert len(day_rows) and PANEL_ZONE in set(day_rows['seat_zone'])
queue = []
for _, row in day_rows.iterrows():
    row_df = row.to_frame().T.astype(day_rows.dtypes.to_dict())
    curve = engine.revenue_curve(row_df)
    best = curve.loc[curve['projected_revenue'].idxmax()]
    current_rev = float(row['ticket_price'] * model.predict(row_df)[0])
    lo_z, hi_z = model.price_support[row['seat_zone']]
    queue.append({
        'zone': ZONE_LABELS[row['seat_zone']],
        'current_price': float(row['ticket_price']),
        'recommended_price': float(best['price']),
        'predicted_sales': float(model.predict(row_df.assign(ticket_price=float(best['price'])))[0]),
        'projected_revenue': float(best['projected_revenue']),
        'current_revenue': current_rev,
        'at_upper_edge': bool(best['price'] >= hi_z - 5),
        'band': [lo_z, hi_z],
        'curve': [[float(pr), float(rv)] for pr, rv in zip(curve['price'], curve['projected_revenue'])] if row['seat_zone'] == PANEL_ZONE else None,
    })
sel = day_rows[day_rows['seat_zone'] == PANEL_ZONE].iloc[0]
panel = {
    'match_id': PANEL_MATCH,
    'days_until_match': PANEL_DAY,
    'opponent_tier': str(sel['opponent_tier']),
    'weather': str(sel['weather_forecast']),
    'selected_zone': ZONE_LABELS[PANEL_ZONE],
    'availability_pct': float(sel['ticket_availability_pct']),
    'queue': sorted(queue, key=lambda q: -q['projected_revenue']),
}

# %% Write every number the paper quotes
numbers = {
    'data': {'rows': int(len(df)), 'series': int(df.groupby(SERIES).ngroups), 'matches': int(df['match_id'].nunique()), 'holdout_rows': int(len(test_df))},
    'accuracy': accuracy,
    'naive_wape_reduction': 1 - accuracy['ensemble']['wape'] / accuracy['naive_last7']['wape'],
    'decision': decision,
    'cold_start': cold_start,
    'price_support': {ZONE_LABELS[z]: list(v) for z, v in model.price_support.items()},
    'panel': panel,
}
(OUT / 'numbers.json').write_text(json.dumps(numbers, indent=2))
print(json.dumps({k: v for k, v in numbers.items() if k != 'cold_start'}, indent=2))
print(json.dumps({k: v for k, v in cold_start.items() if k != 'per_match'}, indent=2))
