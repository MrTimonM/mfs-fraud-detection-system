"""Redraw README images from saved CSVs; requires Matplotlib, no retraining."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'mfs-guard-ml/results'
OUTPUT = ROOT / 'docs/images'
OUTPUT.mkdir(parents=True, exist_ok=True)
TEAL, NAVY, MUTED = '#00858A', '#163248', '#A9BDC9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': NAVY, 'axes.labelcolor': NAVY, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.spines.left': False,
    'axes.edgecolor': '#DAE4E9', 'xtick.color': NAVY, 'ytick.color': NAVY})

def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def save(fig, name):
    fig.savefig(OUTPUT / name, dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)

names = {'LightGBM_200k': 'LightGBM 200k (selected)',
    'Hybrid_WithoutSequence_500k': 'Hybrid without sequence 500k',
    'Hybrid_AddAnomaly_500k': 'Rules + ML + anomaly 500k',
    'Hybrid_RulesSupervised_500k': 'Rules + ML 500k', 'XGBoost_500k': 'XGBoost 500k',
    'LightGBM_500k': 'LightGBM 500k', 'FullHybrid_500k': 'Full hybrid 500k',
    'RulesOnly_500k': 'Rules only 500k', 'IsolationForest_500k': 'Isolation Forest 500k'}
rows = sorted(read(RESULTS / 'final_common_test/model_comparison.csv'),
              key=lambda r: float(r['pr_auc']), reverse=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), sharey=True)
palette = [TEAL if r['model'] == 'LightGBM_200k' else MUTED for r in rows]
for ax, metric, heading, fmt in zip(axes, ['pr_auc', 'p95_latency_ms'],
        ['Average precision / PR-AUC | higher is better',
         'Batch-1 P95 inference (ms) | lower is better'], ['%.4f', '%.2f']):
    values = [float(r[metric]) for r in rows]
    bars = ax.barh(range(len(rows)), values, color=palette, height=.62)
    ax.bar_label(bars, fmt=fmt, padding=5, fontsize=10)
    ax.set_xlim(0, max(values)*1.2)
    ax.set_title(heading, loc='left', fontsize=12, pad=15)
    ax.grid(axis='x', alpha=.16)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(rows)), [names[r['model']] for r in rows])
axes[0].invert_yaxis()
axes[0].axvline(.05223, ls='--', lw=1, color='#C27B3C')
fig.suptitle('Comparable ranking. Less inference overhead.', fontsize=21, weight='bold', x=.03, ha='left')
fig.text(.03, .02, '100,000 common synthetic test rows | 5.223% fraud (dashed AP reference) | Latency excludes history construction, state lookup and HTTP.', fontsize=10)
fig.tight_layout(rect=(0, .07, 1, .94), w_pad=3)
save(fig, 'model-comparison.png')

factors = read(RESULTS / '200k/shap_feature_importance.csv')[:10][::-1]
fig, ax = plt.subplots(figsize=(12, 6.2))
bars = ax.barh([r['feature'].replace('_', ' ') for r in factors],
    [float(r['mean_absolute_shap']) for r in factors], color=TEAL, height=.62)
ax.bar_label(bars, fmt='%.3f', padding=5, fontsize=10)
ax.set_xlim(0, .36)
ax.set_xlabel('Mean absolute SHAP attribution (uncalibrated tree output)', labelpad=12)
ax.set_title('What influences the selected LightGBM?', loc='left', fontsize=20, weight='bold', pad=20)
ax.grid(axis='x', alpha=.16)
ax.set_axisbelow(True)
fig.text(.025, .02, '256 sampled held-out rows from the 200k experiment | Feature influence is not causal evidence or measured performance uplift.', fontsize=10)
fig.tight_layout(rect=(0, .07, 1, 1))
save(fig, 'model-factors.png')
print('Updated docs/images/model-comparison.png and model-factors.png')
