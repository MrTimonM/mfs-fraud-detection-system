"""STEP 3: decision framework over measured common-test results (no new scoring)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import ROOT, dump

R = ROOT/'results/final_common_test'
# Qualitative 1 (best) .. 3 (worst); based on component counts measured in model_comparison.csv.
QUALITATIVE = {'supervised': (1, 1, 1, 1), 'fusion': (3, 2, 3, 3), 'rules': (1, 1, 1, 1), 'anomaly': (2, 3, 2, 2)}


def main():
    m = pd.read_csv(R/'model_comparison.csv')
    ci = pd.read_csv(R/'confidence_intervals.csv')
    types = pd.read_csv(R/'fraud_type_performance.csv')
    cost = pd.read_csv(R/'business_cost_comparison.csv')
    tiers = cost[cost.policy.str.startswith('validation tiers')].set_index('model')
    cands = m[m.role != 'supporting_layer_reference'].copy()
    best_ap = cands.pr_auc.max()
    reference = cands.loc[cands.pr_auc.idxmax(), 'model']
    delta = ci[ci.metric.str.startswith('pr_auc_minus_')].set_index('model')
    cands['tied_with_best'] = [delta.loc[n, 'ci_low'] <= 0 <= delta.loc[n, 'ci_high'] for n in cands.model]
    cands['tiered_cost'] = cands.model.map(tiers.total_estimated_business_cost)
    for i, col in enumerate(['complexity', 'explainability', 'operational_simplicity', 'deployment_complexity']):
        cands[col] = cands.kind.map(lambda k: QUALITATIVE[k][i])
    # Rule: among models statistically tied on PR-AUC, prefer the simplest, then lowest P95 latency.
    tied = cands[cands.tied_with_best]
    chosen = tied.sort_values(['components', 'p95_latency_ms']).iloc[0]
    hybrid = tied[tied.kind == 'fusion'].sort_values('pr_auc', ascending=False)
    winner = lambda col, asc: cands.sort_values(col, ascending=asc).iloc[0]
    facts = {'best_pr_auc': winner('pr_auc', False), 'best_recall': winner('recall', False),
             'fewest_fn': winner('false_negative', True), 'lowest_fpr': winner('false_positive_rate', True),
             'fastest_p95': winner('p95_latency_ms', True), 'lowest_cost': winner('tiered_cost', True)}
    pivot = types.pivot(index='fraud_type', columns='model', values='recall_at_5pct_review_budget')
    lines = ['# Final model selection (common final test set)', '',
        'Synthetic data only (100,000 transactions, seed 2026, 5.22% fraud). Not real upay production performance. '
        'All thresholds were selected on each model\'s own validation threshold block; the final test set was used once for reporting.', '',
        '## Decision table', '',
        '| Model | PR-AUC [95% CI] | Δ PR-AUC vs best [95% CI] | Recall | Precision | FPR | P95 ms (batch 1) | Tiered business cost | Components | Size MB | Recommendation |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in cands.sort_values('pr_auc', ascending=False).itertuples():
        c = ci[(ci.model == r.model) & (ci.metric == 'pr_auc')].iloc[0]
        dl = delta.loc[r.model]
        rec = 'SELECTED (primary)' if r.model == chosen.model else ('statistically tied, more complex' if r.tied_with_best else 'significantly lower PR-AUC')
        lines.append(f'| {r.model} | {r.pr_auc:.4f} [{c.ci_low:.4f}, {c.ci_high:.4f}] | {dl.estimate:+.4f} [{dl.ci_low:+.4f}, {dl.ci_high:+.4f}] | '
                     f'{r.recall:.3f} | {r.precision:.3f} | {r.false_positive_rate:.4f} | {r.p95_latency_ms:.2f} | {r.tiered_cost:,.0f} | '
                     f'{r.components} | {r.model_size_mb:.2f} | {rec} |')
    lines += ['', 'Recall/precision/FPR are at each model\'s validation max-F1 threshold (STEP_UP). Tiered cost applies validation-selected '
              'monitor/step-up/hold thresholds with the assumed costs in config.yaml (synthetic placeholders). Day-cluster bootstrap, 1000 resamples, paired.', '',
              '## Recall by fraud type at a matched 5% review budget', '',
              'Comparing at the same review volume removes threshold-placement effects.', '',
              '| Fraud type | ' + ' | '.join(pivot.columns) + ' |', '|---|' + '---:|'*len(pivot.columns)]
    lines += [f'| {t} | ' + ' | '.join(f'{v:.3f}' for v in row) + ' |' for t, row in pivot.iterrows()]
    f = facts
    lines += ['', '## Answers', '',
        f'- **Which model won on the common test set?** {chosen.model}: statistically tied for best PR-AUC and the simplest of the tied group.',
        f'- **Best PR-AUC:** {f["best_pr_auc"].model} ({f["best_pr_auc"].pr_auc:.4f}). Tied within bootstrap CI: {", ".join(tied.model)}.',
        f'- **Best fraud recall (at validation thresholds):** {f["best_recall"].model} ({f["best_recall"].recall:.4f}). It flags more traffic (FPR {f["best_recall"].false_positive_rate:.4f}); at a matched 5% review budget recall is within ~0.006 across the supervised/hybrid models.',
        f'- **Fewest false negatives:** {f["fewest_fn"].model} ({int(f["fewest_fn"].false_negative)}).',
        f'- **Lowest false-positive rate:** {f["lowest_fpr"].model} ({f["lowest_fpr"].false_positive_rate:.4f}).',
        f'- **Lowest inference latency (P95 batch 1):** {f["fastest_p95"].model} ({f["fastest_p95"].p95_latency_ms:.2f} ms); fastest ML model: '
        f'{cands[cands.kind != "rules"].sort_values("p95_latency_ms").iloc[0].model} ({cands[cands.kind != "rules"].p95_latency_ms.min():.2f} ms).',
        f'- **Lowest estimated business cost (validation tiers):** {f["lowest_cost"].model} ({f["lowest_cost"].tiered_cost:,.0f}). Under the assumed costs (1000 per missed fraud vs ~7 per step-up), '
        'the validation cost-optimal policy steps up 33–100% of all transactions, and rules-only "wins" by flagging everything; cost alone cannot select a model without a review-capacity constraint.',
        f'- **Is the hybrid improvement worth the complexity?** No. The best hybrid ({hybrid.iloc[0].model if len(hybrid) else "n/a"}) differs from {chosen.model} by '
        f'{(hybrid.iloc[0].pr_auc - chosen.pr_auc) if len(hybrid) else 0:+.4f} PR-AUC (CI includes zero), adds 3–4 components and roughly doubles P95 latency. '
        'Its mule/social-engineering recall advantage at matched budget is ≤0.02 absolute.',
        f'- **Should LightGBM be the primary production/demo model?** Yes, {chosen.model}.',
        '- **Supporting layers that remain active:** rule engine (independent fallback and hard blocks), Isolation Forest behavioral anomaly score '
        '(supporting evidence; abstains on cold start), device risk, recipient risk and graph/mule risk scores (decision inputs and explanations).', '',
        '## Limitations', '',
        '- Three saved bundles use isotonic calibration (XGBoost_500k, FullHybrid_500k, LightGBM_500k), which collapses scores into few tied levels '
        '(LightGBM_500k: 240 distinct values, max 0.389, below its validation hold threshold, so it never emits HOLD). Their PR-AUC is evaluated as deployed, '
        'with ties; part of their gap may be calibration resolution rather than a weaker ranker. Matched-budget recall uses stable index order within ties.',
        '- One training seed per model, one synthetic simulator; CIs capture test sampling only, not training variance.',
        '- Latency is in-process on a prepared feature frame (round-robin interleaved timing); API latency is in final_integration_benchmark.md.', '',
        '## Recommendation', '',
        f'Primary real-time fraud model: **LightGBM (trained on 200k), `{chosen.artifact}`**, recommended STEP_UP threshold **{chosen.recommended_threshold:.4f}** (validation max F1).', '',
        'Supporting layers: Rule Engine, Isolation Forest, Recipient Risk, Device Risk, Graph Risk.', '',
        f'Reason: {chosen.model} achieved PR-AUC statistically indistinguishable from the best hybrids on the common test set while being a single component, '
        f'~{cands[cands.kind == "fusion"].p95_latency_ms.median()/chosen.p95_latency_ms:.1f}x faster at P95 and simpler to deploy and explain (exact TreeSHAP).', '']
    (ROOT/'reports/final_model_selection.md').write_text('\n'.join(lines), encoding='utf-8')
    dump(R/'final_selection.json', {'selected_model': chosen.model, 'scale': chosen.scale, 'artifact_model': Path(chosen.artifact).stem,
        'recommended_threshold': chosen.recommended_threshold, 'tied_with_best_pr_auc': list(tied.model),
        'rule': 'among models whose paired PR-AUC delta CI vs best includes 0, choose fewest components then lowest P95 latency',
        'answers': {k: {'model': v.model} for k, v in facts.items()}})
    print(f'Selected {chosen.model} (tied group: {", ".join(tied.model)})')


if __name__ == '__main__':
    main()
