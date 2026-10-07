"""STEP 2: score saved finalists on the common final test set without refitting.

Every threshold comes from the model's own validation threshold block
(results/<tag>/threshold_metrics.csv). The final test set is used for reporting only.
"""
import json
import subprocess
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .config import ROOT, config, dump
from .evaluate import metrics

OUT = ROOT/'results/final_common_test'
TEST = ROOT/'data/final_test/mfs_final_test_100k.parquet'
# (label, scale, saved bundle, role). Required finalists first; then other strong
# saved systems; IsolationForest is a supporting-layer reference only.
FINALISTS = [
    ('LightGBM_200k', '200k', 'LightGBM', 'finalist'),
    ('Hybrid_WithoutSequence_500k', '500k', 'Hybrid_WithoutSequence', 'finalist'),
    ('XGBoost_500k', '500k', 'XGBoost', 'finalist'),
    ('FullHybrid_500k', '500k', 'FullHybrid', 'finalist'),
    ('RulesOnly_500k', '500k', 'RulesOnly', 'finalist'),
    ('LightGBM_500k', '500k', 'LightGBM', 'additional'),
    ('Hybrid_AddAnomaly_500k', '500k', 'Hybrid_AddAnomaly', 'additional'),
    ('Hybrid_RulesSupervised_500k', '500k', 'Hybrid_RulesSupervised', 'additional'),
    ('IsolationForest_500k', '500k', 'IsolationForest', 'supporting_layer_reference'),
]
MONITOR_REVIEW_BUDGET, HOLD_REVIEW_BUDGET = .15, .01


def operating_points(tag, name, bundle):
    """Validation-only thresholds from the model's own threshold block."""
    curve = pd.read_csv(ROOT/f'results/{tag}/threshold_metrics.csv')
    curve = curve[(curve.model == name) & (curve.threshold <= 1)]
    def budget(limit):
        return float(curve[curve.review_volume <= limit].sort_values(['recall', 'precision'], ascending=False).iloc[0].threshold)
    best_f1 = curve.sort_values(['f1', 'precision'], ascending=False).iloc[0]
    step_up = float(best_f1.threshold)
    return {'cost_optimal': float(bundle.threshold), 'monitor': min(budget(MONITOR_REVIEW_BUDGET), step_up),
            'step_up': step_up, 'hold': max(budget(HOLD_REVIEW_BUDGET), step_up),
            'validation_f1_at_step_up': float(best_f1.f1), 'validation_precision_at_step_up': float(best_f1.precision),
            'validation_recall_at_step_up': float(best_f1.recall), 'validation_rows': int(best_f1[['true_positive', 'false_positive', 'true_negative', 'false_negative']].sum())}


def tier_actions(p, points, monitor=True):
    action = np.full(len(p), 'APPROVE', dtype=object)
    if monitor:
        action[p >= points['monitor']] = 'APPROVE_AND_MONITOR'
    action[p >= points['step_up']] = 'STEP_UP_AUTH'
    action[p >= points['hold']] = 'TEMPORARY_HOLD'
    return action


def business_cost(y, action, cfg):
    """Assumed synthetic costs from config.yaml; not real upay economics.

    Residual fraud loss per action uses costs.actions; step-up and hold carry their
    operating costs; legitimate customers interrupted carry friction, and held
    legitimate customers also carry analyst investigation cost.
    """
    c = cfg['costs']
    residual = {'APPROVE': c['actions']['APPROVE'][0], 'APPROVE_AND_MONITOR': c['actions']['MONITOR'][0],
                'STEP_UP_AUTH': c['actions']['STEP_UP_AUTH'][0], 'TEMPORARY_HOLD': c['actions']['HOLD'][0]}
    y = np.asarray(y)
    fraud, legit = y == 1, y == 0
    step, hold = action == 'STEP_UP_AUTH', action == 'TEMPORARY_HOLD'
    loss = c['average_fraud_loss']*sum(residual[a]*np.sum(fraud & (action == a)) for a in residual)
    fp = np.sum(legit & (step | hold))*c['customer_friction_cost'] + np.sum(legit & hold)*c['false_positive_investigation_cost']
    verification, holding = step.sum()*c['step_up_authentication_cost'], hold.sum()*c['temporary_hold_cost']
    total = loss + fp + verification + holding
    return {'estimated_fraud_loss': float(loss), 'estimated_false_positive_cost': float(fp),
            'estimated_verification_cost': float(verification), 'estimated_hold_cost': float(holding),
            'total_estimated_business_cost': float(total), 'cost_per_transaction': float(total/len(y)),
            'approve_count': int(np.sum(action == 'APPROVE')), 'monitor_count': int(np.sum(action == 'APPROVE_AND_MONITOR')),
            'step_up_count': int(step.sum()), 'hold_count': int(hold.sum())}


def latency(bundles, frame):
    """Round-robin timing: every model sees the same rows, machine state and warm-up."""
    rows = {name: [] for name in bundles}
    for batch, repeats in [(1, 300), (10, 100), (100, 40), (1000, 12)]:
        times = {name: [] for name in bundles}
        for name, bundle in bundles.items():
            bundle.predict(frame.iloc[-batch:])
        for r in range(repeats):
            # Alternate tail rows so batch-1 timings are not a single cached row.
            sample = frame.iloc[len(frame)-batch-(r % 50):len(frame)-(r % 50)]
            for name, bundle in bundles.items():
                start = time.perf_counter()
                bundle.predict(sample)
                times[name].append((time.perf_counter()-start)*1000)
        for name, t in times.items():
            rows[name].append({'batch_size': batch, 'repeats': repeats, 'mean_ms': np.mean(t), 'p50_ms': np.percentile(t, 50),
                               'p95_ms': np.percentile(t, 95), 'p99_ms': np.percentile(t, 99),
                               'transactions_per_second': 1000*batch/np.mean(t)})
    return rows


def budget_recall(y, p, budget):
    """Recall/precision when reviewing the top `budget` share by score (ranking only, no tuning).

    Isotonic calibration creates large tied plateaus, so the boundary tie group is
    included fractionally: the expected result under random tie-breaking.
    """
    k = max(1, int(round(budget*len(p))))
    v = np.sort(p)[::-1][k-1]
    above, tied = p > v, p == v
    weight = above + tied*(k-above.sum())/tied.sum()
    return weight, float(np.sum(weight*y)/max(np.sum(y), 1)), float(np.sum(weight*y)/k)


MEMORY_PROBE = '''
import sys, gc, psutil, joblib, pandas as pd
sys.path.insert(0, sys.argv[3])
import src.risk_fusion
frame = pd.read_parquet(sys.argv[2]).tail(1000)
gc.collect(); before = psutil.Process().memory_info().rss
bundle = joblib.load(sys.argv[1]); bundle.predict(frame)
gc.collect(); print((psutil.Process().memory_info().rss - before)/1024**2)
'''


def memory_mb(path):
    """Process RSS growth after loading and scoring 1000 rows in a fresh interpreter."""
    out = subprocess.run([sys.executable, '-c', MEMORY_PROBE, str(path), str(TEST), str(ROOT)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def complexity(bundle):
    def trees(model):
        if hasattr(model, 'booster_'):
            return model.booster_.num_trees()
        if hasattr(model, 'get_booster'):
            return len(model.get_booster().get_dump())
        return 0
    if bundle.kind == 'supervised':
        return {'components': 1, 'component_list': 'preprocessor+tree model+calibration', 'trees': trees(bundle.model),
                'required_raw_features': len(bundle.columns)}
    if bundle.kind == 'rules':
        return {'components': 1, 'component_list': 'weighted rules+calibration', 'trees': 0,
                'required_raw_features': len(config()['rules'])}
    if bundle.kind == 'anomaly':
        return {'components': 1, 'component_list': 'isolation forest+calibration', 'trees': len(bundle.anomaly.model.estimators_),
                'required_raw_features': len(bundle.anomaly.columns)}
    uses_anomaly = 2 in bundle.meta_columns
    parts = ['supervised base', 'rule engine', *(['isolation forest'] if uses_anomaly else []), 'logistic fusion', 'calibration']
    columns = set(bundle.supervised.columns) | {'rule_risk_score', 'device_risk_score', 'recipient_risk_score',
        'mule_network_score', 'sequence_risk_score', 'behavioral_model_active'} | (set(bundle.anomaly.columns) if uses_anomaly else set())
    return {'components': len(parts), 'component_list': '+'.join(parts),
            'trees': trees(bundle.supervised.model) + (len(bundle.anomaly.model.estimators_) if uses_anomaly else 0),
            'required_raw_features': len(columns) + len(config()['rules'])}


def weighted_ap(order, endpoints, y, w):
    ws, ys = w[order], y[order]
    tp = np.cumsum(ws*ys)[endpoints]
    total = np.cumsum(ws)[endpoints]
    if tp[-1] == 0:
        return np.nan
    return np.sum(np.diff(np.r_[0, tp])*np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0))/tp[-1]


def bootstrap(y, scores, thresholds, groups, reference, repeats=1000, seed=2026):
    """Day-cluster bootstrap; identical resamples for every model, so deltas are paired."""
    rng = np.random.default_rng(seed)
    unique, index = np.unique(groups, return_inverse=True)
    ranks = {}
    for name, p in scores.items():
        order = np.argsort(-p, kind='stable')
        ranks[name] = (order, np.r_[np.flatnonzero(np.diff(p[order])), len(p)-1])
    samples = {name: {'pr_auc': [], 'recall': [], 'precision': [], 'f1': [], 'delta': []} for name in scores}
    for _ in range(repeats):
        w = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[index].astype(float)
        ref_ap = weighted_ap(*ranks[reference], y, w)
        for name, p in scores.items():
            pred = p >= thresholds[name]
            tp, fp, fn = np.sum(w*(pred & (y == 1))), np.sum(w*(pred & (y == 0))), np.sum(w*(~pred & (y == 1)))
            precision, recall = tp/max(tp+fp, 1e-12), tp/max(tp+fn, 1e-12)
            ap = weighted_ap(*ranks[name], y, w)
            s = samples[name]
            s['pr_auc'].append(ap)
            s['recall'].append(recall)
            s['precision'].append(precision)
            s['f1'].append(2*precision*recall/max(precision+recall, 1e-12))
            s['delta'].append(ap-ref_ap)
    rows = []
    for name, s in samples.items():
        pred = scores[name] >= thresholds[name]
        point = metrics(y, scores[name], thresholds[name], config())
        for metric in ['pr_auc', 'recall', 'precision', 'f1']:
            v = np.asarray(s[metric])
            rows.append({'model': name, 'metric': metric, 'estimate': point[metric],
                         'ci_low': np.nanquantile(v, .025), 'ci_high': np.nanquantile(v, .975)})
        d = np.asarray(s['delta'])
        rows.append({'model': name, 'metric': f'pr_auc_minus_{reference}',
                     'estimate': average_precision_score(y, scores[name])-average_precision_score(y, scores[reference]),
                     'ci_low': np.nanquantile(d, .025), 'ci_high': np.nanquantile(d, .975)})
    table = pd.DataFrame(rows)
    table['bootstrap_unit'], table['repeats'], table['threshold_source'] = 'UTC day of final test set', repeats, 'validation max-F1 (STEP_UP)'
    return table


def segments(d):
    s = pd.DataFrame(index=d.index)
    s['fraud_type'] = d.fraud_type
    s['transaction_type'] = d.transaction_type
    s['channel'] = d.channel
    s['amount_range'] = pd.cut(d.amount, [0, 500, 2000, 10000, np.inf], labels=['<=500', '500-2k', '2k-10k', '>10k']).astype(str)
    s['device'] = np.where(d.device_is_new == 1, 'new_device', 'known_device')
    s['recipient'] = np.where(d.first_time_recipient == 1, 'first_time_recipient', 'known_recipient')
    s['customer_history'] = np.where(d.behavioral_model_active == 0, 'cold_start_lt5_events', 'warm_history')
    s['account_age'] = pd.cut(d.account_age_days, [0, 90, 365, 1000, np.inf], labels=['<=90d', '91-365d', '1-2.7y', '>2.7y']).astype(str)
    s['night'] = np.where(d.is_night_transaction == 1, 'night', 'day')
    s['velocity'] = np.where(d.tx_count_5m >= 2, 'high_velocity_ge2_in_5m', 'normal_velocity')
    s['graph_mule_risk'] = pd.cut(d.mule_network_score, [-1, 20, 50, 101], labels=['low_<=20', 'medium_20-50', 'high_>50']).astype(str)
    return s


def error_analysis(d, y, scores, thresholds, names):
    seg = segments(d)
    rows = []
    for name in names:
        pred = scores[name] >= thresholds[name]
        for dimension in seg:
            for value, ix in seg.groupby(dimension).groups.items():
                ix = d.index.get_indexer(ix)
                yy, pp = y[ix], pred[ix]
                tp, fp, fn, tn = [int(v) for v in (np.sum(yy & pp), np.sum(~yy & pp), np.sum(yy & ~pp), np.sum(~yy & ~pp))]
                rows.append({'model': name, 'dimension': dimension, 'value': value, 'rows': len(ix), 'fraud': tp+fn,
                             'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'recall': tp/(tp+fn) if tp+fn else np.nan,
                             'false_positive_rate': fp/(fp+tn) if fp+tn else np.nan,
                             'share_of_all_fp': fp/max(np.sum(~y & pred), 1), 'share_of_all_fn': fn/max(np.sum(y & ~pred), 1)})
    return pd.DataFrame(rows)


def run():
    cfg = config()
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(TEST)
    meta = json.loads((ROOT/'data/final_test/final_test_metadata.json').read_text())
    assert all(meta['critical_checks'].values()), 'Final test set failed critical validation'
    y = d.fraud_label.to_numpy().astype(bool)
    days = d.timestamp.dt.floor('D').astype('int64').to_numpy()
    rows, sweeps, cost_rows, latency_rows, type_rows = [], [], [], [], []
    scores, points = {}, {}
    bundles = {label: joblib.load(ROOT/f'models/{tag}/{name}.joblib') for label, tag, name, _ in FINALISTS}
    timings = latency(bundles, d)
    for label, tag, name, role in FINALISTS:
        path = ROOT/f'models/{tag}/{name}.joblib'
        bundle = bundles[label]
        started = time.perf_counter()
        p = bundle.predict(d)
        full_seconds = time.perf_counter()-started
        raw = bundle.raw(d)
        op = operating_points(tag, name, bundle)
        scores[label], points[label] = p, op
        lat = timings[label]
        for r in lat:
            latency_rows.append({'model': label, **r, 'scope': 'single process, 4 threads, prepared historical feature frame '
                                 'through preprocessing, all dependencies and calibration; excludes state lookup and HTTP'})
        info = complexity(bundle)
        recommended = metrics(y.astype(int), p, op['step_up'], cfg)
        cost_opt = metrics(y.astype(int), p, op['cost_optimal'], cfg)
        tiered = business_cost(y, tier_actions(p, op), cfg)
        rows.append({'model': label, 'scale': tag, 'artifact': f'models/{tag}/{name}.joblib', 'role': role, 'kind': bundle.kind,
            'calibration': bundle.calibration.method, 'recommended_threshold': op['step_up'],
            'threshold_source': 'max F1 on validation threshold block of its own training run',
            **{k: recommended[k] for k in ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'mcc', 'balanced_accuracy',
               'false_positive_rate', 'false_negative_rate', 'true_positive', 'false_positive', 'true_negative', 'false_negative',
               'brier_score', 'review_volume']},
            'raw_uncalibrated_pr_auc': average_precision_score(y, raw), 'distinct_calibrated_scores': int(len(np.unique(p))),
            'max_calibrated_probability': float(p.max()),
            'cost_optimal_threshold': op['cost_optimal'], 'cost_optimal_recall': cost_opt['recall'],
            'cost_optimal_precision': cost_opt['precision'], 'cost_optimal_false_positive_rate': cost_opt['false_positive_rate'],
            'cost_optimal_review_volume': cost_opt['review_volume'],
            'tiered_total_business_cost': tiered['total_estimated_business_cost'],
            **{f'{k}_at_{int(b*100)}pct_review_budget': v for b in (.01, .02, .05)
               for k, v in zip(('recall', 'precision'), budget_recall(y, p, b)[1:])},
            'p50_latency_ms': lat[0]['p50_ms'], 'p95_latency_ms': lat[0]['p95_ms'], 'p99_latency_ms': lat[0]['p99_ms'],
            'full_test_scoring_seconds': full_seconds, 'model_size_mb': path.stat().st_size/1024**2,
            'estimated_memory_mb': memory_mb(path), **info})
        for t in np.round(np.arange(.05, .951, .05), 2):
            m = metrics(y.astype(int), p, t, cfg)
            sweep_points = {**op, 'step_up': t, 'hold': max(t, op['hold'])}
            c = business_cost(y, tier_actions(p, sweep_points, monitor=False), cfg)
            sweeps.append({'model': label, 'threshold': t, 'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'],
                'false_positives': m['false_positive'], 'false_negatives': m['false_negative'],
                'review_volume': m['true_positive']+m['false_positive'], 'review_rate': m['review_volume'],
                'step_up_volume': c['step_up_count'], 'hold_volume': c['hold_count'],
                'estimated_business_cost': c['total_estimated_business_cost'],
                'is_recommended_validation_threshold': False})
        sweeps.append({'model': label, 'threshold': op['step_up'], 'precision': recommended['precision'], 'recall': recommended['recall'],
            'f1': recommended['f1'], 'false_positives': recommended['false_positive'], 'false_negatives': recommended['false_negative'],
            'review_volume': recommended['true_positive']+recommended['false_positive'], 'review_rate': recommended['review_volume'],
            'step_up_volume': tiered['step_up_count'], 'hold_volume': tiered['hold_count'],
            'estimated_business_cost': business_cost(y, tier_actions(p, op, monitor=False), cfg)['total_estimated_business_cost'],
            'is_recommended_validation_threshold': True})
        cost_rows.append({'model': label, 'policy': 'validation tiers: monitor/step-up/hold', 'monitor_threshold': op['monitor'],
                          'step_up_threshold': op['step_up'], 'hold_threshold': op['hold'], **tiered})
        cost_rows.append({'model': label, 'policy': 'validation cost-optimal binary (all flagged step-up)',
                          'step_up_threshold': op['cost_optimal'],
                          **business_cost(y, tier_actions(p, {**op, 'step_up': op['cost_optimal'], 'hold': 2}, monitor=False), cfg)})
        budget_hit = budget_recall(y, p, .05)[0]
        for fraud_type in sorted(set(d.fraud_type) - {'LEGITIMATE'}):
            mask = (d.fraud_type == fraud_type).to_numpy()
            type_rows.append({'model': label, 'fraud_type': fraud_type, 'count': int(mask.sum()),
                'recall_at_recommended_threshold': float(np.mean(p[mask] >= op['step_up'])),
                'recall_at_5pct_review_budget': float(np.mean(budget_hit[mask])),  # expected, tie-aware
                'recall_at_hold_threshold': float(np.mean(p[mask] >= op['hold'])),
                'recall_at_cost_optimal_threshold': float(np.mean(p[mask] >= op['cost_optimal'])),
                'mean_probability': float(p[mask].mean())})
        print(f'{label}: PR-AUC {recommended["pr_auc"]:.4f} recall {recommended["recall"]:.3f} '
              f'precision {recommended["precision"]:.3f} P95 {lat[0]["p95_ms"]:.2f} ms', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'model_comparison.csv', index=False)
    dump(OUT/'model_comparison.json', {'test_set': 'data/final_test/mfs_final_test_100k.parquet', 'rows': len(d),
         'fraud_prevalence': float(y.mean()), 'seed': meta['seed'], 'synthetic_only': True,
         'threshold_policy': 'All thresholds selected on each model\'s own validation threshold block; final test used once for reporting.',
         'operating_points': points, 'models': table.to_dict('records')})
    pd.DataFrame(sweeps).to_csv(OUT/'threshold_comparison.csv', index=False)
    pd.DataFrame(cost_rows).to_csv(OUT/'business_cost_comparison.csv', index=False)
    pd.DataFrame(latency_rows).to_csv(OUT/'latency_comparison.csv', index=False)
    pd.DataFrame(type_rows).to_csv(OUT/'fraud_type_performance.csv', index=False)
    finalists = [r[0] for r in FINALISTS if r[3] != 'supporting_layer_reference']
    reference = table[table.model.isin(finalists)].sort_values('pr_auc', ascending=False).model.iloc[0]
    thresholds = {k: v['step_up'] for k, v in points.items()}
    ci = bootstrap(y.astype(int), {k: scores[k] for k in finalists}, thresholds, days, reference)
    ci.to_csv(OUT/'confidence_intervals.csv', index=False)
    top = table[table.role == 'finalist'].sort_values('pr_auc', ascending=False).model.head(4).tolist()
    errors = error_analysis(d, y, scores, thresholds, top)
    errors.to_csv(OUT/'error_analysis.csv', index=False)
    error_report(errors, top)
    predictions = d[['transaction_id', 'fraud_label', 'fraud_type']].copy()
    for k, v in scores.items():
        predictions[k] = v.astype('float32')
    predictions.to_parquet(OUT/'final_test_predictions.parquet', index=False)
    return table


def error_report(errors, names):
    lines = ['# Common final test: error analysis', '',
             'Synthetic data only. Errors are at each model\'s validation-selected STEP_UP threshold (max validation F1). '
             'Counts are descriptive, not causal.', '']
    for dimension in errors.dimension.unique():
        part = errors[errors.dimension == dimension]
        lines += [f'## {dimension}', '', '| Value | ' + ' | '.join(f'{n} recall / FPR / FN / FP' for n in names) + ' |',
                  '|---|' + '---:|'*len(names)]
        for value in part.value.unique():
            cells = []
            for n in names:
                r = part[(part.model == n) & (part.value == value)].iloc[0]
                recall = '—' if pd.isna(r.recall) else f'{r.recall:.3f}'
                fpr = '—' if pd.isna(r.false_positive_rate) else f'{r.false_positive_rate:.3f}'
                cells.append(f'{recall} / {fpr} / {r.fn} / {r.fp}')
            lines.append(f'| {value} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lead = names[0]
    e = errors[(errors.model == lead) & (errors.dimension != 'fraud_type')]
    fn = errors[(errors.model == lead) & (errors.dimension == 'fraud_type')].sort_values('fn', ascending=False).iloc[0]
    fp_top = e.sort_values('share_of_all_fp', ascending=False).head(4)
    fn_top = e[e.fraud > 0].sort_values('recall').head(4)
    lines += ['## Observations', '',
              f'- Largest missed-fraud category for {lead}: **{fn.value}** ({fn.fn} false negatives, recall {fn.recall:.3f}).',
              '- Segments carrying the largest share of false positives: ' + '; '.join(
                  f'{r.dimension}={r.value} ({r.share_of_all_fp:.1%})' for r in fp_top.itertuples()) + '.',
              '- Lowest-recall segments: ' + '; '.join(f'{r.dimension}={r.value} (recall {r.recall:.3f}, n fraud {r.fraud})'
                                                       for r in fn_top.itertuples()) + '.', '']
    (ROOT/'reports/final_common_test_error_analysis.md').write_text('\n'.join(lines), encoding='utf-8')
