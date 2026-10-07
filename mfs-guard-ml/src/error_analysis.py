"""Observed errors by fraud type and synthetic operational segments."""
import numpy as np
import pandas as pd


def analyze(frame, p, threshold, out):
    d = frame.copy()
    d['prediction'] = p >= threshold
    d['amount_band'] = pd.cut(d.amount, [0, 500, 2000, 10000, np.inf]).astype(str)
    d['history_band'] = pd.cut(d.behavioral_history_count, [-1, 4, 20, 100, np.inf]).astype(str)
    d['account_age_group'] = pd.cut(d.account_age_days, [0, 90, 365, 1000, np.inf]).astype(str)
    d['region'] = pd.cut(d.latitude, [0, 23, 24, 25, 90]).astype(str)
    rows = []
    for column in ['fraud_type', 'transaction_type', 'channel', 'amount_band', 'device_is_new',
                   'first_time_recipient', 'history_band', 'transaction_hour', 'customer_segment',
                   'account_age_group', 'region', 'rooted_device']:
        for value, group in d.groupby(column, observed=True):
            y, pred = group.fraud_label.to_numpy(), group.prediction.to_numpy()
            tp, fp, fn, tn = [int(v.sum()) for v in ((y==1)&pred, (y==0)&pred, (y==1)&~pred, (y==0)&~pred)]
            rows.append({'dimension': column, 'value': str(value), 'rows': len(group), 'tp': tp, 'fp': fp,
                'fn': fn, 'tn': tn, 'recall': tp/(tp+fn) if tp+fn else np.nan,
                'false_positive_rate': fp/(fp+tn) if fp+tn else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(out/'error_analysis.csv', index=False)
    result[result.dimension.isin(['customer_segment', 'channel', 'region', 'account_age_group', 'rooted_device'])].to_csv(out/'synthetic_segment_analysis.csv', index=False)
    fp = result[result.dimension == 'channel'].sort_values('fp', ascending=False).iloc[0]
    fn = result[result.dimension == 'fraud_type'].sort_values('fn', ascending=False).iloc[0]
    message = f'Largest false-positive channel count: {fp.value} ({fp.fp}). Largest missed fraud category: {fn.value} ({fn.fn}). These are descriptive counts, not causal explanations.\n'
    (out/'error_analysis.md').write_text(message, encoding='utf-8')
    return message
