"""70/15/15 chronology; five disjoint validation blocks avoid stack leakage."""
import numpy as np


def split(d):
    n = len(d)
    a, b = int(.7*n), int(.85*n)
    val = np.array_split(np.arange(a, b), 5)
    result = {'train': np.arange(a), 'tune': val[0], 'fusion': val[1],
              'calibrate': val[2], 'select': val[3], 'threshold': val[4], 'test': np.arange(b, n)}
    info = {k: {'rows': len(v), 'start': str(d.iloc[v[0]].timestamp),
                'end': str(d.iloc[v[-1]].timestamp), 'fraud_count': int(d.iloc[v].fraud_label.sum())}
            for k, v in result.items()}
    train_users = set(d.iloc[:a].user_id)
    info['user_overlap'] = {'test_user_seen_in_training_fraction': float(d.iloc[b:].user_id.isin(train_users).mean()),
        'interpretation': 'Returning-customer temporal evaluation; cold-start results reported separately.'}
    for v in result.values():
        assert d.iloc[v].fraud_label.nunique() == 2, 'Split must contain both classes'
    return result, info
