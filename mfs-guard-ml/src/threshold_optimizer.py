"""Thresholds selected on the last validation block; simulated costs only."""
import numpy as np
import pandas as pd
from .evaluate import metrics


def optimize(y, p, cfg):
    thresholds = np.unique(np.r_[np.linspace(.005, .995, 100), np.quantile(p, np.linspace(0, 1, 81)), 1.000001])
    # Sorting-based discrimination metrics are threshold invariant: compute once.
    base = metrics(y, p, .5, cfg)
    y, p = np.asarray(y), np.asarray(p)
    positive, negative = y == 1, y == 0
    costs = cfg['costs']
    records = []
    for threshold in thresholds:
        predicted = p >= threshold
        tp, fp = int(np.sum(positive & predicted)), int(np.sum(negative & predicted))
        fn, tn = int(positive.sum())-tp, int(negative.sum())-fp
        precision, recall = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
        denom = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
        cost = fn*costs['average_fraud_loss'] + fp*(costs['false_positive_investigation_cost']+costs['customer_friction_cost'])
        records.append({**base, 'threshold': float(threshold), 'precision': precision, 'recall': recall,
            'accuracy': (tp+tn)/len(y), 'balanced_accuracy': .5*(recall+tn/max(tn+fp, 1)),
            'f1': 2*precision*recall/max(precision+recall, 1e-15), 'mcc': (tp*tn-fp*fn)/denom if denom else 0,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
            'false_positive_rate': fp/max(fp+tn, 1), 'false_negative_rate': fn/max(fn+tp, 1),
            'expected_business_cost': cost, 'cost_per_transaction': cost/len(y), 'review_volume': float(predicted.mean())})
    table = pd.DataFrame(records)
    winner = table.sort_values(['expected_business_cost', 'false_positive_rate']).iloc[0]
    return float(winner.threshold), table


def actions(p, cfg):
    names, parameters = zip(*cfg['costs']['actions'].items())
    p = np.asarray(p)
    cost = np.column_stack([p*residual*cfg['costs']['average_fraud_loss']+(1-p)*legit
                           for residual, legit in parameters])
    return np.asarray(names)[np.argmin(cost, axis=1)]
