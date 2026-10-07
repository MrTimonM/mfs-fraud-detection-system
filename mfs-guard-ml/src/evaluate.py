"""Binary metrics and uncertainty without accuracy-driven selection."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, brier_score_loss)


def metrics(y, p, threshold, cfg):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    costs = cfg['costs']
    cost = fn*costs['average_fraud_loss'] + fp*(costs['false_positive_investigation_cost']+costs['customer_friction_cost'])
    return {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0), 'roc_auc': roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        'pr_auc': average_precision_score(y, p) if np.sum(y) else np.nan,
        'average_precision': average_precision_score(y, p) if np.sum(y) else np.nan,
        'mcc': matthews_corrcoef(y, pred), 'false_positive_rate': fp/max(fp+tn, 1),
        'false_negative_rate': fn/max(fn+tp, 1), 'true_positive': tp, 'false_positive': fp,
        'true_negative': tn, 'false_negative': fn, 'brier_score': brier_score_loss(y, np.clip(p, 0, 1)),
        'expected_business_cost': cost, 'cost_per_transaction': cost/len(y), 'threshold': threshold,
        'review_volume': float(np.mean(pred))}


def paired_bootstrap(y, full, alternative, groups, repeats=150, seed=42):
    """Cluster resampling by test day to retain within-day episode dependence."""
    rng = np.random.default_rng(seed)
    unique, group_index = np.unique(groups, return_inverse=True)
    def ranking(p):
        order = np.argsort(-p, kind='stable')
        endpoints = np.r_[np.flatnonzero(np.diff(p[order])), len(p)-1]
        return order, endpoints
    rankings = [ranking(full), ranking(alternative)]
    def weighted_ap(rank, weights):
        order, endpoints = rank
        w = weights[order]
        tp = np.cumsum(w*y[order])[endpoints]
        total = np.cumsum(w)[endpoints]
        if tp[-1] == 0:
            return np.nan
        return np.sum(np.diff(np.r_[0, tp])*np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total>0))/tp[-1]
    deltas = []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[group_index]
        delta = weighted_ap(rankings[0], weights)-weighted_ap(rankings[1], weights)
        if np.isfinite(delta):
            deltas.append(delta)
    return {'pr_auc_delta': average_precision_score(y, full)-average_precision_score(y, alternative),
        'delta_ci_low': np.quantile(deltas, .025), 'delta_ci_high': np.quantile(deltas, .975),
        'bootstrap_unit': 'UTC test day', 'bootstrap_repeats': len(deltas)}
