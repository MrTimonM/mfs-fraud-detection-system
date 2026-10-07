"""Separated calibration fitting and calibration-family selection."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score


class ProbabilityMap:
    def __init__(self, method='identity'):
        self.method, self.model = method, None

    def fit(self, p, y):
        if self.method == 'sigmoid':
            self.model = LogisticRegression(C=100, max_iter=300).fit(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1), y)
        elif self.method == 'isotonic':
            self.model = IsotonicRegression(out_of_bounds='clip').fit(p, y)
        return self

    def predict(self, p):
        if self.method == 'sigmoid':
            return self.model.predict_proba(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1]
        if self.method == 'isotonic':
            return self.model.predict(p)
        return np.clip(p, 0, 1)


def calibrate(fit_p, fit_y, select_p, select_y):
    candidates, report = [], []
    base_ap = average_precision_score(select_y, select_p)
    for method in ('identity', 'sigmoid', 'isotonic'):
        mapping = ProbabilityMap(method).fit(fit_p, fit_y)
        p = mapping.predict(select_p)
        ap = average_precision_score(select_y, p)
        row = {'method': method, 'brier_score': brier_score_loss(select_y, p),
               'pr_auc': ap, 'roc_auc': roc_auc_score(select_y, p), 'eligible': ap >= base_ap - .015}
        candidates.append(mapping)
        report.append(row)
    winner = min([i for i, r in enumerate(report) if r['eligible']], key=lambda i: report[i]['brier_score'])
    for i, r in enumerate(report):
        r['selected'] = i == winner
    return candidates[winner], report
