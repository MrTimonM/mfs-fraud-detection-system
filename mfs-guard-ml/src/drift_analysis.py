"""Observed train/test shift under gradual simulated amount inflation."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def drift(train, test, out):
    rows = []
    for col in ['amount', 'tx_count_1h', 'device_risk_score', 'amount_vs_user_mean', 'recipient_risk_score']:
        bins = np.unique(np.r_[-np.inf, train[col].quantile(np.linspace(.1, .9, 9)), np.inf])
        a = np.maximum(np.histogram(train[col], bins)[0]/len(train), 1e-6)
        b = np.maximum(np.histogram(test[col], bins)[0]/len(test), 1e-6)
        rows.append({'feature': col, 'psi': np.sum((b-a)*np.log(b/a)),
                     'ks_statistic': ks_2samp(train[col], test[col]).statistic})
    pd.DataFrame(rows).to_csv(out/'drift_analysis.csv', index=False)
