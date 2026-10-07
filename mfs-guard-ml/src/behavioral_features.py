"""Customer-relative baselines using strictly previous observations."""
import numpy as np


def baseline(values, current, fallback):
    count = len(values)
    mean = float(np.mean(values)) if count else fallback
    median = float(np.median(values)) if count else fallback
    std = float(np.std(values)) if count > 1 else fallback * .7
    return {'behavioral_history_count': count, 'user_mean_amount': mean,
        'user_median_amount': median, 'user_std_amount': std,
        'amount_vs_user_mean': current / max(mean, 1),
        'amount_vs_user_median': current / max(median, 1),
        'amount_zscore_user': (current - mean) / max(std, 1),
        'amount_percentile_user': float(np.mean(np.asarray(values) < current)) if count else .5}
