"""Difficult observed synthetic fraud slices; no inconsistent post-hoc edits."""
import numpy as np
import pandas as pd


def adversarial(frame, scores, thresholds, out):
    fraud = frame.fraud_label.to_numpy() == 1
    slices = {
        'amount_just_below_2x_relative_threshold': (frame.amount_vs_user_mean >= 1.8) & (frame.amount_vs_user_mean < 2),
        'splitting_and_small_repeated_transfers': (frame.tx_count_5m >= 2) & (frame.amount_vs_user_mean < 1),
        'slow_fraud': frame.tx_count_1h == 0,
        'trusted_device_fraud': frame.device_is_new == 0,
        'mule_network_activity': frame.fraud_type == 'MULE_ACTIVITY',
        'known_device_account_takeover': (frame.device_is_new == 0) & (frame.fraud_type == 'ACCOUNT_TAKEOVER'),
        'normal_location_fraud': frame.distance_from_home_km < 10,
        'subtle_fraud': frame.fraud_type == 'SUBTLE_FRAUD',
    }
    records = []
    for name, mask in slices.items():
        active = np.asarray(mask) & fraud
        for model, p in scores.items():
            records.append({'scenario': name, 'model': model, 'fraud_count': int(active.sum()),
                'recall': float(np.mean(p[active] >= thresholds[model])) if active.any() else np.nan,
                'scope': 'observed difficult synthetic slice; not adaptive attacker simulation'})
    pd.DataFrame(records).to_csv(out/'adversarial_test_results.csv', index=False)
