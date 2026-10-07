import numpy as np
import pandas as pd
from src.feature_engineering import engineer, haversine
from src.config import config


def original_columns(d):
    return list(d.columns[:d.columns.get_loc('balance_change_ratio')+1])


def test_prefix_and_target_invariance(dataset):
    original = dataset[original_columns(dataset)].copy()
    prefix = engineer(original.iloc[:600], config())
    compare_cols = list(prefix.columns)
    pd.testing.assert_frame_equal(prefix, dataset.iloc[:600][compare_cols], check_dtype=False, atol=1e-3, rtol=1e-5)
    changed = original.iloc[:600].copy()
    changed['fraud_label'] = 1-changed.fraud_label
    changed['fraud_type'] = 'NOT_A_FEATURE'
    rerun = engineer(changed, config())
    safe = [c for c in prefix if c not in ['fraud_label', 'fraud_type']]
    pd.testing.assert_frame_equal(prefix[safe], rerun[safe])


def test_geodesic_known_distance():
    assert np.isclose(haversine(0, 0, 0, 1), 111.195, atol=.01)


def test_current_amount_does_not_change_own_baseline(dataset):
    original = dataset[original_columns(dataset)].iloc[:200].copy()
    before = engineer(original, config())
    original.loc[199, 'amount'] *= 10
    after = engineer(original, config())
    for c in ['user_mean_amount', 'user_median_amount', 'tx_count_1h', 'amount_sum_1h', 'receiver_in_degree']:
        assert before.iloc[-1][c] == after.iloc[-1][c]
