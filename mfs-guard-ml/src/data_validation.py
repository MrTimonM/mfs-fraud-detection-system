"""Fail-fast schema, accounting and independent sampled history checks."""
import numpy as np
from .entity_generator import TYPES, CHANNELS
from .fraud_scenarios import FRAUD_TYPES
from .feature_engineering import haversine

MANDATORY = '''transaction_id user_id receiver_id device_id transaction_type channel amount fee
balance_before balance_after depletion_ratio failed_pin_attempts otp_resend_count
pin_reset_recently device_is_new rooted_device emulator_detected vpn_active
screen_share_detected channel_changed_recently tx_count_5m tx_count_15m tx_count_1h
tx_count_24h amount_sum_5m amount_sum_1h balance_inquiry_count_5m turnaround_latency_seconds
latitude longitude impossible_travel impossible_travel_speed recipient_risk_score device_risk_score
user_behavior_score blacklisted_device blacklisted_agent rule_risk_score timestamp fraud_label fraud_type'''.split()


def validate(d, rows):
    assert set(MANDATORY) <= set(d), 'Missing mandatory columns'
    assert len(d) == rows and d.transaction_id.is_unique
    assert d.timestamp.notna().all() and d.timestamp.is_monotonic_increasing
    assert (d.timestamp.diff().dropna().dt.total_seconds() > 0).all()
    assert d.fraud_label.isin([0, 1]).all()
    assert d.fraud_type.isin(FRAUD_TYPES).all()
    assert np.array_equal(d.fraud_label, (d.fraud_type != 'LEGITIMATE').astype(int))
    assert d.transaction_type.isin(TYPES).all() and d.channel.isin(CHANNELS).all()
    assert (d.amount > 0).all() and (d.balance_after >= 0).all() and (d.fee >= 0).all()
    numeric = d.select_dtypes('number')
    assert np.isfinite(numeric.to_numpy()).all(), 'Nonfinite numeric values'
    assert not d[MANDATORY].isna().any().any()
    outgoing = ~d.transaction_type.isin(['CASH_IN', 'REMITTANCE'])
    expected = np.where(outgoing, d.balance_before-d.amount-d.fee, d.balance_before+d.amount)
    assert np.allclose(d.balance_after, expected, atol=.15, rtol=1e-5), 'Balance equation'
    assert np.allclose(d.depletion_ratio, (d.amount+d.fee)/np.maximum(d.balance_before, 1e-6), atol=1e-4)
    checks = 0
    for u in d.user_id.drop_duplicates().head(8):
        hist = d[d.user_id == u]
        for idx in np.linspace(0, len(hist)-1, min(12, len(hist))).astype(int):
            row = hist.iloc[idx]
            past = hist.iloc[:idx]
            seconds = (row.timestamp - past.timestamp).dt.total_seconds()
            for suffix, window in [('5m', 300), ('15m', 900), ('1h', 3600), ('24h', 86400)]:
                relevant = past[(seconds <= window) & (seconds > 0)]
                assert row[f'tx_count_{suffix}'] == len(relevant), 'Window count'
                assert np.isclose(row[f'amount_sum_{suffix}'], relevant.amount.sum(), atol=.2, rtol=1e-5)
            if idx:
                assert np.isclose(row.user_mean_amount, past.amount.mean(), rtol=1e-5)
                assert np.isclose(row.distance_from_previous_tx_km,
                    haversine(past.iloc[-1].latitude, past.iloc[-1].longitude, row.latitude, row.longitude), atol=.005)
            checks += 1
    prevalence = float(d.fraud_label.mean())
    if rows >= 100000:
        assert .04 <= prevalence <= .06, f'Prevalence {prevalence}'
    return {'status': 'PASSED', 'rows': rows, 'columns': len(d.columns),
        'fraud_prevalence': prevalence, 'sampled_history_checks': checks,
        'fraud_types': d.fraud_type.value_counts().to_dict(),
        'missing_values': d.isna().sum().to_dict(), 'duplicates': int(d.duplicated().sum()),
        'amount_quantiles': d.amount.quantile([0, .5, .99, 1]).to_dict()}
