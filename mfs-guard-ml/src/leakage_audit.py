"""Explicit feature inventory and exclusions; tests establish prefix invariance."""
import numpy as np

EXCLUDE = set('''transaction_id user_id receiver_id device_id agent_id merchant_id session_id
timestamp fraud_label fraud_type rule_risk_score triggered_rule_count highest_rule_severity
balance_after balance_change_ratio home_latitude home_longitude typical_transaction_amount
wallet_balance_tendency preferred_channel preferred_transaction_type normal_active_hour_start
normal_active_hour_end normal_location_radius_km'''.split())
GRAPH = ['sender_in_degree', 'sender_out_degree', 'receiver_in_degree', 'receiver_out_degree',
    'unique_sender_count', 'unique_receiver_count', 'connected_component_size',
    'reciprocal_transfer_ratio', 'shared_recipient_count', 'fund_concentration',
    'rapid_forwarding_ratio', 'shared_device_count', 'mule_network_score']
SEQUENCE = ['pin_reset_then_transfer', 'new_device_then_transfer', 'failed_pin_then_success',
    'balance_inquiry_then_cashout', 'incoming_then_rapid_outgoing', 'new_recipient_then_large_transfer',
    'multiple_otp_then_transfer', 'device_change_then_high_value', 'sim_change_then_transfer',
    'credential_change_then_cashout', 'channel_hop_sequence', 'sequence_risk_score']
ANOMALY = ['amount_vs_user_mean', 'amount_vs_user_median', 'amount_zscore_user',
    'amount_percentile_user', 'unusual_hour_score', 'unusual_channel_score',
    'unusual_location_score', 'tx_count_1h', 'time_since_previous_tx_seconds']


def feature_columns(d):
    return [c for c in d if c not in EXCLUDE]


def audit(d):
    columns = feature_columns(d)
    assert not EXCLUDE.intersection(columns)
    correlations = d.select_dtypes('number').corrwith(d.fraud_label).dropna()
    suspicious = {c: float(v) for c, v in correlations.items() if c in columns and abs(v) > .95}
    assert not suspicious, f'Suspicious target correlation: {suspicious}'
    return {'status': 'PASSED', 'features': columns, 'excluded': sorted(EXCLUDE),
        'inventory': {c: ('excluded: outcome, identifier, simulator profile or rule aggregate' if c in EXCLUDE
            else 'observable pre-decision context or strictly previous event history') for c in d},
        'notes': ['Target columns removed at feature-engineering boundary.',
            'No current/future labels create risk scores or blacklists.',
            'Proposed amount and balance_before are available before decision; balance_after excluded.',
            'Synthetic authentication counts model pre-transaction telemetry, not ledger counts.',
            'Recipient graph inserted after current features; PageRank deliberately omitted.',
            'History includes previously observed test transactions, never their labels.'],
        'max_absolute_target_correlation': float(correlations.reindex(columns).abs().max())}
