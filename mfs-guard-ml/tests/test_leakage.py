from src.leakage_audit import audit, feature_columns, EXCLUDE


def test_leakage_inventory(dataset):
    assert audit(dataset)['status'] == 'PASSED'
    assert not EXCLUDE.intersection(feature_columns(dataset))
    assert 'balance_after' not in feature_columns(dataset)
    assert 'rule_risk_score' not in feature_columns(dataset)
