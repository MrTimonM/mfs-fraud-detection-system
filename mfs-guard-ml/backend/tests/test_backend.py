import json
import warnings
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings('ignore')

from backend import demo_scenarios as demo  # noqa: E402
from backend.app.main import Analyzer, create_app  # noqa: E402
from backend.app.services.decision_engine import DecisionEngine  # noqa: E402
from backend.app.services.ml_inference_service import MODEL_DIR, MLInferenceService  # noqa: E402

THRESHOLDS = json.loads((MODEL_DIR / 'thresholds.json').read_text())
RESPONSE_KEYS = {'transaction_id', 'fraud_probability', 'risk_level', 'decision', 'model', 'model_version',
                 'rule_risk_score', 'triggered_rules', 'anomaly_score', 'device_risk', 'recipient_risk',
                 'graph_risk', 'top_risk_factors', 'latency_ms', 'system_mode', 'profile_source',
                 'missing_optional_fields', 'requires_human_review'}


@pytest.fixture(scope='session')
def ml():
    return MLInferenceService()


@pytest.fixture
def client(ml, tmp_path):
    return TestClient(create_app(Analyzer(ml=ml, log_dir=tmp_path)))


def web_tx(**kw):
    tx = {'transaction_id': 'T1', 'user_id': 'U1', 'receiver_id': 'R1', 'device_id': 'D1',
          'timestamp': '2026-03-01T10:00:00+06:00', 'transaction_type': 'SEND_MONEY', 'channel': 'APP',
          'amount': 1500, 'balance_before': 50000}
    tx.update(kw)
    return tx


def test_model_loads(ml):
    assert ml.loaded and ml.error is None
    assert ml.metadata['model_name'] == 'LightGBM'


def test_feature_order(ml):
    assert list(ml.preprocessor.feature_names_in_) == ml.features
    assert len(ml.features) == ml.metadata['feature_count']


def test_normal_transaction(client):
    hist, tx = demo.normal()
    client.post('/api/v1/history/ingest', json={'transactions': hist}).raise_for_status()
    r = client.post('/api/v1/transactions/analyze', json=tx).json()
    assert r['decision'] == 'APPROVE' and r['risk_level'] == 'LOW'
    assert r['requires_human_review'] is False and r['system_mode'] == 'NORMAL'


def test_high_risk_transaction(client):
    hist, tx = demo.account_takeover()
    client.post('/api/v1/history/ingest', json={'transactions': hist})
    r = client.post('/api/v1/transactions/analyze', json=tx).json()
    assert r['decision'] in ('TEMPORARY_HOLD', 'REJECT_AND_FREEZE')
    assert r['requires_human_review'] is True
    assert r['top_risk_factors'] and all(isinstance(f, str) for f in r['top_risk_factors'])


def test_missing_optional_features_reported(client):
    r = client.post('/api/v1/transactions/analyze',
                    json=web_tx(latitude=None, receiver_id=None, msisdn='x', imei='y')).json()
    for f in ('failed_pin_attempts', 'latitude', 'longitude', 'fee', 'receiver_id', 'vpn_active'):
        assert f in r['missing_optional_fields']
    assert r['fraud_probability'] is not None


def test_cold_start_customer(client):
    r = client.post('/api/v1/transactions/analyze', json=web_tx(user_id='BRAND_NEW')).json()
    assert r['profile_source'] == 'COHORT_DEFAULT'
    assert r['anomaly_score'] is None  # behavioral layer abstains with no history
    assert 0 <= r['fraud_probability'] <= 1


def test_web_payload_contract(client):
    r = client.post('/api/v1/transactions/analyze', json=web_tx(
        transaction_type='PAYMENT', agent_id='', msisdn='017', imei='35', sim_id='s', ip_address='1.1.1.1',
        pin_reset_recently=True, device_is_new=False, channel_changed_recently=False,
        previous_latitude=1, previous_timestamp='2026-01-01T00:00:00Z', unknown_extra='ignored'))
    assert r.status_code == 200, r.text
    assert set(RESPONSE_KEYS) <= set(r.json())


def test_model_failure_degraded_rules_decide(client, monkeypatch):
    a = client.app.state.analyzer

    def boom(frame):
        raise RuntimeError('model down')
    monkeypatch.setattr(a.ml, 'predict', boom)
    r = client.post('/api/v1/transactions/analyze', json=web_tx(
        transaction_id='T9', blacklisted_device=1, failed_pin_attempts=5, otp_resend_count=4,
        balance_inquiry_count_5m=4, emulator_detected=1, amount=48000)).json()
    assert r['system_mode'] == 'DEGRADED' and r['fraud_probability'] is None
    assert r['decision'] == 'REJECT_AND_FREEZE' and r['rule_risk_score'] >= 70


def test_model_failure_never_defaults_approve_when_rules_flag(client, monkeypatch):
    a = client.app.state.analyzer
    monkeypatch.setattr(a.ml, 'predict', lambda f: (_ for _ in ()).throw(RuntimeError('x')))
    r = client.post('/api/v1/transactions/analyze', json=web_tx(
        transaction_id='T10', failed_pin_attempts=4, otp_resend_count=4, emulator_detected=1,
        balance_inquiry_count_5m=4, amount=45000)).json()
    assert r['system_mode'] == 'DEGRADED' and r['decision'] != 'APPROVE'


def test_history_store_failure_degraded(client, monkeypatch):
    a = client.app.state.analyzer

    def broken():
        raise ConnectionError('ledger unavailable')
    monkeypatch.setattr(a.features.history, 'snapshot', broken)
    r = client.post('/api/v1/transactions/analyze', json=web_tx(transaction_id='T11'))
    assert r.status_code == 200 and r.json()['system_mode'] == 'DEGRADED'


@pytest.mark.parametrize('p,expected', [
    (0.0, 'APPROVE'), ('monitor', 'APPROVE_AND_MONITOR'), ('step_up', 'STEP_UP_AUTH'), ('hold', 'TEMPORARY_HOLD')])
def test_threshold_action_mapping(p, expected):
    de = DecisionEngine(THRESHOLDS)
    prob = THRESHOLDS['ml_probability'][p] if isinstance(p, str) else p
    assert de.decide(prob, 0, False)['decision'] == expected
    if isinstance(p, str):
        assert de.decide(prob - 1e-9, 0, False)['decision'] != expected


def test_rule_bands_and_reject_combination():
    de = DecisionEngine(THRESHOLDS)
    rs = THRESHOLDS['rule_score']
    assert de.decide(0.0, rs['step_up'], False)['decision'] == 'STEP_UP_AUTH'
    assert de.decide(0.0, rs['hold'], False)['decision'] == 'TEMPORARY_HOLD'
    hold = THRESHOLDS['ml_probability']['hold']
    out = de.decide(hold, THRESHOLDS['reject_requires']['rule_score_at_least'], False)
    assert out['decision'] == 'REJECT_AND_FREEZE' and out['requires_human_review']
    assert de.decide(0.0, 0, True)['decision'] == 'REJECT_AND_FREEZE'
    assert de.decide(None, 0, False)['system_mode'] == 'DEGRADED'


def test_api_schema_and_endpoints(client, tmp_path):
    r = client.post('/api/v1/transactions/analyze', json=web_tx(transaction_id='S1'))
    body = r.json()
    assert set(RESPONSE_KEYS) <= set(body)
    assert isinstance(body['triggered_rules'], list) and all(isinstance(x, str) for x in body['triggered_rules'])
    assert client.get('/health').json()['system_mode'] == 'NORMAL'
    assert client.get('/api/v1/model').json()['metadata']['version'] == body['model_version']
    assert client.get('/api/v1/dashboard/summary').json()['total'] >= 1
    lab = client.post('/api/v1/alerts/S1/label', json={'label': 'FALSE_POSITIVE'})
    assert lab.status_code == 200
    assert (tmp_path / 'labels.jsonl').exists() and (tmp_path / 'audit.jsonl').exists()
    audit = json.loads((tmp_path / 'audit.jsonl').read_text().splitlines()[-1])
    assert {'transaction_id', 'model_version', 'final_action', 'latency_ms', 'system_mode'} <= set(audit)
    assert client.post('/api/v1/alerts/S1/label', json={'label': 'MAYBE'}).status_code == 422


def test_history_affects_features(client):
    # Same transaction scored with vs. without a prior relationship: first_time_recipient must flip.
    a = client.app.state.analyzer
    t0 = datetime(2026, 3, 1, 10, tzinfo=timezone.utc)
    prior = [web_tx(transaction_id=f'P{i}', timestamp=(t0 + timedelta(hours=i)).isoformat()) for i in range(3)]
    client.post('/api/v1/history/ingest', json={'transactions': prior})
    raw, _, _ = a.features.build_raw(web_tx(transaction_id='Q', timestamp=(t0 + timedelta(hours=5)).isoformat()))
    frame, _ = a.features.compute(raw)
    assert frame['first_time_recipient'].iloc[0] == 0 and frame['behavioral_history_count'].iloc[0] == 3


@pytest.mark.parametrize('name', list(demo.SCENARIOS))
def test_demo_scenarios(client, name):
    hist, tx = demo.SCENARIOS[name]()
    client.post('/api/v1/history/ingest', json={'transactions': hist}).raise_for_status()
    r = client.post('/api/v1/transactions/analyze', json=tx).json()
    assert r['decision'] in demo.EXPECTED[name], r


def test_training_serving_consistency(ml):
    """Service path (HistoryStore + compute) reproduces stored offline features bit-for-bit."""
    import numpy as np
    import pandas as pd
    from src.config import ROOT
    from backend.app.services.feature_engine import FeatureEngine
    d = pd.read_parquet(ROOT / 'data/raw/mfs_100k.parquet').head(400)
    raw_cols = list(d.columns[:list(d.columns).index('balance_change_ratio') + 1])
    fe = FeatureEngine()
    rows = d[raw_cols].to_dict('records')
    for r in rows[:-1]:
        fe.commit(r)
    out, mode = fe.compute(rows[-1])
    assert mode == 'NORMAL'
    for c in ml.features:
        a, b = out[c].iloc[0], d[c].iloc[-1]
        assert a == b or (isinstance(a, float) and np.isclose(a, b, rtol=0, atol=0)), c
