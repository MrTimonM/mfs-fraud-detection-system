"""MFS Guard real-time scoring API. Run from mfs-guard-ml/: uvicorn backend.app.main:app"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

ML_ROOT = Path(__file__).resolve().parents[2]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from backend.app.services.decision_engine import DecisionEngine  # noqa: E402
from backend.app.services.feature_engine import FeatureEngine  # noqa: E402
from backend.app.services.ml_inference_service import MLInferenceService  # noqa: E402

LOG_DIR = ML_ROOT / 'backend' / 'logs'

TxType = Literal['SEND_MONEY', 'CASH_OUT', 'CASH_IN', 'MERCHANT_PAYMENT', 'MOBILE_RECHARGE',
                 'BILL_PAYMENT', 'BANK_TRANSFER', 'REMITTANCE']
TxIn = Literal['SEND_MONEY', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'MERCHANT_PAYMENT', 'MOBILE_RECHARGE',
               'BILL_PAYMENT', 'BANK_TRANSFER', 'REMITTANCE']
Channel = Literal['APP', 'USSD', 'AGENT', 'WEB', 'API']


class CustomerProfile(BaseModel):
    typical_transaction_amount: Optional[float] = None
    home_latitude: Optional[float] = None
    home_longitude: Optional[float] = None
    preferred_channel: Optional[Channel] = None
    preferred_transaction_type: Optional[TxType] = None
    customer_segment: Optional[str] = None
    account_age_days: Optional[int] = None
    normal_active_hour_start: Optional[int] = None
    normal_active_hour_end: Optional[int] = None
    normal_location_radius_km: Optional[float] = None
    wallet_balance_tendency: Optional[float] = None


Flag = Optional[Union[bool, int]]


class Transaction(BaseModel):
    """Accepts the Next.js app payload; unknown fields are ignored."""
    model_config = ConfigDict(extra='ignore')
    transaction_id: str
    user_id: str
    amount: float = Field(gt=0)
    balance_before: float
    timestamp: datetime
    transaction_type: TxIn
    receiver_id: Optional[str] = None
    device_id: Optional[str] = None
    channel: Optional[Channel] = None
    currency: str = 'BDT'
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    merchant_id: Optional[str] = None
    imei: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    fee: Optional[float] = None
    balance_after: Optional[float] = None
    depletion_ratio: Optional[float] = None
    # Optional device / session telemetry (defaults to 0 and is reported as missing).
    failed_pin_attempts: Optional[int] = None
    otp_resend_count: Optional[int] = None
    balance_inquiry_count_5m: Optional[int] = None
    pin_reset_recently: Flag = None
    password_reset_recently: Flag = None
    biometric_failed_recently: Flag = None
    rooted_device: Flag = None
    emulator_detected: Flag = None
    vpn_active: Flag = None
    screen_share_detected: Flag = None
    sim_changed_recently: Flag = None
    device_os_changed: Flag = None
    device_fingerprint_changed: Flag = None
    blacklisted_device: Flag = None
    blacklisted_agent: Flag = None
    # Accepted but not used as features: engineer() recomputes these from history.
    device_is_new: Flag = None
    channel_changed_recently: Flag = None
    customer_profile: Optional[CustomerProfile] = None

    def to_tx(self) -> dict:
        tx = self.model_dump(mode='python')
        if tx['transaction_type'] == 'PAYMENT':
            tx['transaction_type'] = 'MERCHANT_PAYMENT'
        missing = []
        if not tx.get('channel'):
            tx['channel'] = 'AGENT' if tx['transaction_type'] == 'CASH_OUT' else 'APP'
            missing.append('channel')
        if not tx.get('device_id'):
            tx['device_id'] = tx.get('imei') or f"UNKNOWN_DEVICE:{tx['user_id']}"
            missing.append('device_id')
        if not tx.get('receiver_id'):
            tx['receiver_id'] = tx.get('agent_id') or tx.get('merchant_id') or 'UNKNOWN_RECEIVER'
            missing.append('receiver_id')
        tx['_extra_missing'] = missing
        return tx


class IngestRequest(BaseModel):
    transactions: list[Transaction]


class LabelRequest(BaseModel):
    label: Literal['CONFIRMED_FRAUD', 'FALSE_POSITIVE']
    analyst: Optional[str] = None
    note: Optional[str] = None


class Analyzer:
    def __init__(self, ml: MLInferenceService | None = None, log_dir: Path = LOG_DIR):
        self.ml = ml or MLInferenceService()
        self.decider = DecisionEngine(self.ml.thresholds)
        self.features = FeatureEngine()
        self.features.rules_cfg = {'rules': self.ml.thresholds['rules']}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.counts: Counter = Counter()
        self.modes: Counter = Counter()
        self.lock = threading.Lock()  # history must see transactions in commit order

    def _audit(self, path: str, record: dict) -> None:
        try:
            with open(self.log_dir / path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception:
            pass

    def analyze(self, tx: dict) -> dict:
        start = time.perf_counter()
        with self.lock:
            raw, profile_source, missing = self.features.build_raw(tx)
            missing = list(tx.get('_extra_missing', [])) + missing
            frame, history_mode = self.features.compute(raw)
            rule_score = float(frame['rule_risk_score'].iloc[0])
            triggered = [{'rule': r['feature'], 'threshold': r['threshold'], 'weight': r['weight'],
                          'value': float(frame[r['feature']].iloc[0])}
                         for r in self.ml.thresholds['rules'] if frame[r['feature']].iloc[0] >= r['threshold']]
            probability, factors, ml_error = None, [], None
            try:
                probability = self.ml.predict(frame)
                factors = self.ml.top_factors(frame)
            except Exception as exc:
                probability, ml_error = None, f'{type(exc).__name__}: {exc}'
            blacklisted = bool(raw['blacklisted_device'] or raw['blacklisted_agent'])
            d = self.decider.decide(probability, rule_score, blacklisted)
            mode = 'DEGRADED' if 'DEGRADED' in (d['system_mode'], history_mode) else 'NORMAL'
            anomaly = self.ml.anomaly_score(frame) if probability is not None else None
            # Scored before authorization; now commit to history.
            try:
                self.features.commit(raw)
            except Exception:
                mode = 'DEGRADED'
            if probability is None:
                level = {'REJECT_AND_FREEZE': 'CRITICAL', 'TEMPORARY_HOLD': 'HIGH',
                         'STEP_UP_AUTH': 'MEDIUM'}.get(d['decision'], 'UNKNOWN')
            else:
                level = self.ml.risk_level(probability)
            self.counts[d['decision']] += 1
            self.modes[mode] += 1
        latency = (time.perf_counter() - start) * 1000
        f0 = frame.iloc[0]
        response = {
            'transaction_id': tx['transaction_id'],
            'fraud_probability': None if probability is None else round(probability, 6),
            'risk_level': level, 'decision': d['decision'],
            'model': self.ml.metadata.get('model_name'), 'model_version': self.ml.version,
            'rule_risk_score': rule_score, 'triggered_rules': [r['rule'] for r in triggered],
            'anomaly_score': anomaly,
            'device_risk': float(f0['device_risk_score']), 'recipient_risk': float(f0['recipient_risk_score']),
            'graph_risk': float(f0['mule_network_score']),
            'top_risk_factors': [f['description'] for f in factors], 'latency_ms': round(latency, 2), 'system_mode': mode,
            'profile_source': profile_source, 'missing_optional_fields': missing,
            'requires_human_review': d['requires_human_review'],
        }
        response['details'] = {'ml_action': d['ml_action'], 'rule_action': d['rule_action'],
                               'decision_reasons': d['decision_reasons'], 'triggered_rules': triggered,
                               'top_risk_factors': factors, 'ml_error': ml_error}
        self._audit('audit.jsonl', {
            'transaction_id': tx['transaction_id'], 'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_version': self.ml.version, 'fraud_probability': response['fraud_probability'],
            'rule_risk_score': rule_score, 'anomaly_score': anomaly,
            'triggered_rules': [r['rule'] for r in triggered], 'final_action': d['decision'],
            'latency_ms': response['latency_ms'], 'system_mode': mode})
        return response

    def ingest(self, txs: list[dict]) -> int:
        with self.lock:
            return self.features.ingest(sorted(txs, key=lambda t: t['timestamp']))

    def label(self, transaction_id: str, body: dict) -> dict:
        rec = {'transaction_id': transaction_id, **body, 'labelled_at': datetime.now(timezone.utc).isoformat()}
        self._audit('labels.jsonl', rec)
        return rec

    def health(self) -> str:
        return 'NORMAL' if self.ml.loaded else 'DEGRADED'


def create_app(analyzer: Analyzer | None = None) -> FastAPI:
    app = FastAPI(title='MFS Guard Scoring API', version='1.0')
    app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
                       allow_methods=['*'], allow_headers=['*'])
    app.state.analyzer = analyzer or Analyzer()

    def svc() -> Analyzer:
        return app.state.analyzer

    @app.post('/api/v1/transactions/analyze')
    def analyze(tx: Transaction):
        return svc().analyze(tx.to_tx())

    @app.post('/api/v1/history/ingest')
    def ingest(req: IngestRequest):
        n = svc().ingest([t.to_tx() for t in req.transactions])
        return {'ingested': n, 'history_size': len(svc().features.history)}

    @app.post('/api/v1/alerts/{transaction_id}/label')
    def label(transaction_id: str, req: LabelRequest):
        return svc().label(transaction_id, req.model_dump())

    @app.get('/health')
    def health():
        a = svc()
        return {'status': 'ok', 'system_mode': a.health(), 'model_loaded': a.ml.loaded,
                'model_error': a.ml.error, 'history_size': len(a.features.history)}

    @app.get('/api/v1/model')
    def model():
        a = svc()
        return {'metadata': a.ml.metadata, 'thresholds': {k: a.ml.thresholds[k] for k in
                ('ml_probability', 'rule_score', 'reject_requires', 'risk_level')},
                'feature_count': len(a.ml.features), 'loaded': a.ml.loaded}

    @app.get('/api/v1/dashboard/summary')
    def summary():
        a = svc()
        return {'total': sum(a.counts.values()), 'decisions': dict(a.counts),
                'system_modes': dict(a.modes), 'model_version': a.ml.version}

    return app


app = create_app()
