"""Serving-time feature computation that REUSES src.feature_engineering.engineer.

Training-serving consistency: no feature is reimplemented here. For each request we
build a chronologically sorted frame of (past raw rows + the current raw row), run
the exact offline `engineer()` and `apply_rules()` code on it, and keep the last row.

Cost note: this is O(len(history)) per request because engineer() replays the whole
ledger. The ledger is capped at `max_rows` (oldest rows dropped) to bound latency;
dropping very old rows only affects features with >30d windows / lifetime graph
degree counts. A production system would replace this with an incremental state
store exposing the same History/HistoricalGraph objects.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, config
from src.feature_engineering import engineer
from src.rule_engine import apply_rules

PROFILE_COLUMNS = ['typical_transaction_amount', 'home_latitude', 'home_longitude',
                   'preferred_channel', 'preferred_transaction_type', 'customer_segment',
                   'account_age_days', 'normal_active_hour_start', 'normal_active_hour_end',
                   'normal_location_radius_km', 'wallet_balance_tendency']
TELEMETRY_COLUMNS = ['failed_pin_attempts', 'otp_resend_count', 'balance_inquiry_count_5m',
                     'pin_reset_recently', 'password_reset_recently', 'biometric_failed_recently',
                     'rooted_device', 'emulator_detected', 'vpn_active', 'screen_share_detected',
                     'sim_changed_recently', 'device_os_changed', 'device_fingerprint_changed',
                     'blacklisted_device', 'blacklisted_agent']
INT_COLUMNS = ['account_age_days', 'normal_active_hour_start', 'normal_active_hour_end']


def cohort_defaults(path: Path | None = None) -> dict:
    """Medians (numeric) / modes (categorical) of the training-scale profile table."""
    path = path or ROOT / 'data/metadata/profiles_200k.parquet'
    p = pd.read_parquet(path)
    out = {}
    for c in PROFILE_COLUMNS:
        if p[c].dtype.kind in 'if':
            v = float(p[c].median())
            out[c] = int(round(v)) if c in INT_COLUMNS else v
        else:
            out[c] = str(p[c].mode().iloc[0])
    return out


class IdMap:
    """Stable string -> int ids (training used integer ids)."""

    def __init__(self):
        self._ids: dict[str, int] = {}

    def get(self, key) -> int:
        key = str(key)
        if key not in self._ids:
            self._ids[key] = len(self._ids)
        return self._ids[key]


class HistoryStore:
    """In-memory ledger of past raw rows (completed transactions)."""

    def __init__(self, max_rows: int = 20000):
        self.rows: list[dict] = []
        self.max_rows = max_rows
        self.lock = threading.Lock()

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.rows)

    def append(self, row: dict) -> None:
        with self.lock:
            self.rows.append(row)
            if len(self.rows) > self.max_rows:
                del self.rows[: len(self.rows) - self.max_rows]

    def clear(self) -> None:
        with self.lock:
            self.rows.clear()

    def __len__(self):
        return len(self.rows)


class FeatureEngine:
    def __init__(self, history: HistoryStore | None = None, cfg: dict | None = None):
        self.cfg = cfg or config()
        self.history = history or HistoryStore()
        self.accounts, self.devices, self.agents, self.merchants = IdMap(), IdMap(), IdMap(), IdMap()
        self.defaults = cohort_defaults()
        self.profiles: dict[str, dict] = {}
        self._session = IdMap()
        self.rules_cfg = self.cfg  # overridden with thresholds.json rules by the app

    # ---- raw row construction -------------------------------------------------
    def build_raw(self, tx: dict) -> tuple[dict, str, list[str]]:
        """Map an API transaction dict to the raw schema engineer() expects."""
        missing: list[str] = []
        user = str(tx['user_id'])
        given = tx.get('customer_profile') or None
        if given:
            profile = {**self.defaults, **{k: v for k, v in given.items() if v is not None}}
            self.profiles[user] = profile
            source = 'PROVIDED'
        elif user in self.profiles:
            profile, source = self.profiles[user], 'STORED'
        else:
            profile, source = dict(self.defaults), 'COHORT_DEFAULT'
        ts = pd.Timestamp(tx['timestamp'])
        ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
        amount = float(tx['amount'])
        row = {
            'transaction_id': str(tx['transaction_id']),
            'user_id': self.accounts.get(user),
            'receiver_id': self.accounts.get(tx['receiver_id']),
            'device_id': self.devices.get(tx['device_id']),
            'session_id': self._session.get(tx.get('session_id') or tx['transaction_id']),
            'timestamp': ts,
            'transaction_type': tx['transaction_type'], 'channel': tx['channel'],
            'currency': tx.get('currency') or 'BDT', 'amount': amount,
            'agent_id': self.agents.get(tx['agent_id']) if tx.get('agent_id') not in (None, '') else -1,
            'merchant_id': self.merchants.get(tx['merchant_id']) if tx.get('merchant_id') not in (None, '') else -1,
            # Placeholders: engineer() drops these before computing anything.
            'fraud_label': 0, 'fraud_type': 'LEGITIMATE',
        }
        row.update({c: profile[c] for c in PROFILE_COLUMNS})
        for c in TELEMETRY_COLUMNS:
            v = tx.get(c)
            if v is None:
                missing.append(c)
                v = 0
            row[c] = int(v)
        for c, fallback in (('latitude', profile['home_latitude']), ('longitude', profile['home_longitude']), ('fee', 0.0)):
            v = tx.get(c)
            if v is None:
                missing.append(c)
                v = fallback
            row[c] = float(v)
        before = tx.get('balance_before')
        if before is None:
            missing.append('balance_before')
            before = float(profile['typical_transaction_amount']) * 70  # generator's opening balance
        incoming = row['transaction_type'] in ('CASH_IN', 'REMITTANCE')
        after = tx.get('balance_after')
        if after is None:
            after = before + amount if incoming else before - amount - row['fee']
        row['balance_before'], row['balance_after'] = float(before), float(after)
        dep = tx.get('depletion_ratio')
        row['depletion_ratio'] = float(dep) if dep is not None else (amount + row['fee']) / max(before, 1e-6)
        row['balance_change_ratio'] = (after - before) / max(before, 1e-6)
        return row, source, missing

    # ---- feature computation ---------------------------------------------------
    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        d = pd.DataFrame(rows)
        d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True).astype('datetime64[ns, UTC]')
        d = d.sort_values('timestamp', kind='stable').reset_index(drop=True)
        # Same quantization as the generator applies before engineer().
        for c in d.select_dtypes('float64'):
            d[c] = d[c].astype('float32')
        return d

    def compute(self, raw: dict) -> tuple[pd.DataFrame, str]:
        """Return (single-row engineered+ruled frame, history_mode)."""
        mode = 'NORMAL'
        try:
            past = [r for r in self.history.snapshot() if r['timestamp'] <= raw['timestamp']]
        except Exception:
            past, mode = [], 'DEGRADED'
        d = self._frame(past + [raw])
        idx = int(np.flatnonzero(d['transaction_id'].to_numpy() == raw['transaction_id'])[-1])
        d = engineer(d, self.cfg)
        out = d.iloc[[idx]].reset_index(drop=True)
        apply_rules(out, self.rules_cfg)
        return out, mode

    def commit(self, raw: dict) -> None:
        self.history.append(raw)

    def ingest(self, txs: list[dict]) -> int:
        for tx in txs:
            raw, _, _ = self.build_raw(tx)
            self.commit(raw)
        return len(txs)
