"""Loads the exported LightGBM bundle once and serves calibrated probabilities + SHAP factors."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[2] / 'models' / 'final'


class ModelUnavailable(RuntimeError):
    pass


def _fmt(v):
    return f'{v:,.0f}' if abs(v) >= 100 else f'{v:.2f}'


def describe(name: str, v: float, row: pd.Series) -> str:
    """Human phrase built only from the actual feature value."""
    v = float(v) if isinstance(v, (int, float, np.number)) else v
    phrases = {
        'device_is_new': lambda: 'new device for this customer' if v else None,
        'device_first_seen': lambda: 'device never seen before' if v else None,
        'first_time_recipient': lambda: 'first-time recipient' if v else None,
        'amount_vs_user_mean': lambda: f'amount {v:.1f}x above user baseline' if v > 1 else f'amount {v:.1f}x of user baseline',
        'amount_vs_user_median': lambda: f'amount {v:.1f}x user median',
        'amount_zscore_user': lambda: f'amount z-score {v:.1f} vs user history',
        'failed_pin_attempts': lambda: f'{int(v)} failed PIN attempts',
        'otp_resend_count': lambda: f'{int(v)} OTP resends',
        'pin_reset_recently': lambda: 'PIN reset recently' if v else None,
        'password_reset_recently': lambda: 'password reset recently' if v else None,
        'depletion_ratio': lambda: f'transaction drains {v:.0%} of balance',
        'unusual_hour_score': lambda: 'outside customer\'s normal active hours' if v else None,
        'is_night_transaction': lambda: 'night-time transaction' if v else None,
        'receiver_unique_senders_24h': lambda: f'recipient received from {int(v)} distinct senders in 24h',
        'receiver_transaction_count_24h': lambda: f'recipient had {int(v)} incoming transactions in 24h',
        'mule_network_score': lambda: f'mule-network score {v:.0f}/100',
        'recipient_risk_score': lambda: f'recipient risk score {v:.0f}/100',
        'device_risk_score': lambda: f'device risk score {v:.0f}/100',
        'rapid_fund_turnaround': lambda: 'funds forwarded within 10 min of receipt' if v else None,
        'turnaround_latency_seconds': lambda: f'funds moved {v:.0f}s after last inflow' if v < 1e8 else None,
        'tx_count_5m': lambda: f'{int(v)} transactions in last 5 min',
        'tx_count_1h': lambda: f'{int(v)} transactions in last hour',
        'distance_from_home_km': lambda: f'{v:.0f} km from home location',
        'behavioral_history_count': lambda: f'only {int(v)} prior transactions (thin history)' if v < 5 else f'{int(v)} prior transactions',
    }
    if name in phrases:
        text = phrases[name]()
        if text:
            return text
    if isinstance(v, str):
        return f'{name} = {v}'
    return f'{name} = {_fmt(v)}'


class MLInferenceService:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False
        self.error: str | None = None
        self.thresholds = json.loads((self.model_dir / 'thresholds.json').read_text())
        self.metadata = json.loads((self.model_dir / 'model_metadata.json').read_text())
        self.features = json.loads((self.model_dir / 'feature_list.json').read_text())
        try:
            self.model = joblib.load(self.model_dir / 'model.joblib')
            self.preprocessor = joblib.load(self.model_dir / 'preprocessor.joblib')
            self.calibrator = joblib.load(self.model_dir / 'calibrator.joblib')
            try:
                self.anomaly = joblib.load(self.model_dir / 'anomaly.joblib')
            except Exception:
                self.anomaly = None
            expected = list(self.preprocessor.feature_names_in_)
            if expected != self.features:
                raise ModelUnavailable('feature_list.json order does not match preprocessor input order')
            self.out_names = list(self.preprocessor.get_feature_names_out())
            self.out_to_raw = [self._raw_name(n) for n in self.out_names]
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            self.loaded = True
        except Exception as exc:  # flagged; decision engine falls back to rules-only
            self.error = f'{type(exc).__name__}: {exc}'

    def _raw_name(self, out_name: str) -> str:
        if out_name in self.features:
            return out_name
        for c in ('transaction_type', 'channel', 'currency', 'customer_segment'):
            if out_name.startswith(c + '_'):
                return c
        return out_name

    @property
    def version(self) -> str:
        return self.metadata.get('version', 'unknown')

    def risk_level(self, p: float | None) -> str:
        if p is None:
            return 'UNKNOWN'
        level = 'LOW'
        for name, cut in sorted(self.thresholds['risk_level'].items(), key=lambda kv: kv[1]):
            if p >= cut:
                level = name
        return level

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ModelUnavailable(f'missing features: {missing[:5]}')
        return self.preprocessor.transform(frame[self.features]).astype('float32')

    def predict(self, frame: pd.DataFrame) -> float:
        if not self.loaded:
            raise ModelUnavailable(self.error or 'model not loaded')
        x = self._matrix(frame)
        raw = self.model.predict_proba(x)[:, 1]
        return float(self.calibrator.predict(raw)[0])

    def anomaly_score(self, frame: pd.DataFrame) -> float | None:
        """Rank in [0,1]; None (abstain) on cold start or when the layer is unavailable."""
        if self.anomaly is None or int(frame['behavioral_model_active'].iloc[0]) != 1:
            return None
        try:
            return float(self.anomaly.predict(frame)[0])
        except Exception:
            return None

    def top_factors(self, frame: pd.DataFrame, k: int = 5) -> list[dict]:
        if not self.loaded:
            return []
        x = self._matrix(frame)
        sv = self.explainer.shap_values(x)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
        agg: dict[str, float] = {}
        for raw, val in zip(self.out_to_raw, sv):
            agg[raw] = agg.get(raw, 0.0) + float(val)
        row = frame.iloc[0]
        ranked = sorted((kv for kv in agg.items() if kv[1] > 0), key=lambda kv: -kv[1])[:k]
        out = []
        for name, contrib in ranked:
            value = row[name]
            value = value.item() if isinstance(value, np.generic) else value
            out.append({'feature': name, 'value': value, 'shap_contribution': round(contrib, 4),
                        'description': describe(name, value, row)})
        return out
