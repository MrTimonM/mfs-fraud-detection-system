"""Serializable inference bundles including all preprocessing and calibration."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .calibration import ProbabilityMap
from .rule_engine import apply_rules

FUSION_COLUMNS = ['supervised', 'rules', 'anomaly', 'device_risk_score',
                  'recipient_risk_score', 'mule_network_score', 'sequence_risk_score', 'behavioral_model_active']


def meta_features(d, supervised, anomaly):
    return np.column_stack([supervised, d.rule_risk_score.to_numpy()/100, anomaly,
        d.device_risk_score.to_numpy()/100, d.recipient_risk_score.to_numpy()/100,
        d.mule_network_score.to_numpy()/100, d.sequence_risk_score.to_numpy()/100,
        d.behavioral_model_active.to_numpy()])


def fit_fusion(x, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=400)).fit(x, y)


class RiskBundle:
    """Input is a pre-decision frame with historical features, not a raw HTTP event."""
    def __init__(self, kind, cfg, model=None, preprocessor=None, columns=None,
                 supervised=None, anomaly=None, meta_columns=None):
        self.kind, self.cfg, self.model = kind, cfg, model
        self.preprocessor, self.columns = preprocessor, columns
        self.supervised, self.anomaly, self.meta_columns = supervised, anomaly, meta_columns
        self.calibration = ProbabilityMap()
        self.threshold = .5

    def raw(self, frame):
        if self.kind == 'supervised':
            return self.model.predict_proba(self.preprocessor.transform(frame[self.columns]).astype('float32'))[:, 1]
        if self.kind == 'rules':
            return apply_rules(frame.copy(), self.cfg)
        if self.kind == 'anomaly':
            return self.anomaly.predict(frame)
        sp = self.supervised.predict(frame)
        ap = self.anomaly.predict(frame) if 2 in self.meta_columns else np.zeros(len(frame))
        matrix = meta_features(frame, sp, ap)
        return self.model.predict_proba(matrix[:, self.meta_columns])[:, 1]

    def predict(self, frame):
        return self.calibration.predict(self.raw(frame))
