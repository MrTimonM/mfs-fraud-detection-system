"""Legitimate-only behavioral Isolation Forest with explicit cold-start fallback."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .leakage_audit import ANOMALY


class BehavioralAnomaly:
    def fit(self, frame, cfg):
        self.columns = ANOMALY
        self.prior = float(frame.fraud_label.mean())
        legitimate = frame[(frame.fraud_label == 0) & (frame.behavioral_model_active == 1)]
        self.preprocessor = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        x = self.preprocessor.fit_transform(legitimate[self.columns])
        self.model = IsolationForest(n_estimators=80, max_samples=512, contamination='auto',
            random_state=cfg['seed'], n_jobs=cfg['threads']).fit(x)
        self.reference = np.sort(-self.model.score_samples(x))
        return self

    def predict(self, frame):
        scores = -self.model.score_samples(self.preprocessor.transform(frame[self.columns]))
        ranks = np.searchsorted(self.reference, scores) / len(self.reference)
        return np.where(frame.behavioral_model_active.to_numpy() == 1, ranks, self.prior)
