import joblib
import numpy as np
from lightgbm import LGBMClassifier
from src.config import config
from src.leakage_audit import feature_columns
from src.preprocessing import preprocessor
from src.risk_fusion import RiskBundle
from src.train_anomaly import BehavioralAnomaly


def test_saved_bundle_roundtrip_and_unseen_categories(dataset, tmp_path):
    train, test = dataset.iloc[:4000], dataset.iloc[-20:].copy()
    columns = feature_columns(dataset)
    prep = preprocessor(dataset, columns).fit(train[columns])
    model = LGBMClassifier(n_estimators=8, n_jobs=2, verbosity=-1).fit(prep.transform(train[columns]), train.fraud_label)
    bundle = RiskBundle('supervised', config(), model=model, preprocessor=prep, columns=columns)
    test['channel'] = 'UNSEEN_CHANNEL'
    expected = bundle.predict(test)
    path = tmp_path/'bundle.joblib'
    joblib.dump(bundle, path)
    actual = joblib.load(path).predict(test)
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()


def test_anomaly_explicit_cold_start(dataset):
    anomaly = BehavioralAnomaly().fit(dataset.iloc[:4000], config())
    cold = dataset.iloc[:10].copy()
    cold['behavioral_model_active'] = 0
    np.testing.assert_allclose(anomaly.predict(cold), anomaly.prior)
