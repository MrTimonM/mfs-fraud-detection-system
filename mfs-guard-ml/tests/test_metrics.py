import numpy as np
from src.evaluate import metrics, paired_bootstrap
from src.config import config
from src.threshold_optimizer import optimize
from src.calibration import calibrate


def test_confusion_and_business_cost():
    y, p = np.array([0, 0, 1, 1]), np.array([.1, .8, .2, .9])
    m = metrics(y, p, .5, config())
    assert m['true_positive'] == m['false_positive'] == m['true_negative'] == m['false_negative'] == 1
    assert m['expected_business_cost'] == 1020
    assert m['precision'] == m['recall'] == .5


def test_threshold_minimizes_validation_cost():
    y, p = np.array([0, 0, 1, 1]), np.array([.01, .03, .1, .8])
    t, curve = optimize(y, p, config())
    assert metrics(y, p, t, config())['expected_business_cost'] == curve.expected_business_cost.min()


def test_calibration_bounds():
    p = np.linspace(.01, .99, 200)
    y = (np.arange(200) % 9 == 0).astype(int)
    mapping, report = calibrate(p[:100], y[:100], p[100:], y[100:])
    assert sum(r['selected'] for r in report) == 1
    output = mapping.predict(np.array([0, .5, 1]))
    assert np.all((output >= 0) & (output <= 1))


def test_fast_bootstrap_matches_explicit_cluster_resampling():
    from sklearn.metrics import average_precision_score
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    a = np.array([.2, .4, .3, .4, .1, .8, .7, .9])
    b = np.array([.4, .4, .3, .4, .2, .7, .5, .8])
    groups = np.repeat(np.arange(4), 2)
    result = paired_bootstrap(y, a, b, groups, repeats=30)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(30):
        ix = np.concatenate([np.flatnonzero(groups == j) for j in rng.integers(4, size=4)])
        differences.append(average_precision_score(y[ix], a[ix])-average_precision_score(y[ix], b[ix]))
    assert np.isclose(result['delta_ci_low'], np.quantile(differences, .025))
    assert np.isclose(result['delta_ci_high'], np.quantile(differences, .975))
