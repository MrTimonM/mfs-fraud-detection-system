import pandas as pd
from src.rule_engine import apply_rules


def test_rule_boundary_and_saturation():
    d = pd.DataFrame({'signal': [.49, .5, .9]})
    cfg = {'rules': [{'feature': 'signal', 'threshold': .5, 'weight': 150}]}
    assert apply_rules(d, cfg).tolist() == [0, 1, 1]
    assert d.triggered_rule_count.tolist() == [0, 1, 1]
