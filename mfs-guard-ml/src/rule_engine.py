"""Configurable deterministic signals, never used as standalone ML features."""
import numpy as np


def apply_rules(d, cfg):
    scores = np.zeros(len(d))
    count = np.zeros(len(d), dtype='int16')
    severity = np.zeros(len(d), dtype='int16')
    for rule in cfg['rules']:
        hit = d[rule['feature']].to_numpy() >= rule['threshold']
        scores += hit * rule['weight']
        count += hit
        severity = np.maximum(severity, hit * rule['weight'])
    d['rule_risk_score'] = np.minimum(100, scores)
    d['triggered_rule_count'], d['highest_rule_severity'] = count, severity
    return d['rule_risk_score'].to_numpy() / 100
