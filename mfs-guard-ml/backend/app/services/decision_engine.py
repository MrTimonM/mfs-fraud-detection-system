"""Fuses calibrated ML probability with rule score. All cut-offs come from thresholds.json."""
from __future__ import annotations

ACTIONS = ['APPROVE', 'APPROVE_AND_MONITOR', 'STEP_UP_AUTH', 'TEMPORARY_HOLD', 'REJECT_AND_FREEZE']
SEVERITY = {a: i for i, a in enumerate(ACTIONS)}
CONSEQUENTIAL = {'TEMPORARY_HOLD', 'REJECT_AND_FREEZE'}


def _band(value: float, cuts: dict) -> str:
    if value >= cuts['hold']:
        return 'TEMPORARY_HOLD'
    if value >= cuts['step_up']:
        return 'STEP_UP_AUTH'
    if value >= cuts['monitor']:
        return 'APPROVE_AND_MONITOR'
    return 'APPROVE'


class DecisionEngine:
    def __init__(self, thresholds: dict):
        self.t = thresholds

    def ml_action(self, p: float) -> str:
        return _band(p, self.t['ml_probability'])

    def rule_action(self, score: float) -> str:
        return _band(score, self.t['rule_score'])

    def decide(self, probability: float | None, rule_score: float, blacklisted: bool) -> dict:
        reasons = []
        rule_act = self.rule_action(rule_score)
        if probability is None:
            ml_act, mode = None, 'DEGRADED'
            candidates = [rule_act]
            reasons.append('ML unavailable: rules-only decision')
        else:
            ml_act, mode = self.ml_action(probability), 'NORMAL'
            candidates = [ml_act, rule_act]
            req = self.t['reject_requires']
            ml_cut = self.t['ml_probability'][req['ml_at_least']]
            if probability >= ml_cut and rule_score >= req['rule_score_at_least']:
                candidates.append('REJECT_AND_FREEZE')
                reasons.append('ML >= hold and rule score >= reject threshold')
        if blacklisted:
            candidates.append('REJECT_AND_FREEZE')
            reasons.append('blacklisted device or agent')
        final = max(candidates, key=SEVERITY.get)
        return {'decision': final, 'ml_action': ml_act, 'rule_action': rule_act,
                'system_mode': mode, 'requires_human_review': final in CONSEQUENTIAL,
                'decision_reasons': reasons}
