"""Latent episodes, with overlapping legitimate stress events and label noise.

Scenario IDs stay internal: only resulting observable events become features.
"""
import numpy as np

FRAUD_TYPES = np.array(['LEGITIMATE', 'ACCOUNT_TAKEOVER', 'MULE_ACTIVITY',
    'VELOCITY_FRAUD', 'SUBTLE_FRAUD', 'DEVICE_FRAUD', 'SOCIAL_ENGINEERING',
    'AGENT_FRAUD', 'CASH_OUT_ABUSE', 'SIM_SWAP_PATTERN'])


def episodes(n, rng):
    active = rng.random(n) < .064
    scenario = np.where(active, rng.choice(np.arange(1, 10), n,
        p=[.18, .13, .12, .14, .1, .12, .06, .08, .07]), 0)
    # A risky episode is not necessarily fraudulent; legitimate stress overlaps.
    stress = (rng.random(n) < .07) & ~active
    observed = np.where(stress, rng.integers(1, 10, n), scenario)
    label = (active & (rng.random(n) < .76)) | (~active & (rng.random(n) < .0015))
    subtype = np.where(label, np.where(scenario > 0, scenario, 4), 0)
    return observed, subtype
